import math
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
from einops import rearrange, repeat
from mamba_ssm import Mamba as MambaLayer
from mamba_ssm.modules.mamba2_simple import Mamba2Simple as Mamba2Layer

# imports for Mamba-2 & SSD
try:
    from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated, LayerNorm
except ImportError:
    RMSNormGated, LayerNorm = None, None
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined

from models import MATCH

class GLU(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, input_dim * 2)
    def forward(self, x):
        out = self.linear(x)
        return out[:, :, :x.shape[2]] * torch.sigmoid(out[:, :, x.shape[2]:])

class SSD(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        expand=1, # set this to 1 for Neurips 2024 experiments
        headdim=32,
        ngroups=1, # set this to 1 for Neurips 2024 experiments
        A_init_range=(1, 16),
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        dt_limit=(0.0, float("inf")),
        learnable_init_states=False,
        bias=False,
        # Fused kernel and sharding options
        chunk_size=256,
        layer_idx=None,  # Absorb kwarg for general module
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.headdim = headdim
        self.ngroups = ngroups
        assert self.d_inner % self.headdim == 0
        self.nheads = self.d_inner // self.headdim
        # for Neurips 2024 experiments we want nheads == 1
        #assert self.nheads == 1, "Number of heads is greater than 1!"
        self.dt_limit = dt_limit
        self.learnable_init_states = learnable_init_states
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx

        # Order: [x, B, C, dt]
        d_in_proj = self.d_inner + 2 * self.ngroups * self.d_state + self.nheads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=bias, **factory_kwargs)

        if self.learnable_init_states:
            self.init_states = nn.Parameter(torch.zeros(self.nheads, self.headdim, self.d_state, **factory_kwargs))
            self.init_states._no_weight_decay = True

        # Initialize log dt bias
        dt = torch.exp(
            torch.rand(self.nheads, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        # Just to be explicit. Without this we already don't put wd on dt_bias because of the check
        # name.endswith("bias") in param_grouping.py
        self.dt_bias._no_weight_decay = True

        # A parameter
        assert A_init_range[0] > 0 and A_init_range[1] >= A_init_range[0]
        A = torch.empty(self.nheads, dtype=torch.float32, device=device).uniform_(*A_init_range)
        A_log = torch.log(A).to(dtype=dtype)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.nheads, device=device))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, u, seq_idx=None):
        """
        u: (B, L, D)
        Returns: same shape as u
        """
        batch, seqlen, dim = u.shape

        xbcdt = self.in_proj(u)  # (B, L, d_in_proj)
        A = -torch.exp(self.A_log)  # (nheads) or (d_inner, d_state)
        initial_states=repeat(self.init_states, "... -> b ...", b=batch) if self.learnable_init_states else None
        dt_limit_kwargs = {} if self.dt_limit == (0.0, float("inf")) else dict(dt_limit=self.dt_limit)

        xBC, dt = torch.split(
            xbcdt, [self.d_inner + 2 * self.ngroups * self.d_state, self.nheads], dim=-1
        )
        dt = F.softplus(dt + self.dt_bias)  # (B, L, nheads)

        # Split into 3 main branches: X, B, C
        # These correspond to V, K, Q respectively in the SSM/attention duality
        x, B, C = torch.split(xBC, [self.d_inner, self.ngroups * self.d_state, self.ngroups * self.d_state], dim=-1)
        y = mamba_chunk_scan_combined(
            rearrange(x, "b l (h p) -> b l h p", p=self.headdim),
            dt,
            A,
            rearrange(B, "b l (g n) -> b l g n", g=self.ngroups),
            rearrange(C, "b l (g n) -> b l g n", g=self.ngroups),
            chunk_size=self.chunk_size,
            D=self.D,
            z=None,
            seq_idx=seq_idx,
            initial_states=initial_states,
            **dt_limit_kwargs,
        )
        y = rearrange(y, "b l h p -> b l (h p)")

        out = self.out_proj(y)
        return out

class MambaBlock(torch.nn.Module):
    def __init__(self, version, hidden_dim, state_dim, conv_dim, expansion, dropout, glu, norm, prenorm):
        super().__init__()
        if version == "mamba1":
            self.mamba = MambaLayer(d_model=hidden_dim, d_state=state_dim, d_conv=conv_dim, expand=expansion)
        elif version == "mamba2":
            self.mamba = SSD(d_model=hidden_dim, d_state=state_dim, expand=expansion, headdim=hidden_dim)
        else:
            raise RuntimeError("Non supported version")
        if glu:
            self.glu = GLU(hidden_dim)
        else:
            self.glu = None
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        if norm in ["layer"]:
            self.norm = nn.LayerNorm(hidden_dim)
        elif norm in ["batch"]:
            # TODO: add batch norm
            raise RuntimeError("dimensions don't agree for batch norm to work")
            self.norm = nn.BatchNorm1d(hidden_dim)
        self.prenorm = prenorm
        self.lambd = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        skip = self.lambd*x
        if self.prenorm:
            x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(self.activation(x))
        if self.glu is not None:
            x = self.glu(x)
        x = self.dropout(x)
        x = x + skip
        if not self.prenorm:
            x = self.norm(x)
        return x
    
class Mamba(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        version = cfg["version"]
        num_blocks = cfg["num_layers"]
        input_dim = cfg["input_dim"]
        output_dim = cfg["output_dim"]
        hidden_dim = cfg["hidden_dim"]
        state_dim = cfg["state_dim"]
        conv_dim = cfg["conv_dim"]
        expansion = cfg["expansion"]
        dropout = cfg["dropout"]
        glu = cfg["glu"]
        norm = cfg["norm"]
        prenorm = cfg["prenorm"]
        dual = cfg["dual"]
        pooling = cfg["pooling"]

        self.linear_encoder = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.Sequential(*[MambaBlock(version, hidden_dim, state_dim, conv_dim, expansion, dropout, glu, norm, prenorm) for _ in range(num_blocks)])
        self.linear_decoder = nn.Linear(hidden_dim, output_dim)
        self.pooling = pooling
        self.softmax = nn.LogSoftmax(dim=1)
        self.dual = dual
        if dual:
            self.match = MATCH(output_dim*2, output_dim)

    def forward(self, x):
        x = self.linear_encoder(x)
        x = self.blocks(x)
        if self.pooling in ["mean"]:
            x = torch.mean(x, dim=1)
        elif self.pooling in ["max"]:
            x = torch.max(x, dim=1)[0]
        elif self.pooling in ["last"]:
            x = x[:,-1,:]
        else:
            x = x # no pooling
        x = self.linear_decoder(x)
        if self.dual:
            (x1, x2) = torch.split(x, int(x.shape[0]/2))
            x = self.match(torch.concatenate((x1, x2), dim=1))
        return torch.softmax(x, dim=1)
