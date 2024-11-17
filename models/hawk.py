# Adapted from Hippogriff, see LICENCE

import torch
from torch import nn
from torch.nn.functional import softplus, gelu
from accelerated_scan.warp import scan

from models import MATCH

def init_forget_gate(size, a=0.001, b=0.1, lo=-9, hi=-4.323):
    x = torch.log(torch.expm1(torch.linspace(a, b, size)))
    x = (x - x.min()) / (x.max() - x.min())
    x = x * abs(hi-lo) + lo
    return x

class RMSNorm(nn.Module):
    def __init__(self, *, dim):
        super().__init__()
        self.scale = dim**-0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        x = x / x.norm(p=2, dim=-1, keepdim=True)
        return self.gamma / self.scale * x
    
class GatedMLP(nn.Module):
    def __init__(self, *, dim=256, expansion_factor=2):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.grow = torch.nn.Linear(dim, 2 * hidden, bias=False)
        self.shrink = torch.nn.Linear(hidden, dim, bias=False)

        with torch.no_grad():
            self.grow.weight.normal_(std=dim**-0.5)
            self.shrink.weight.normal_(std=hidden**-0.5)

    def forward(self, x):
        gate, x = self.grow(x).chunk(2, dim=-1)
        x = gelu(gate) * x
        return self.shrink(x)

class RGLRU(nn.Module):
    def __init__(self, dim=256, expansion_factor=2, kernel_size=4, init="uniform"):
        super().__init__()
        hidden = int(dim * expansion_factor)
        self.input = nn.Linear(dim, 2*hidden, bias=False)
        self.conv = nn.Conv1d(in_channels=hidden, out_channels=hidden, bias=True,
                              kernel_size=kernel_size, groups=hidden, padding=kernel_size-1)
        self.gates = nn.Linear(hidden, 2*hidden, bias=True)
        if init in ["uniform"]:
            self.forget_base = nn.Parameter(init_forget_gate(hidden))
        elif init in ["exp"]:
            self.forget_base = nn.Parameter(torch.linspace(-4.323, -9, hidden))
        else:
            raise RuntimeError("Invalid init option {0}".format(init))
        self.output = nn.Linear(hidden, dim, bias=False)
        self.alpha_log_scale = nn.Parameter(-8 * torch.ones(1), requires_grad=False)

        with torch.no_grad():
            self.input.weight.normal_(std=dim**-0.5)
            self.gates.weight.normal_(std=hidden**-0.5)
            self.output.weight.normal_(std=hidden**-0.5)

    def forward(self, x):
        _, T, _ = x.shape
        gate, x = self.input(x).chunk(2, dim=-1)
        x = self.conv(x.mT)[..., :T].mT

        # RG-LRU: linear recurrent unit with input-dependent gating
        forget, input = self.gates(x).chunk(2, dim=-1)
        alpha = (self.alpha_log_scale * softplus(self.forget_base) * forget.sigmoid()).exp()
        beta = (1 - alpha**2 + 1e-6).sqrt()
        x = beta * input.sigmoid() * x

        h = scan(alpha.mT.contiguous(), x.mT.contiguous()).mT
        x = self.output(gelu(gate) * h)
        return x

class SELRU(nn.Module):
    def __init__(self, d=256, n=64, kernel_size=4, init="uniform"):
        super().__init__()
        self.d = d
        self.n = n
        self.input = nn.Linear(d, 2*d, bias=False)
        self.conv = nn.Conv1d(in_channels=d, out_channels=d, bias=True,
                              kernel_size=kernel_size, groups=d, padding=kernel_size-1)
        self.W_bc = nn.Linear(d, 2*n, bias=True)
        self.W_lambda = nn.Linear(d, d, bias=True)
        if init in ["uniform"]:
            self.A = nn.Parameter(init_forget_gate(d).repeat(n,1))
        elif init in ["exp"]:
            self.A = nn.Parameter(torch.linspace(-4.323, -9, d).repeat(n,1))
        else:
            raise RuntimeError("Invalid init option {0}".format(init))
        self.output = nn.Linear(d, d, bias=False)

        with torch.no_grad():
            self.input.weight.normal_(std=d**-0.5)
            self.W_bc.weight.normal_(std=n**-0.5)
            self.W_lambda.weight.normal_(std=d**-0.5)
            self.output.weight.normal_(std=d**-0.5)

    def forward(self, x):
        B, L, _ = x.shape
        skip, x = self.input(x).chunk(2, dim=-1)
        x = self.conv(x.mT)[..., :L].mT

        ### state expanded RG-LRU
        b, c = self.W_bc(x).chunk(2, dim=-1)
        Lambda = self.W_lambda(x)
        Lambda = torch.einsum('bld,nd->blnd', Lambda.sigmoid(), -8*softplus(self.A)).exp()
        bu = torch.einsum('bln,bld->blnd', b, x)
        bu = (1 - Lambda**2 + 1e-6).sqrt() * bu

        # change view for accelerated scan
        Lambda = Lambda.view(B,L,-1)
        bu = bu.view(B,L,-1)

        h = scan(Lambda.mT.contiguous(), bu.mT.contiguous()).mT

        # reshape back to original size
        h = h.reshape(B,L,self.n,self.d)

        x = torch.einsum('bln,blnd->bld', c, h)

        x = self.output(gelu(skip)*x)
        return x

class HawkBlock(nn.Module):
    def __init__(self, dim, expansion, gmlp_expansion, kernel_size, dropout, init):
        super().__init__()
        self.hawk_norm = RMSNorm(dim=dim)
        self.hawk = RGLRU(dim, expansion, kernel_size, init)
        self.hawk_gmlp_norm = RMSNorm(dim=dim)
        self.hawk_gmlp = GatedMLP(dim=dim, expansion_factor=gmlp_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.hawk(self.hawk_norm(x))
        x = self.dropout(x)
        x = x + self.hawk_gmlp(self.hawk_gmlp_norm(x))
        x = self.dropout(x)
        return x

class SEaHawkBlock(nn.Module):
    def __init__(self, dim, state_dim, gmlp_expansion, kernel_size, dropout, init):
        super().__init__()
        self.hawk_norm = RMSNorm(dim=dim)
        self.hawk = SELRU(dim, state_dim, kernel_size, init)
        self.hawk_gmlp_norm = RMSNorm(dim=dim)
        self.hawk_gmlp = GatedMLP(dim=dim, expansion_factor=gmlp_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.hawk(self.hawk_norm(x))
        x = self.dropout(x)
        x = x + self.hawk_gmlp(self.hawk_gmlp_norm(x))
        x = self.dropout(x)
        return x
    
class Hawk(nn.Module):
    def __init__(self, num_blocks, input_dim, output_dim, hidden_dim, expansion, gmlp_expansion, kernel_size, dropout, init, dual, pooling):
        super().__init__()
        self.linear_encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.blocks = torch.nn.Sequential(*[HawkBlock(hidden_dim, expansion, gmlp_expansion, kernel_size, dropout, init) for _ in range(num_blocks)])
        self.linear_decoder = torch.nn.Linear(hidden_dim, output_dim)
        self.pooling = pooling
        self.softmax = torch.nn.LogSoftmax(dim=1)
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

class SEaHawk(nn.Module):
    def __init__(self, num_blocks, input_dim, output_dim, hidden_dim, state_dim, gmlp_expansion, kernel_size, dropout, init, dual, pooling):
        super().__init__()
        self.linear_encoder = torch.nn.Linear(input_dim, hidden_dim)
        self.blocks = torch.nn.Sequential(*[SEaHawkBlock(hidden_dim, state_dim, gmlp_expansion, kernel_size, dropout, init) for _ in range(num_blocks)])
        self.linear_decoder = torch.nn.Linear(hidden_dim, output_dim)
        self.pooling = pooling
        self.softmax = torch.nn.LogSoftmax(dim=1)
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
