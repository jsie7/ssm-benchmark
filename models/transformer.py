import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math

from models import MATCH, MLP, ClassifierHead, TokenEmbeddings

class SelfAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, qk, v):
        """Implements multihead softmax attention.
        Arguments
        ---------
            qk: Tensor containing the queries and keys. (B, S, 2, H, D)
            v:  Tensor containing the values. (B, S, H, D)
        """
        seqlen = qk.shape[1]
        q, k = qk.unbind(dim=2)
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
        scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
        causal_mask = torch.triu(
            torch.full((seqlen, seqlen), -10000.0, device=scores.device), 1
        )
        scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1, dtype=v.dtype)
        attention = self.dropout(attention)
        output = torch.einsum("bhts,bshd->bthd", attention, v)
        return output

class SelfLinAttention(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, qk, v):
        """Implements multihead linear attention.
        Arguments
        ---------
            qk: Tensor containing the queries and keys. (B, S, 2, H, D)
            v:  Tensor containing the values. (B, S, H, D)
        """
        seqlen = qk.shape[1]
        q, k = qk.unbind(dim=2)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        scores = torch.einsum("bthd,bshd->bhts", q, k)
        mask_mul = torch.tril(
            torch.full((seqlen, seqlen), 1, device=scores.device), 0
        )
        scores = scores * mask_mul.to(dtype=scores.dtype)
        nu = scores.sum(dim=-1)
        attention = torch.div(scores, nu[:,:,:,None])
        attention = self.dropout(attention)
        output = torch.einsum("bhts,bshd->bthd", attention, v)
        return output

class SelfLinAttention2(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, qk, v):
        """Implements multihead linear attention.
        Arguments
        ---------
            qk: Tensor containing the queries and keys. (B, S, 2, H, D)
            v:  Tensor containing the values. (B, S, H, D)
        """
        seqlen = qk.shape[1]
        q, k = qk.unbind(dim=2)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        kv = torch.einsum("bshd,bsht->bshdt",k,v)
        kv = torch.cumsum(kv, dim=1)

        k = torch.cumsum(k, dim=1)
        n = torch.einsum("bshd,bshd -> bsh", q, k)
        n = n.pow(-1)

        output = torch.einsum("bshd,bshdt->bsht",q,kv)
        output = n[:,:,:,None]*output
        return self.dropout(output)
    
class MHA(nn.Module):
    """Multi-head self-attention
    """
    def __init__(
        self,
        d_model: int,
        d_qk: int=None,
        num_heads: int=1,
        lin_att: bool=True,
        dropout: float=0.0,
        bias: bool=True,
        layer_idx: int=None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        if d_qk is None:
            self.d_qk = d_model
        else:
            self.d_qk = d_qk

        self.layer_idx = layer_idx
        self.num_heads = num_heads
        assert (
            self.d_qk % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        assert (
            self.d_model % num_heads == 0
        ), "self.vdim must be divisible by num_heads"
        self.head_dim = self.d_qk // num_heads
        self.v_dim = self.d_model // num_heads
        self.Wqk = nn.Linear(
            d_model, 2 * self.d_qk, bias=bias
        )
        self.Wv = nn.Linear(d_model, d_model, bias=bias)
        if lin_att:
            self.inner_attn = SelfLinAttention2(dropout)
        else:
            self.inner_attn = SelfAttention(dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        """"""
        qk = self.Wqk(x)
        v = self.Wv(x)
        qk = rearrange(
            qk, "... (two h d) -> ... two h d", two=2, d=self.head_dim
        )
        v = rearrange(
            v, "... (h d) -> ... h d", d=self.v_dim
        )
        context = self.inner_attn(qk, v)
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out
    
    def state_size(self, batch_size: int=1, sequence_length: int=2048):
        return 2 * self.d_model * self.d_qk

class SelfNormAttention(nn.Module):
    def __init__(self, norm_fn, dropout=0.0):
        super().__init__()
        if norm_fn in ["exp"]:
            self.norm_fn = lambda x: torch.exp(-x)
        elif norm_fn in ["softplus"]:
            self.norm_fn = lambda x: 1/F.softplus(x)
        elif norm_fn in ["sigmoid"]:
            self.norm_fn = lambda x: 1 + torch.exp(-x)
        else:
            raise RuntimeError("normalization function {0} not implemented!".format(norm_fn))
        self.dropout = nn.Dropout(dropout)

    def forward(self, qk, v, n):
        """Implements the multihead linear attention with normalization.
        Arguments
        ---------
            qk: Tensor containing the queries and keys. (B, S, 2, H, D)
            v:  Tensor containing the values. (B, S, H, D)
            n:  Tensor containing the normalization values. (B, S, H)
        """
        seqlen = qk.shape[1]
        q, k = qk.unbind(dim=2)
        scores = torch.einsum("bthd,bshd->bhts", q, k)
        mask_mul = torch.tril(
            torch.full((seqlen, seqlen), 1, device=scores.device), 0
        )
        scores = scores * mask_mul.to(dtype=scores.dtype)

        n = self.norm_fn(rearrange(n, "b l h -> b h l"))
        attention = n[:,:,:,None]*scores

        attention = self.dropout(attention)
        output = torch.einsum("bhts,bshd->bthd", attention, v)
        return output

class SelfNormAttention2(nn.Module):
    def __init__(self, norm_fn, dropout=0.0):
        super().__init__()
        if norm_fn in ["exp"]:
            self.norm_fn = lambda x: torch.exp(-x)
        elif norm_fn in ["softplus"]:
            self.norm_fn = lambda x: 1/F.softplus(x)
        elif norm_fn in ["sigmoid"]:
            self.norm_fn = lambda x: 1 + torch.exp(-x)
        else:
            raise RuntimeError("normalization function {0} not implemented!".format(norm_fn))
        self.dropout = nn.Dropout(dropout)

    def forward(self, qk, v, n):
        """Implements the multihead linear attention with normalization.
        Arguments
        ---------
            qk: Tensor containing the queries and keys. (B, S, 2, H, D)
            v:  Tensor containing the values. (B, S, H, D)
            n:  Tensor containing the normalization values. (B, S, H)
        """
        q, k = qk.unbind(dim=2)
        kv = torch.einsum("bshd,bsht->bshdt",k,v)
        kv = torch.cumsum(kv, dim=1)
        n = self.norm_fn(n)

        output = torch.einsum("bshd,bshdt->bsht",q,kv)
        output = n[:,:,:,None]*output

        return self.dropout(output)

class MHNA(nn.Module):
    """Multi-head self-attention with normalization
    """
    def __init__(
        self,
        d_model: int,
        d_qk: int=None,
        num_heads: int=1,
        norm_fn: str="exp",
        dropout: float=0.0,
        bias: bool=True,
        layer_idx: int=None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        if d_qk is None:
            self.d_qk = d_model
        else:
            self.d_qk = d_qk

        self.layer_idx = layer_idx
        self.num_heads = num_heads
        assert (
            self.d_qk % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        assert (
            self.d_model % num_heads == 0
        ), "self.vdim must be divisible by num_heads"
        self.head_dim = self.d_qk // num_heads
        self.v_dim = self.d_model // num_heads

        self.Wqk = nn.Linear(
            d_model, 2 * self.d_qk, bias=bias
        )
        self.Wv = nn.Linear(d_model, d_model, bias=bias)
        self.Wn = nn.Linear(d_model, num_heads, bias=bias)
        self.inner_attn = SelfNormAttention2(norm_fn, dropout)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor):
        """"""
        qk = self.Wqk(x)
        v = self.Wv(x)
        n = self.Wn(x)
        qk = rearrange(
            qk, "... (two h d) -> ... two h d", two=2, d=self.head_dim
        )
        v = rearrange(
            v, "... (h d) -> ... h d", d=self.v_dim
        )
        context = self.inner_attn(qk, v, n)
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out
    
    def state_size(self, batch_size: int=1, sequence_length: int=2048):
        return 2 * self.d_model * self.d_qk

class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, cfg):
        super().__init__()
        # remove configs only used at this level
        d_model = hidden_dim
        d_qk = cfg["state_dim"]
        num_heads = cfg["num_heads"]
        att_dropout = cfg["att_dropout"]
        mlp_dim = cfg["mlp_dim"]
        dropout = cfg["dropout"]
        norm = cfg["norm"]
        self.attention_fn = cfg["attention_fn"]

        # attention function
        if self.attention_fn in ["sm-attention"]:
            self.attention = MHA(d_model, d_qk, num_heads,
                                 lin_att=False, dropout=att_dropout)
        elif self.attention_fn in ["lin-attention"]:
            self.attention = MHA(d_model, d_qk, num_heads,
                                 lin_att=True, dropout=att_dropout)
        elif self.attention_fn in ["norm-attention"]:
            norm_fn = cfg["norm_fn"]
            self.attention = MHNA(d_model, d_qk, num_heads, norm_fn,
                                  dropout=att_dropout)
        

        # MLP
        self.mlp = MLP(hidden_dim, mlp_dim, dropout=dropout)

        if norm in ["layer"]:
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            raise RuntimeError("{0} norm not implemented yet!".format(norm))
        self.dropout = nn.Dropout(dropout)
        self.lambd = nn.Parameter(torch.ones(1), requires_grad = True)

    def forward(self, x):
        skip = self.lambd*x
        x = self.norm(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + skip

        y = self.norm(x)
        y = self.mlp(y)
    
        return x + y
    
class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()      
        # remove configs only used at this level
        input_dim = cfg.pop("input_dim")
        output_dim = cfg.pop("output_dim")
        num_layers = cfg.pop("num_layers")
        hidden_dim = cfg.pop("hidden_dim")
        embed = cfg.pop("embedding")
        vocab_size = cfg.pop("vocab_size")
        max_len = cfg.pop("max_pos_embed")
        pooling = cfg.pop("pooling")
        self.dual = cfg.pop("dual")
        self.classify = cfg.pop("classifier")

        mlp_dim = cfg["mlp_dim"]
        norm = cfg["norm"]
        dropout = cfg["dropout"]

        if embed:
            self.encoder = TokenEmbeddings(hidden_dim, vocab_size, max_len)
        else:
            self.encoder = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.Sequential(*[TransformerBlock(hidden_dim, cfg) for _ in range(num_layers)])
        if self.classify:
            self.classifier = ClassifierHead(hidden_dim, mlp_dim, output_dim, pooling)
        else:
            self.decoder = nn.Linear(hidden_dim, output_dim)
        if self.dual:
            self.match = MATCH(output_dim*2, mlp_dim, output_dim)
        if norm in ["layer"]:
            self.norm = nn.LayerNorm(hidden_dim)
        else:
            raise RuntimeError("{0} norm not implemented yet!".format(norm))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.dropout(x)
        x = self.layers(x)
        x = self.norm(x)
        if self.classify:
            x = self.classifier(x)
        else:
            x = self.decoder(x)
        if self.dual:
            (x1, x2) = torch.split(x, int(x.shape[0]/2))
            x = self.match(torch.concatenate((x1, x2), dim=1))
        return torch.softmax(x, dim=1)
