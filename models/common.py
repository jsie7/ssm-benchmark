import torch
from torch import nn
import torch.nn.functional as F

# reference implementation of the matching layer used in the LRA retrieval task
# https://github.com/google-research/long-range-arena/blob/main/lra_benchmarks/models/layers/common_layers.py#L197
class MATCH(nn.Module):
    def __init__(self, input_dim, mlp_dim, output_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, mlp_dim)
        self.middle = nn.Linear(mlp_dim, int(mlp_dim//2))
        self.decoder = nn.Linear(int(mlp_dim//2), output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.encoder(x)
        x = self.activation(x)
        x = self.middle(x)
        x = self.activation(x)
        x = self.decoder(x)
        return x

# reference implementation of the transformer MLP
# https://github.com/google-research/long-range-arena/blob/main/lra_benchmarks/models/layers/common_layers.py#L144
class MLP(nn.Module):
    def __init__(self, input_dim, mlp_dim, output_dim=None, dropout=0.0):
        super().__init__()
        self.output_dim = input_dim if output_dim is None else output_dim
        self.encoder = nn.Linear(input_dim, mlp_dim)
        self.decoder = nn.Linear(mlp_dim, self.output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.encoder(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.decoder(x)
        x = self.dropout(x)
        return x

# reference implementation of the classifier head implemented in LRA
# https://github.com/google-research/long-range-arena/blob/main/lra_benchmarks/models/layers/common_layers.py#L166
class ClassifierHead(nn.Module):
    def __init__(self, input_dim, mlp_dim, num_classes, pooling):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.pooling = pooling
        if self.mlp_dim != 0:
            self.encoder = nn.Linear(input_dim, mlp_dim)
            self.decoder = nn.Linear(mlp_dim, num_classes)
            self.activation = nn.ReLU()
        
    def forward(self, x):
        # pooling
        if self.pooling in ["mean"]:
            x = torch.mean(x, dim=1)
        elif self.pooling in ["max"]:
            x = torch.max(x, dim=1)[0]
        elif self.pooling in ["sum"]:
            x = torch.sum(x, dim=1)
        elif self.pooling in ["cls"]: # if classifier scalar is learnt concurrently
            x = x[:,0,:]
        else:
            x = x # no pooling
        
        if self.mlp_dim != 0:
            x = self.encoder(x)
            x = self.activation(x)
            x = self.decoder(x)
        return x

class TokenEmbeddings(nn.Module):
    def __init__(
        self,
        embed_dim,
        vocab_size,
        max_position_embeddings,
        padding_idx=None,
        word_embed_proj_dim=None,
        learnable: bool = True,
        device='cuda',
        dtype='torch.float32',
    ):
        """
        GPT-2 Learnable Token and Position Embeddings.
        If max_position_embeddings <= 0, there's no position embeddings
        We embed to word_embe_proj_dim dimension then project up to embed_dim
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        if word_embed_proj_dim is None:
            self.word_embeddings = nn.Embedding(
                vocab_size, embed_dim, padding_idx=padding_idx
            )
            self.project_in = None
        else:
            self.word_embeddings = nn.Embedding(
                vocab_size,
                word_embed_proj_dim,
                padding_idx=padding_idx,
            )
            self.project_in = nn.Linear(
                word_embed_proj_dim, embed_dim, bias=False
            )
        if not learnable:
            self.word_embeddings.weight.requires_grad = False

        self.max_position_embeddings = max_position_embeddings
        if self.max_position_embeddings > 0:
            self.position_embeddings = nn.Embedding(
                max_position_embeddings, embed_dim
            )
    
    def forward(self, input_ids, position_ids=None):
        """
        input_ids: (batch, seqlen)
        position_ids: (batch, seqlen)
        """
        batch_size, seqlen = input_ids.shape[:2]
        embeddings = self.word_embeddings(input_ids)
        if self.project_in is not None:
            embeddings = self.project_in(embeddings)
        if self.max_position_embeddings > 0:
            if position_ids is None:
                position_ids = torch.arange(
                    seqlen, dtype=torch.long, device=self.device
                )
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = embeddings + position_embeddings
        return embeddings
