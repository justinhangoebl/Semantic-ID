from torch import nn
from modules.normalize import L2NormalizationLayer


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.0, normalize=False, eps=1e-12):
        super(MLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.out_dim = output_dim
        
        self.dropout = dropout
        self.eps = eps
        
        self.dims = [input_dim] + hidden_dims + [output_dim]
        
        self.mlp = nn.ModuleList()
        for i, (d_in, d_out) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            self.mlp.append(nn.Linear(d_in, d_out))
            if i < len(self.dims) - 2:
                self.mlp.append(nn.SiLU())
                if dropout > 0.0:
                    self.mlp.append(nn.Dropout(dropout))
        
        self.mlp.append(
            L2NormalizationLayer(dim=-1, eps=self.eps)
            if normalize else nn.Identity())
        self.mlp = nn.Sequential(*self.mlp)
        
    def forward(self, x):
        return self.mlp(x)
