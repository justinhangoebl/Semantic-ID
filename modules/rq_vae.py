import torch
import torch.nn as nn
from modules.quantization import Quantization
from modules.mlp import MLP
from modules.loss import ReconstructionLoss, CategoricalReconstuctionLoss
from einops import rearrange
from modules.normalize import l2norm

class RQ_VAE(nn.Module):
    def __init__(self,
                 input_dim,
                 latent_dim,
                 hidden_dims,
                 codebook_size,
                 codebook_kmeans=True,
                 codebook_normalization=False,
                 codebook_sim_vq=False,
                 num_layers=3,
                 beta_commit=0.25,
                 forward_mode='STE',
                 num_cat_features=18):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.beta_commit = beta_commit
        self.num_cat_features = num_cat_features
        
        self.residual_quantize_stack = nn.ModuleList(modules=[
            Quantization(
                latent_dim=latent_dim,
                n_embed=codebook_size,
                k_means_init=codebook_kmeans,
                codebook_normalization=i == 0 and codebook_normalization,
                sim_vq=codebook_sim_vq,
                beta_commit=beta_commit,
                forward_mode=forward_mode
            ) for i in range(num_layers)
        ])
        
        self.encoder = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=latent_dim,
            normalize=codebook_normalization,
            dropout=0.1
        )

        self.decoder = MLP(
            input_dim=latent_dim,
            hidden_dims=hidden_dims[-1::-1],
            output_dim=input_dim,
            normalize=True,
            dropout=0.1
        )
        
        self.reconstruction_loss = ReconstructionLoss()
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def get_semantic_ids(
        self,
        x,
        gumbel_t: float = 0.001
    ):
        res = self.encode(x)
        
        quantize_loss = 0
        embs, residuals, sem_ids = [], [], []

        for layer in self.residual_quantize_stack:
            residuals.append(res)
            # emb_out, ids, loss
            quantized = layer(res, temperature=gumbel_t)
            quantize_loss += quantized[2]
            emb, id = quantized[0], quantized[1]
            res = res - emb
            sem_ids.append(id)
            embs.append(emb)

        embeddings = rearrange(embs, "b h d -> h d b")
        residuals = rearrange(residuals, "b h d -> h d b")
        sem_ids = rearrange(sem_ids, "b d -> d b")
        quantize_loss = quantize_loss
        
        return embeddings, residuals, sem_ids, quantize_loss
    
    def forward(self, x, gumbel_t= 0.001):
        quantized = self.get_semantic_ids(x, gumbel_t)
        embs, residuals = quantized[0], quantized[1]
        x_hat = self.decode(embs.sum(axis=-1))
        x_hat = torch.cat([l2norm(x_hat[...,:-self.num_cat_features]), x_hat[...,-self.num_cat_features:]], axis=-1)

        reconstuction_loss = self.reconstruction_loss(x_hat, x)
        rqvae_loss = quantized[3]
        loss = (reconstuction_loss + rqvae_loss).mean()

        with torch.no_grad():
            # Compute debug ID statistics
            embs_norm = embs.norm(dim=1)
            p_unique_ids = (~torch.triu(
                (rearrange(quantized[2], "b d -> b 1 d") == rearrange(quantized[2], "b d -> 1 b d")).all(axis=-1), diagonal=1)
            ).all(axis=1).sum() / quantized[2].shape[0]

        
        loss=loss
        reconstruction_loss=reconstuction_loss.mean()
        rqvae_loss=rqvae_loss.mean()
        embs_norm=embs_norm
        p_unique_ids=p_unique_ids
        
        #z = self.encode(x)
        #x_hat = self.decode(z)
        #
        #loss = self.reconstruction_loss(x_hat, x).mean()
        #reconstruction_loss = loss
        #rqvae_loss = 0
        #embs_norm = 0
        #p_unique_ids = 0
        
        
        return loss, reconstruction_loss, rqvae_loss, embs_norm, p_unique_ids