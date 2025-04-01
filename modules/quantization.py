import torch
from torch import nn
from torch.nn import functional as F
from modules.normalize import L2NormalizationLayer
from modules.loss import QuantizeLoss
from sklearn.cluster import KMeans
from distributions.gumbel import gumbel_softmax_sample

class Quantization(nn.Module):
    def __init__(self, 
                 latent_dim, 
                 n_embed, 
                 k_means_init = True, 
                 codebook_normalization = False, 
                 sim_vq = False, 
                 beta_commit = 0.25,
                 forward_mode = 'STE',
                 ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.n_embed = n_embed
        self.k_means_init = k_means_init
        self.codebook_normalization = codebook_normalization
        self.sim_vq = sim_vq
        self.beta_commit = beta_commit
        
        self.embedding = nn.Embedding(n_embed, latent_dim)
        self.kmeans_initted = False
        self.forward_mode = forward_mode
        
        self.out_proj = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim, bias=False) if sim_vq else nn.Identity(),
            L2NormalizationLayer(dim=-1) if self.codebook_normalization else nn.Identity()
        )
        
        self.quantize_loss = QuantizeLoss(self.beta_commit)
        self._init_weights()

    @property
    def weight(self):
        return self.embedding.weight

    @property
    def device(self):
        return self.embedding.weight.device
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight)
    
    @torch.no_grad()
    def _kmeans_init(self, x):
        
        x = x.view(-1, self.latent_dim).cpu().numpy()
        kmeans = KMeans(n_clusters=self.n_embed, n_init=10, max_iter=300)
        kmeans.fit(x)
        
        self.embedding.weight.copy_(torch.from_numpy(kmeans.cluster_centers_).to(self.device))
        self.kmeans_initted = True
    
    def get_item_embeddings(self, item_ids):
        return self.out_proj(self.embedding(item_ids))
    
    def forward(self, x, temperature):
        
        if self.kmeans_initted and not self.kmeans_initted:
            self._kmeans_init(x=x)

        codebook = self.out_proj(self.embedding.weight)
        
        dist = (
            (x**2).sum(axis=1, keepdim=True) +
            (codebook.T**2).sum(axis=0, keepdim=True) -
            2 * x @ codebook.T
        )
        
        #probs = F.softmax(-dist, dim=1)
        #ids = torch.multinomial(probs, num_samples=1).squeeze(1)
        _, ids = (dist.detach()).min(axis=1)

        if self.training:
            if self.forward_mode == 'Gumbel':
                weights = gumbel_softmax_sample(
                    -dist, temperature=temperature, device=self.device
                )
                emb = weights @ codebook
                emb_out = emb
            else: # should be STE, but in case use STE as fallback
                emb = self.get_item_embeddings(ids)
                emb_out = x + (emb - x).detach()
            
            loss = self.quantize_loss(query=x, value=emb)
        else:
            emb_out = self.get_item_embeddings(ids)
            loss = self.quantize_loss(query=x, value=emb_out)

        return emb_out, ids, loss