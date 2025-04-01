from torch import nn

class QuantizeLoss(nn.Module):
    def __init__(self, beta_commit = 1.0):
        super().__init__()
        self.beta_commit = beta_commit

    def forward(self, query, value):
        emb_loss = ((query.detach() - value)**2).sum(axis=[-1])
        query_loss = ((query - value.detach())**2).sum(axis=[-1])
        return emb_loss + self.beta_commit * query_loss
    
class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_hat, x):
        return ((x_hat - x)**2).sum(axis=-1)
    
class CategoricalReconstuctionLoss(nn.Module):
    def __init__(self, num_cat_features):
        super().__init__()
        self.reconstruction_loss = ReconstructionLoss()
        self.num_cat_features = num_cat_features
    
    def forward(self, x_hat, x):
        reconstr = self.reconstruction_loss(
            x_hat[:, :-self.num_cat_features],
            x[:, :-self.num_cat_features]
        )
        if self.num_cat_features > 0:
            cat_reconstr = nn.functional.binary_cross_entropy_with_logits(
                x_hat[:, -self.num_cat_features:],
                x[:, -self.num_cat_features:],
                reduction='none'
            ).sum(axis=-1)
            reconstr += cat_reconstr
        return reconstr