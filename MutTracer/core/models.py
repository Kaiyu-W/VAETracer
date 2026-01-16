import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import QuantileTransformer


class FlexibleVAE(nn.Module):
    """
    Lightweight variational module for stochastic latent sampling.

    This module parameterizes a Gaussian latent distribution via learnable
    mean and log-variance projections and applies the reparameterization
    trick to enable gradient-based optimization. A learnable temperature
    parameter is included to modulate sampling smoothness.
    """
    def __init__(self, latent_dim):
        super().__init__()
        self.mu = nn.Sequential(
            nn.Linear(256, latent_dim), 
            nn.LayerNorm(latent_dim))
        self.logvar = nn.Sequential(
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def reparameterize(self, mu, logvar):
        logvar = torch.clamp(logvar, min=-5, max=3)  
        std = torch.exp(0.5 * logvar)

        return mu + torch.randn_like(std) * (std + 1e-3)  


class TimeAwarePredictor(nn.Module):
    """
    Time-aware feature modulation module.

    This module encodes discrete time indices into learnable embeddings
    and injects temporal information into latent representations by
    additive fusion, enabling time-dependent feature adjustment.
    """
    def __init__(self, hidden_dim, max_time=5):
        super().__init__()
        self.time_embed = nn.Embedding(max_time, hidden_dim)
        
    def forward(self, x, times):
        time_feat = self.time_embed(times)  # [T, D]
        return x + time_feat.unsqueeze(0)  


class BidirectionalPredictor(nn.Module):
    """
    Bidirectional temporal predictor for joint latent trajectory modeling.

    This model integrates transcriptional and auxiliary latent representations
    using modality-specific projections, time-aware embeddings, and
    bidirectional recurrent networks. Forward and backward temporal contexts
    are fused to generate stochastic latent predictions via a variational
    bottleneck, with adaptive weighting across modalities and time steps.
    """
    def __init__(self, zt_dim, zxt_dim, hidden_dim=128):
        super().__init__()
        self.zt_dim = zt_dim
        self.zxt_dim = zxt_dim
        self.total_dim = zt_dim + zxt_dim 
        self.time_encoder = TimeAwarePredictor(hidden_dim)

        self.zt_proj = nn.Sequential(
            nn.Linear(zt_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)  
            )
        self.zxt_proj = nn.Sequential(
            nn.Linear(zxt_dim, hidden_dim//2), 
            nn.LayerNorm(hidden_dim//2)
        )
        self.forward_lstm = nn.LSTM(hidden_dim + hidden_dim//2, hidden_dim, num_layers=2,batch_first=True,dropout=0.1)
        self.backward_lstm = nn.LSTM(hidden_dim + hidden_dim//2, hidden_dim,num_layers=2, batch_first=True,dropout=0.1)
        self.traj_diff = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.total_dim)
        )
        self.weight_net = nn.Sequential(
            nn.Linear(hidden_dim*2, 2),
            nn.Softmax(dim=-1)
        )

        self.vae = FlexibleVAE(latent_dim=self.total_dim)
        self.zt_head = nn.Linear(zt_dim, zt_dim) 
        self.zxt_head = nn.Linear(zxt_dim, zxt_dim)  

        self.zt_proj_loss = nn.Linear(hidden_dim, zt_dim)  
        self.zxt_proj_loss = nn.Linear(hidden_dim//2, zt_dim) 
        self.temp = nn.Parameter(torch.tensor(1.0))
        self.time_encoder = nn.Sequential(
            nn.Embedding(100, hidden_dim//4),  
            nn.Linear(hidden_dim//4, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.zt_time_proj = nn.Linear(hidden_dim//4, hidden_dim)
        self.zxt_time_proj = nn.Linear(hidden_dim//4, hidden_dim//2)
        self.time_embed = nn.Sequential(
            nn.Embedding(100, hidden_dim//4), 
            nn.Linear(hidden_dim//4, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.weight_net = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        self.hidden_dim = hidden_dim 
        self.temporal_weight = nn.Sequential(
            nn.Linear(2, hidden_dim//4), 
            nn.LayerNorm(hidden_dim//4),
            nn.Linear(hidden_dim//4, 2),
            nn.Sigmoid() 
        )
        self.tsne_proj = nn.Sequential(
            nn.Linear(zt_dim + zxt_dim, hidden_dim//2),
            nn.Tanh()
        )
        
        self.fusion_weight = nn.Sequential(
            nn.Linear(hidden_dim*2 + 2, hidden_dim),  
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        self.delta_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim//4), 
            nn.LayerNorm(hidden_dim//4),
            nn.LeakyReLU(0.1)
        )
        self.direction_embed = nn.Embedding(2, hidden_dim//4)

    def forward(self, zt_seq, zxt_seq,time_indices, delta_t=None):
        batch_size, seq_len = zt_seq.shape[:2]
        time_feat = self.time_embed(time_indices.long()) 
        if time_feat.dim() == 2:  # [B,D]
            time_feat = time_feat.unsqueeze(1)  # [B,1,D]
        
        if delta_t is not None:
            delta_feat = self.delta_encoder(delta_t.unsqueeze(-1))  # [B,T,D//4]
            if delta_feat.dim() == 2:  # [B,D//4]
                delta_feat = delta_feat.unsqueeze(1)  # [B,1,D//4]

            time_feat = torch.cat([
                time_feat[:, :, :self.hidden_dim*3//4], 
                delta_feat
            ], dim=-1)
        time_feat = time_feat[:, :, :self.hidden_dim]

        zt_proj = self.zt_proj(zt_seq) + time_feat
        zxt_proj = self.zxt_proj(zxt_seq) + time_feat[:, :, :self.zxt_proj[0].out_features]
        
        combined = torch.cat([zt_proj, zxt_proj], dim=-1)  # [B,T,H//2+H//4]
        
        f_out, _ = self.forward_lstm(combined)
        b_out, _ = self.backward_lstm(torch.flip(combined, dims=[1]))
        b_out = torch.flip(b_out, dims=[1])

        def smooth_sequence(x, alpha=0.7):
            x_prev = torch.zeros_like(x)
            x_prev[:, 1:] = x[:, :-1]
            return alpha * x + (1 - alpha) * x_prev

        f_out = smooth_sequence(f_out, alpha=0.7)
        b_out = smooth_sequence(b_out.flip(dims=[1]), alpha=0.7).flip(dims=[1])

        fused = torch.cat([
            f_out[:, :, :self.hidden_dim],  
            b_out[:, :, :self.hidden_dim] 
        ], dim=-1)  # -> [B,T,hidden_dim*2] 
    
        mu = self.vae.mu(fused)
        logvar = self.vae.logvar(fused)
        z_pred = self.vae.reparameterize(mu, logvar)
        weights = self.weight_net(fused)  # [B,T,2]
        
        if time_indices.dim() == 1:
            time_indices = time_indices.unsqueeze(0)  

        device = weights.device
        time_factor = torch.arange(
            weights.size(1), 
            device=device
        ).float() / weights.size(1)  
        
        modulated = weights * (0.9 + 0.1 * time_factor.unsqueeze(0).unsqueeze(-1))  # [1,T,1]
        weights = modulated / modulated.sum(dim=-1, keepdim=True)  
        zt_weight = weights[..., 0].unsqueeze(-1)  # [B,T,1]
        zxt_weight = weights[..., 1].unsqueeze(-1)  # [B,T,1]

        zt_loss_proj = self.zt_proj_loss(zt_proj)  # [B,T,64]
        zxt_loss_proj = self.zxt_proj_loss(zxt_proj)  # [B,T,64]
    
        zt_pred = zt_weight * self.zt_head(z_pred[..., :self.zt_dim])  
        zxt_pred = zxt_weight * self.zxt_head(z_pred[..., self.zt_dim:]) 
       
        return zt_pred, zxt_pred, mu, logvar,zt_loss_proj,zxt_loss_proj  

        
class EarlyStopper:
    """
    Early stopping utility for training stabilization.

    This class monitors a scalar loss value across iterations and triggers
    early termination when no sufficient improvement is observed for a
    predefined number of steps, helping prevent overfitting.
    """
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_loss = float('inf')

    def __call__(self, current_loss):
        if current_loss < (self.min_loss - self.min_delta):
            self.min_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def align_zxt_features(split_preds, original_zxt_dict, real_times, pred_times):
    """
    Align predicted auxiliary latent features to reference distributions.

    This function applies distribution alignment to predicted zxt features
    using quantile normalization based on observed reference time points.
    For unseen time points, aligned features are obtained by weighted
    interpolation from the closest available reference distributions.
    """
    for t in real_times:
        if t in split_preds and t in original_zxt_dict:
            split_preds[t]['zxt_pred'] = QuantileTransformer(
                n_quantiles=1000, 
                output_distribution='normal'
            ).fit(original_zxt_dict[t]).transform(split_preds[t]['zxt_pred'])

    for t in pred_times:
        if t not in split_preds:
            continue
            
        closest_times = sorted(
            real_times, 
            key=lambda x: abs(x - t)
        )[:2] 

        aligned = np.zeros_like(split_preds[t]['zxt_pred'])
        total_weight = 0
        
        for ref_t in closest_times:
            if ref_t not in original_zxt_dict:
                continue
                
            weight = np.exp(-abs(t - ref_t))  
            ref_feat = original_zxt_dict[ref_t]
            
            qt = QuantileTransformer(n_quantiles=1000, output_distribution='normal')
            qt.fit(ref_feat)
            aligned += weight * qt.transform(split_preds[t]['zxt_pred'])
            total_weight += weight
        
        if total_weight > 0:
            split_preds[t]['zxt_pred'] = aligned / total_weight
    
    return split_preds
