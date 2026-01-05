import torch
import torch.nn as nn
from collections import defaultdict

from .baseVAE import (
    Encoder,
    Decoder, 
    VAE, 
    AutoEncoderModel
)
from .zinb_dist import ZeroInflatedNegativeBinomial
from .utils import set_seed, DEVICE
from .typing import Tuple, Optional, Literal


class scVIEncoder(Encoder):
    """
    Encoder network for scVI model, inherits from base Encoder.
    
    This encoder transforms gene expression data into a latent space representation,
    producing both mean (mu) and variance (logvar) for the latent distribution.
    
    Args:
        input_dim (int): Number of input features (genes)
        hidden_dims (list): Dimensions of hidden layers
        z_dim (int): Dimension of the latent space
        activation (str, optional): Activation function. Defaults to 'relu'
            Options: 'relu', 'leakyrelu', 'gelu'
            
    Attributes:
        hidden_layers (nn.Sequential): Sequential container of hidden layers
        fc_mu (nn.Linear): Linear layer for mean of latent distribution
        fc_logvar (nn.Linear): Linear layer for log variance of latent distribution
    """
    def __init__(
            self, input_dim, hidden_dims, z_dim, activation='relu', 
            dropout_rate=0.1, use_batch_norm=False, use_layer_norm=True
        ):
        super().__init__(
            input_dim, hidden_dims, z_dim, model_type='VAE', 
            dropout_rate=dropout_rate, 
            activation=activation,
            use_batch_norm=use_batch_norm, 
            use_layer_norm=use_layer_norm
        )

class scVIDecoder(Decoder):
    """
    Decoder network for scVI model, modified to output ZINB distribution parameters.
    
    This decoder transforms latent representations back to gene expression space,
    but outputs parameters for a zero-inflated negative binomial distribution
    instead of direct reconstructions.
    
    Args:
        z_dim (int): Dimension of the latent space
        hidden_dims (list): Dimensions of hidden layers
        output_dim (int): Number of output features (genes)
        activation (str, optional): Activation function. Defaults to 'relu'
            Options: 'relu', 'leakyrelu', 'gelu'
            
    Attributes:
        hidden_layers (nn.Sequential): Sequential container of hidden layers
        px_scale (nn.Linear): Linear layer for mean parameter of ZINB
        px_r (nn.Linear): Linear layer for dispersion parameter of ZINB
        px_dropout (nn.Linear): Linear layer for dropout parameter of ZINB
    """
    def __init__(
            self, z_dim, hidden_dims, output_dim, activation='none', # decoder should have no activation
            dropout_rate=0, use_batch_norm=False, use_layer_norm=True
        ):
        super().__init__(
            z_dim=z_dim, hidden_dims=hidden_dims, output_dim=output_dim, activation=activation, 
            dropout_rate=dropout_rate, use_batch_norm=use_batch_norm, use_layer_norm=use_layer_norm
        )
        
        # ZINB-specific output layers, from scVI
        # mean gamma
        self.px_scale = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            # nn.Softplus(),
            nn.Softmax(dim=-1), 
            # 1) nn.Softplus(): scVI code default,
            # 2) nn.Softmax(dim=-1): scVI paper method,
            # actually, here px_scale is probability, of which ~[0,1] and sum should be 1, so Softmax fits better
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r = nn.Linear(hidden_dims[-1], output_dim)
        # self.px_r = nn.Parameter(torch.randn(output_dim)) # all genes share one theta

        # dropout
        self.px_dropout = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, z, library):
        """
        Forward pass of the decoder.
        
        Args:
            z (torch.Tensor): Latent space representation
            
        Returns:
            dict: Dictionary containing ZINB parameters
                - 'scale': Mean parameter (μ)
                - 'r': Inverse dispersion parameter
                - 'rate': rate parameter
                - 'dropout': Dropout probability
        """
        # from scVI
        px = self.hidden_layers(z)

        # Dropout
        px_dropout = self.px_dropout(px)

        # Dispersion parameter (theta), gene-specific
        px_r = self.px_r(px) # torch.clamp(, min=0, max=7) # log(1000)~6.9
        # px_r = self.px_r # all genes share one theta

        # Mean parameter (mu)
        px_scale = self.px_scale(px) # torch.clamp(, min=0, max=1)

        # Library scaling
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale

        return {
            'scale': px_scale,
            'r': px_r,
            'rate': px_rate,
            'dropout': px_dropout,
        }

class scVI(VAE):
    """
    Complete scVI model combining encoder and decoder.
    
    This model implements the full scVI architecture, using a VAE framework
    with a zero-inflated negative binomial (ZINB) distribution for gene expression modeling.
    
    Attributes:
        encoder (scVIEncoder): Encoder network
        decoder (scVIDecoder): Decoder network
    """
    def __init__(
        self, input_dim, hidden_dims, z_dim, activation='relu', 
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
    ):
        super().__init__(
            input_dim=input_dim, hidden_dims=hidden_dims, z_dim=z_dim, activation=activation, 
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm
        )

        self.encoder = scVIEncoder(
            input_dim, hidden_dims, z_dim, 
            activation=activation,
            use_batch_norm=self.use_batch_norm_encoder, 
            use_layer_norm=self.use_layer_norm_encoder
        )
        self.decoder = scVIDecoder(
            z_dim, hidden_dims[::-1], input_dim, 
            activation='none', 
            use_batch_norm=self.use_batch_norm_decoder, 
            use_layer_norm=self.use_layer_norm_decoder
        )
    
    def forward(self, x, library):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input gene expression data
            
        Returns:
            tuple:
                - outputs (dict): ZINB parameters from decoder
                - mu (torch.Tensor): Mean of latent distribution
                - logvar (torch.Tensor): Log variance of latent distribution
                - qz (torch.Tensor): Distribution of latent distribution
                - z (torch.Tensor): Latent space representation
        """
        mu, logvar = self.encoder(x)
        qz, z = self.reparameterize(mu, logvar)
        outputs = self.decoder(z, library)
        return x, outputs, mu, logvar, qz, z

class scVIModel(AutoEncoderModel):
    """
    scVI model manager for training and inference.
    
    This class handles the training and evaluation of the scVI model,
    specifically designed for single-cell RNA sequencing data analysis.
    
    Args:
        input_dim (int): Number of input features (genes)
        hidden_dims (list): Dimensions of hidden layers
        z_dim (int): Dimension of the latent space
        num_epochs (int): Number of training epochs
        lr (float, optional): Learning rate. Defaults to 1e-3
        device (str, optional): Device to use. Defaults to DEVICE
        seed (int, optional): Random seed. Defaults to 42
        activation (str, optional): Activation function. Defaults to 'relu'
        use_batch_norm
            Specifies where to use :class:`~torch.nn.BatchNorm1d` in the model. One of the following:

            * ``"none"``: don't use batch norm in either encoder(s) or decoder.
            * ``"encoder"``: use batch norm only in the encoder(s).
            * ``"decoder"``: use batch norm only in the decoder.
            * ``"both"``: use batch norm in both encoder(s) and decoder.

            Note: if ``use_layer_norm`` is also specified, both will be applied (first
            :class:`~torch.nn.BatchNorm1d`, then :class:`~torch.nn.LayerNorm`).
        use_layer_norm
            Specifies where to use :class:`~torch.nn.LayerNorm` in the model. One of the following:

            * ``"none"``: don't use layer norm in either encoder(s) or decoder.
            * ``"encoder"``: use layer norm only in the encoder(s).
            * ``"decoder"``: use layer norm only in the decoder.
            * ``"both"``: use layer norm in both encoder(s) and decoder.

            Note: if ``use_batch_norm`` is also specified, both will be applied (first
            :class:`~torch.nn.BatchNorm1d`, then :class:`~torch.nn.LayerNorm`).
        **kwargs: Additional arguments passed to AutoEncoderModel
        
    Attributes:
        model (scVI): The scVI model instance
        optimizer (torch.optim.Adam): Adam optimizer
        train_metrics (list): List of training metrics
        valid_metrics (list): List of validation metrics
        current_metrics (dict): Current epoch metrics
    """
    def __init__(
        self, input_dim, hidden_dims, z_dim, num_epochs, 
        lr=1e-3, beta=1, device=DEVICE, dtype=torch.float32,
        seed=42, eps=1e-10,
        activation: Literal["relu", "leakyrelu", "gelu"] = 'relu',
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        **kwargs
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            z_dim=z_dim,
            num_epochs=num_epochs,
            lr=lr,
            beta=beta,
            device=device,
            dtype=dtype,
            seed=seed,
            eps=eps,
            activation=activation,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **kwargs
        )
        self.loss = self.loss_function
        self.model = scVI(
            self.input_dim, self.hidden_dims, self.z_dim, 
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
            use_layer_norm=self.use_layer_norm
        ).to(self.device).to(self.dtype)
        self.initialize_weights(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # torch.autograd.set_detect_anomaly(True)

    def _compute_library(self, X):
        library = torch.log(X.sum(1)).unsqueeze(1)
        return library
    
    def _px(self, px_param_dict):
        px = ZeroInflatedNegativeBinomial(
            mu=px_param_dict['rate'],
            theta=px_param_dict['r'].exp(),
            zi_logits=px_param_dict['dropout'],
            scale=px_param_dict['scale'],
        )
        return px

    def _zinb_loss(self, x, px):
        reconst_loss = -px.log_prob(x).sum(-1).sum()
        return reconst_loss
    
    def loss_function(self, x, outputs, mu, logvar, qz, z, *args, **kwargs):
        """
        Calculate the total loss combining ZINB and KL divergence.
        
        Args:
            x (torch.Tensor): Input data
            outputs (dict): ZINB parameters from decoder
            mu (torch.Tensor): Mean of latent distribution
            logvar (torch.Tensor): Log variance of latent distribution
            others
            
        Returns:
            torch.Tensor: Total loss value
        """
        batch_size = x.size(0)
        px = self._px(outputs)
        zinb_loss = self._zinb_loss(x, px) / batch_size
        kl_loss = self._kl_loss(mu, logvar) / batch_size
        # kl_loss = self._kl_loss_by_qz(z, qz) / batch_size
        total_loss = zinb_loss + self.beta * kl_loss

        self.current_metrics = {
            'zinb_loss': zinb_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item()
        }
        
        return total_loss
    
    def _train_epoch(self):
        """
        Train the model for one epoch.
        
        Returns:
            dict: Average metrics for the epoch
        """
        self.model.train()
        total_metrics = defaultdict(float)
        n_samples = 0
        
        for batch in self.data_loader:
            self.optimizer.zero_grad()
            x = batch[0].to(self.device)
            library = self._compute_library(x)

            batch_size = x.size(0)
            n_samples += batch_size

            outs = self.model(x, library) # (x, outputs, mu, logvar, qz, z)
            loss = self.loss(*outs)
            loss.backward()
            self.optimizer.step()
            
            for k, v in self.current_metrics.items():
                total_metrics[k] += v * batch_size
        
        avg_metrics = {k: v/n_samples for k, v in total_metrics.items()}
        self.train_metrics.append(avg_metrics)
        return avg_metrics

    def _sample_from_zinb(self, px):
        """
        Sample from ZINB distribution.
            
        Returns:
            Samples from ZINB distribution (non-negative integers)
        """
        set_seed(self.seed)
        return px.sample()

    def _get_reconstructions(self, use_mu=False):
        """
        Get latent representations and reconstructions for the full dataset.

        Args:
            use_mu (bool): Use Mean Expression Layer output as reconstructed X, default False
            
        Returns:
            (Z, Xhat): embedded layer and reconstructed X
        """
        self.model.eval()
        z_list, xhat_list = [], []
        
        with torch.no_grad():
            for batch in self.data_loader_order:
                x = batch[0].to(self.device)
                library = self._compute_library(x)
                outs = self.model(x, library) # (x, outputs, mu, logvar, qz, z)
                outputs, mu = outs[1], outs[2]
                if use_mu:
                    xhat = outputs['rate']
                else:
                    px = self._px(outputs)
                    xhat = self._sample_from_zinb(px)
                z_list.append(mu.cpu())
                xhat_list.append(xhat.cpu())
        
        # Convert to numpy arrays
        Z = torch.cat(z_list, dim=0).numpy()
        Xhat = torch.cat(xhat_list, dim=0).numpy()
        
        return Z, Xhat