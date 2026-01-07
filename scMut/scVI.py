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
        self,
        input_dim: int,
        hidden_dims: List[int],
        z_dim: int,
        activation: Literal["relu", "leakyrelu", "gelu"] = 'relu',
        dropout_rate: float = 0.1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
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
        activation (str, optional): Activation function. Defaults to 'none'
        
    Attributes:
        hidden_layers (nn.Sequential): Hidden layer stack
        px_scale (nn.Sequential): Output layer for normalized mean (uses Softmax)
        px_r (nn.Linear): Dispersion parameter (r) per gene
        px_dropout (nn.Linear): Dropout probability logits per gene
    """
    def __init__(
        self,
        z_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: Literal["none"] = 'none',
        dropout_rate: float = 0,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
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

    def forward(self, z: torch.Tensor, library: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the decoder.
        
        Args:
            z (torch.Tensor): Latent tensor of shape (batch_size, z_dim)
            library (torch.Tensor): Library size log-sum tensor of shape (batch_size, 1)

        Returns:
            dict: Dictionary containing ZINB parameters:
                - 'scale': Normalized mean (softmax-normalized), shape (batch_size, n_genes)
                - 'r': Dispersion parameter (theta inverse), shape (batch_size, n_genes)
                - 'rate': Final rate λ = exp(library) * scale, shape (batch_size, n_genes)
                - 'dropout': Dropout logits, shape (batch_size, n_genes)
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
        self,
        input_dim: int,
        hidden_dims: List[int],
        z_dim: int,
        activation: Literal["relu", "leakyrelu", "gelu"] = 'relu',
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
    
    def forward(
        self,
        x: torch.Tensor,
        library: torch.Tensor
    ) -> Tuple[
        torch.Tensor,                    # x_input
        Dict[str, torch.Tensor],         # outputs (ZINB params)
        torch.Tensor,                    # mu
        torch.Tensor,                    # logvar
        Optional[torch.distributions.Normal],  # qz
        torch.Tensor                     # z
    ]:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
            library (torch.Tensor): Log-library size tensor of shape (batch_size, 1)

        Returns:
            tuple: Contains:
                - x: Original input
                - outputs: ZINB parameter dictionary from decoder
                - mu: Latent mean
                - logvar: Latent log-variance
                - qz: Approximate posterior distribution
                - z: Sampled latent vector
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
    
    It extends AutoEncoderModel with ZINB-based reconstruction loss
    and library size normalization.

    Example:
        model = scVIModel(
            input_dim=2000,
            hidden_dims=[128],
            z_dim=10,
            num_epochs=200
        )
        model.load_data(X, batch_size=256)
        losses = model.train()
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        z_dim: int,
        num_epochs: int,
        lr: float = 1e-3,
        beta: float = 1.0,
        device: Union[str, torch.device] = DEVICE,
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
        eps: float = 1e-10,
        activation: Literal["relu", "leakyrelu", "gelu"] = 'relu',
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        **kwargs
    ):
        """
        Initialize the scVI model.
        
        Args:
            input_dim: Number of input features (genes).
            hidden_dims: Sizes of hidden layers in encoder/decoder.
            z_dim: Dimensionality of latent space.
            num_epochs: Maximum number of training epochs.
            lr: Learning rate for Adam optimizer.
            beta: Weight for KL divergence term.
            device: Device to run on ('cpu' or 'cuda').
            dtype: Data type for tensors.
            seed: Random seed for reproducibility.
            eps: Small epsilon value for numerical stability.
            activation: Activation function ('relu', 'leakyrelu', 'gelu').
            use_batch_norm: Where to apply BatchNorm1d.
            use_layer_norm: Where to apply LayerNorm.
            **kwargs: Additional arguments passed to parent class.

        See Also:
            AutoEncoderModel.__init__ for additional details on normalization options.
        """
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

    def _compute_library(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute log library size for each cell.

        Used to normalize total expression before decoding.

        Args:
            X (torch.Tensor): Raw count matrix of shape (batch_size, n_genes)

        Returns:
            torch.Tensor: Log-sum of counts per cell, reshaped to (batch_size, 1)
        """
        library = torch.log(X.sum(1)).unsqueeze(1)
        return library
    
    def _px(self, px_param_dict: Dict[str, torch.Tensor]) -> ZeroInflatedNegativeBinomial:
        """
        Construct a ZINB distribution from parameter dictionary.

        Args:
            px_param_dict: Output from decoder containing 'rate', 'r', 'dropout', 'scale'

        Returns:
            ZeroInflatedNegativeBinomial: A fully specified ZINB distribution object.
        """
        px = ZeroInflatedNegativeBinomial(
            mu=px_param_dict['rate'],
            theta=px_param_dict['r'].exp(),
            zi_logits=px_param_dict['dropout'],
            scale=px_param_dict['scale'],
        )
        return px

    def _zinb_loss(self, x: torch.Tensor, px: ZeroInflatedNegativeBinomial) -> torch.Tensor:
        """
        Compute ZINB negative log-likelihood loss.

        Args:
            x (torch.Tensor): Observed count data
            px (ZeroInflatedNegativeBinomial): Predicted ZINB distribution

        Returns:
            Scalar reconstruction loss summed over genes and averaged over batch.
        """
        reconst_loss = -px.log_prob(x).sum(-1).sum()
        return reconst_loss
    
    def loss_function(
        self,
        x: torch.Tensor,
        outputs: Dict[str, torch.Tensor],
        mu: torch.Tensor,
        logvar: torch.Tensor,
        qz: Optional[torch.distributions.Normal],
        z: torch.Tensor,
        *args, **kwargs
    ) -> torch.Tensor:
        """
        Calculate the total loss combining ZINB reconstruction and KL divergence.
        
        Args:
            x: Input data (raw counts)
            outputs: ZINB parameters predicted by decoder
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
            qz: Posterior distribution (for monitoring)
            z: Sampled latent vector

        Returns:
            Total loss scalar (reconstruction + beta * KL)
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
    
    def _train_epoch(self) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Returns:
            Average metrics for this epoch (e.g., zinb_loss, kl_loss, total_loss)
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

    def _sample_from_zinb(self, px: ZeroInflatedNegativeBinomial) -> torch.Tensor:
        """
        Sample from ZINB distribution.
        
        Args:
            px: A trained ZINB distribution object

        Returns:
            Samples from ZINB distribution (non-negative integers)
        """
        set_seed(self.seed)
        return px.sample()

    def _get_reconstructions(self, use_mu: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get latent representations and reconstructions for the full dataset.

        Args:
            use_mu (bool): If True, uses mean expression (rate) as reconstruction;
                          if False, samples from ZINB distribution.

        Returns:
            tuple: (Z_latent, Xhat_reconstructed) arrays of shape (n_samples, z_dim) and (n_samples, input_dim)
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