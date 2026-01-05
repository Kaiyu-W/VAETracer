from copy import deepcopy
from collections import defaultdict
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader, TensorDataset

from .utils import (
    set_seed, 
    DEVICE,
    _input_tensor
)
from .typing import Tuple, Optional, Literal
from .log import logger


class DummyPbar:
    def update(self, n):
        pass

    def close(self):
        pass


class Encoder(nn.Module):
    def __init__(
        self, input_dim, hidden_dims, z_dim, model_type='VAE', activation='relu', # encoder should have any activation
        dropout_rate=0.1, use_batch_norm=False, use_layer_norm=True
    ):
        super().__init__()

        # Choose activation function
        if activation.lower() == 'relu':
            act_fn = nn.ReLU(inplace=True)
        elif activation.lower() == 'leakyrelu':
            act_fn = nn.LeakyReLU(0.2, inplace=True)
        elif activation.lower() == 'gelu':
            act_fn = nn.GELU()
        elif activation.lower() == 'none':
            act_fn = None
        else:
            raise ValueError("Supported activations: relu, leakyrelu, gelu, none")

        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(filter(lambda x: x is not None, [
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=0.001) if use_batch_norm else None,
                nn.LayerNorm(hidden_dim, elementwise_affine=False) if use_layer_norm else None,
                act_fn,
                nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
            ]))
            in_dim = hidden_dim
        self.hidden_layers = nn.Sequential(*layers)
        self.fc_z = nn.Linear(hidden_dims[-1], z_dim)  # for AE
        self.fc_mu = nn.Linear(hidden_dims[-1], z_dim)  # for VAE
        self.fc_logvar = nn.Linear(hidden_dims[-1], z_dim)  # for VAE

        if model_type == 'VAE':
            self.forward = self.forward_vae
        elif model_type == 'AE':
            self.forward = self.forward_ae
        else:
            raise ValueError("Invalid model type. Choose 'AE' or 'VAE'.")

    def forward_vae(self, x):
        h = self.hidden_layers(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def forward_ae(self, x):
        h = self.hidden_layers(x)
        z = self.fc_z(h)
        return z

class Decoder(nn.Module):
    def __init__(
        self, z_dim, hidden_dims, output_dim, activation='none', # decoder should have no activation
        dropout_rate=0, use_batch_norm=False, use_layer_norm=True
    ):
        super().__init__()

        # Choose activation function
        if activation.lower() == 'relu':
            act_fn = nn.ReLU(inplace=True)
        elif activation.lower() == 'leakyrelu':
            act_fn = nn.LeakyReLU(0.2, inplace=True)
        elif activation.lower() == 'gelu':
            act_fn = nn.GELU()
        elif activation.lower() == 'none':
            act_fn = None
        else:
            raise ValueError("Supported activations: relu, leakyrelu, gelu, none")

        layers = []
        in_dim = z_dim
        for hidden_dim in hidden_dims:
            layers.extend(filter(lambda x: x is not None, [
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=0.001) if use_batch_norm else None,
                nn.LayerNorm(hidden_dim, elementwise_affine=False) if use_layer_norm else None,
                act_fn,
                nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
            ]))
            in_dim = hidden_dim
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, z):
        h = self.hidden_layers(z)
        x_hat = self.output_layer(h)
        return x_hat

class AE(nn.Module):
    def __init__(
        self, input_dim, hidden_dims, z_dim, activation='relu', 
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
    ):
        super().__init__()

        self.use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        self.use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        self.use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        self.use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        self.encoder = Encoder(
            input_dim, hidden_dims, z_dim, model_type='AE', 
            activation=activation, 
            use_batch_norm=self.use_batch_norm_encoder, 
            use_layer_norm=self.use_layer_norm_encoder
        )
        self.decoder = Decoder(
            z_dim, hidden_dims[::-1], input_dim, 
            activation='none', 
            use_batch_norm=self.use_batch_norm_decoder, 
            use_layer_norm=self.use_layer_norm_decoder
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = F.softplus(self.decoder(z)) # Xhat should be non-negative
        return x, x_hat, z

    @torch.no_grad()
    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

class VAE(nn.Module):
    def __init__(
        self, input_dim, hidden_dims, z_dim, activation='relu', 
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
    ):
        super().__init__()

        self.use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        self.use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        self.use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        self.use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        self.encoder = Encoder(
            input_dim, hidden_dims, z_dim, model_type='VAE', 
            activation=activation, 
            use_batch_norm=self.use_batch_norm_encoder, 
            use_layer_norm=self.use_layer_norm_encoder
        )
        self.decoder = Decoder(
            z_dim, hidden_dims[::-1], input_dim,
            activation='none', 
            use_batch_norm=self.use_batch_norm_decoder, 
            use_layer_norm=self.use_layer_norm_decoder
        )

    def reparameterize(self, mu, logvar):
        std = F.softplus(0.5 * logvar) # torch.exp(0.5 * logvar)

        if torch.isnan(mu).any() or torch.isnan(std).any() or torch.isinf(mu).any() or torch.isinf(std).any():
            return None, torch.full_like(mu, float('nan'))
            
        dist = Normal(
            mu, 
            std.clamp(min=torch.finfo(std.dtype).tiny)
        )
        z = dist.rsample()
        return dist, z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        qz, z = self.reparameterize(mu, logvar)
        x_hat = F.softplus(self.decoder(z))  # Xhat should be non-negative
        return x, x_hat, mu, logvar, qz, z

    @torch.no_grad()
    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


class AutoEncoderModel:
    """
    A unified implementation of Autoencoder (AE) and Variational Autoencoder (VAE).
    
    Args:
        input_dim (int): Dimension of input features
        hidden_dims (list): List of hidden dimensions for encoder/decoder
        z_dim (int): Dimension of latent space
        num_epochs (int): Number of training epochs
        lr (float, optional): Learning rate. Defaults to 1e-3
        beta (float, optional): Weight of KL divergence for VAE. Defaults to 1
        model_type (str, optional): Type of model ('AE' or 'VAE'). Defaults to 'VAE'
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
    """

    def __init__(
        self, input_dim, hidden_dims, z_dim, num_epochs, lr=1e-3, beta=1, seed=42, eps=1e-8,
        model_type='VAE', device=DEVICE, dtype=torch.float32,
        activation: Literal["relu", "leakyrelu", "gelu"] = 'relu',
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        **kwargs
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.z_dim = z_dim
        self.num_epochs = num_epochs
        self.lr = lr
        self.model_type = model_type
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.beta = beta # for VAE
        self._eps = eps
        self.kwargs = kwargs
        self._validate_inputs()

        self.best_model_state = None
        self.num_workers = 4
        self.pin_memory = False
        self.dtype = dtype
        self.batch_size = 256
        self.data_loader = None
        self.valid_loader = None
        self.X = None
        self.Z = None
        self.Xhat = None
        self.train_metrics = []
        self.valid_metrics = []
        self.current_metrics = {}

        self.seed = seed
        self.DEVICE = device # self.device.type
        self.device = torch.device(self.DEVICE)
        if self.DEVICE != DEVICE:
            logger.info(f'(Model) Using device: {self.device.type}')
        set_seed(self.seed, device=self.device)

        if self.model_type=='VAE':
            self.model_class = VAE
            self.loss = self.loss_vae
            if self.beta is None or not isinstance(self.beta, (int,float)):
                self.beta = 1
                logger.info('(Model) Using beta for VAE: 1')
        elif self.model_type=='AE':
            self.model_class = AE
            self.loss = self.loss_ae
        else:
            return

        self.model = self.model_class(
            self.input_dim, self.hidden_dims, self.z_dim, 
            activation=self.activation, 
            use_batch_norm=self.use_batch_norm, 
            use_layer_norm=self.use_layer_norm
        ).to(self.device).to(self.dtype)
        self.initialize_weights(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _validate_inputs(self):
        """
        Validate model inputs and configurations.
        
        Raises:
            ValueError: For invalid input dimensions or configurations
        """
        if self.model_type not in [None,'AE','VAE']:
            raise ValueError("Invalid model type. Choose None, 'AE' or 'VAE'.")
        if self.activation.lower() not in ['relu','leakyrelu','gelu']:
            raise ValueError("Supported activations: relu, leakyrelu, gelu")
        if self.use_batch_norm not in ["encoder", "decoder", "none", "both"]:
            raise ValueError("Supported use_batch_norm: encoder, decoder, none, both")
        if self.use_layer_norm not in ["encoder", "decoder", "none", "both"]:
            raise ValueError("Supported use_layer_norm: encoder, decoder, none, both")
        if self.input_dim <= 0 or self.z_dim <= 0:
            raise ValueError("input_dim and z_dim must be positive!")
        if self.z_dim >= self.input_dim:
            logger.warning("Latent dimension >= input dimension may lead to poor compression")
        if self.beta < 0:
            raise ValueError("Beta must be non-negative")
        if not self.hidden_dims or any(h <= 0 for h in self.hidden_dims):
            raise ValueError("Hidden dimensions must be positive")
        if self.lr <= 0:
            raise ValueError("learning rate must be positive!")

    def load_data(self, X, batch_size, num_workers=0, pin_memory=False, dtype=None, shuffle=True):
        """
        Load training data into DataLoader.
        
        Args:
            X (numpy.ndarray): Input data
            batch_size (int): Size of each training batch
            num_workers (int, optional): Number of workers for DataLoader. Defaults to 0
            pin_memory (bool): Whether to pin memory in CPU. Defaults to False
            dtype (torch.dtype, optional): Data type of tensors. Defaults to self.dtype
            shuffle (bool, optional): Whether to shuffle the data. Defaults to True
        
        Notes:
            - pin_memory=True is recommended when using GPU
            - Requires more CPU memory but speeds up GPU transfer

        Raises:
            ValueError: If input dimension doesn't match model's input_dim
        """

        if X.shape[1] != self.input_dim:
            raise ValueError(f"Input data must have {self.input_dim} features, but got {X.shape[1]} features.")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        if dtype is not None:
            if self.dtype is None:
                self.dtype = dtype
            elif dtype != self.dtype:
                logger.info(f"Try to load data as {dtype}, while model used {self.dtype}! Use model's one.")
        self.X = _input_tensor(X, dtype=self.dtype)

        # load (shuffle) X and search optimal batch_size if out of memory
        try:
            self.data_loader = self._create_dataloader(
                self.X, self.batch_size, shuffle
            )
        except RuntimeError as e:
            error_msg = str(e).lower()
            is_memory_error = (
                ("out of memory" in error_msg and self.device.type == 'cuda') or
                ("memory" in error_msg and self.device.type == 'cpu')
            )
            
            if is_memory_error:
                # Find optimal batch size
                optimal_batch = self._handle_memory_error(batch_size, min_batch=10)
                self.data_loader = self._create_dataloader(
                    self.X, optimal_batch, shuffle
                )

                # update batch size, if works
                self.batch_size = optimal_batch
            else:
                raise e

        # load ordered X
        if shuffle:
            self.data_loader_order = self._create_dataloader(
                self.X, self.batch_size, shuffle=False
            )
        else:
            self.data_loader_order = self.data_loader

    def load_valid_data(self, Xvalid):
        """
        Load validation data into DataLoader.
        
        Args:
            Xvalid (numpy.ndarray): Validation data

        Raises:
            ValueError: If input dimension doesn't match model's input_dim
        """

        if Xvalid.shape[1] != self.input_dim:
            raise ValueError(f"Input data must have {self.input_dim} features, but got {Xvalid.shape[1]} features.")
        self.Xvalid = _input_tensor(Xvalid, dtype=self.dtype)
        self.valid_loader = self._create_dataloader(
            self.Xvalid, self.batch_size, shuffle=False
        )

    def _create_dataloader(self, data, batch_size, shuffle, sample_related_vars=None):
        """
        Helper method to create DataLoader with given batch size.
        """
        self.batch_size = batch_size
        _data = (data, ) if sample_related_vars is None else (data, sample_related_vars)
        return DataLoader(
            TensorDataset(*_data),
            batch_size=self.batch_size,
            shuffle=shuffle,              # random shuffle data
            num_workers=self.num_workers, # 0 means main-process
            pin_memory=self.pin_memory
        )

    def _clear(self):
        import gc
        gc.collect()

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            
        gc.collect()

    def clear_cuda(self):
        self.to_cpu()
        self.to_cuda()

    def to_cpu(self):
        if self.device.type != 'cpu':
            self.device = torch.device('cpu')
            self.model = self.model.to(self.device)
            self._clear()

    def to_cuda(self):
        if self.device.type != 'cuda':
            self.device = torch.device('cuda')
            self.model = self.model.to(self.device)
            self._clear()

    def _handle_memory_error(self, batch_size, min_batch=10):
        """
        Intelligently handle memory errors for both CPU and GPU.
        
        Args:
            batch_size (int): Current batch size
            min_batch (int): Minimum acceptable batch size
            
        Returns:
            int: Optimal batch size that fits in memory
            
        Raises:
            RuntimeError: If even minimum batch size doesn't fit in memory
        """
        # Clear memory based on device type
        self._clear()
        
        # Binary search for the largest batch size that works
        left, right = min_batch, batch_size
        optimal_batch = None
        
        while left <= right:
            mid = (left + right) // 2
            try:
                # Test batch size with a sample forward pass
                sample_batch = torch.randn(mid, self.input_dim).to(self.device)
                _ = self.model(sample_batch)
                
                # If successful, try larger batch size
                optimal_batch = mid
                left = mid + 1
                
            except RuntimeError as e:
                error_msg = str(e).lower()
                is_memory_error = (
                    ("out of memory" in error_msg and self.device.type == 'cuda') or
                    ("memory" in error_msg and self.device.type == 'cpu')
                )
                
                if is_memory_error:
                    # Try smaller batch size
                    right = mid - 1
                    # Clear memory again
                    self._clear()
                else:
                    raise e
        
        if optimal_batch is None:
            raise RuntimeError(f"Unable to find working batch size >= {min_batch}")
            
        logger.info(f"Found optimal batch size: {optimal_batch} on {self.device.type}")
        return optimal_batch

    def save_model(self, path):
        """
        Save model state and configuration to file.

        Args:
            path (str): Path to save the model
        """

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'z_dim': self.z_dim,
                'model_type': self.model_type,
                'device': self.device
            }
        }, path)

    def load_model(self, path, map_location=None):
        """
        Load model state and configuration from file.

        Args:
            path (str): Path to load the model from
            map_location (torch.device, optional): Device to map model to. Defaults to None
        """

        if map_location is None:
            map_location = self.device
        checkpoint = torch.load(path, map_location=map_location)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # make sure optimizer in device
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)

    def initialize_weights(self, model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                n_in = m.in_features
                n_out = m.out_features
                if n_in >= n_out: # encoder
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                else: # decoder
                    nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @torch.jit.script
    def _kl_loss(mu, logvar):
        return -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=-1
        ).sum()
    
    def _kl_loss_by_qz(self, z, qz):
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        return kl_divergence(qz, pz).sum(-1).sum()

    def loss_vae(self, x, x_hat, mu, logvar, qz, z, *args, **kwargs): # qz, z
        batch_size = x.size(0)
        MSE = F.mse_loss(x_hat, x, reduction='sum') / batch_size
        KLD = self._kl_loss(mu, logvar) / batch_size
        # KLD = self._kl_loss_by_qz(z, qz) / batch_size

        # Store individual losses for monitoring
        self.current_metrics = {
            'reconstruction_loss': MSE.item(),
            'kl_loss': KLD.item(),
            'total_loss': MSE.item() + self.beta * KLD.item()
        }

        return MSE + self.beta * KLD

    def loss_ae(self, x, x_hat, *args, **kwargs): # z
        MSE = F.mse_loss(x_hat, x, reduction='sum')

        # Store individual losses for monitoring
        self.current_metrics = {
            'reconstruction_loss': MSE.item(),
            'total_loss': MSE.item()
        }

        return MSE

    def _get_reconstructions(self):
        self.model.eval()
        z_list, xhat_list = [], []
        
        with torch.no_grad():
            for batch in self.data_loader_order: # use right order but not shuffle!
                x = batch[0].to(self.device)
                outs = self.model(x) # (x, x_hat, mu, logvar, qz, z) for VAE; (x, x_hat, z) for AE
                z, x_hat = outs[2], outs[1] # z should be mu for VAE
                z_list.append(z.cpu())
                xhat_list.append(x_hat.cpu())

        Z = torch.cat(z_list, dim=0).numpy()
        Xhat = torch.cat(xhat_list, dim=0).numpy()

        return Z, Xhat

    def _train_epoch(self):
        """
        Train model for one epoch.

        Returns:
            dict: Dictionary containing average metrics for the epoch
        """

        self.model.train()
        total_metrics = defaultdict(float)
        n_samples = 0
        
        for batch in self.data_loader:
            # 1. remove previous grad
            self.optimizer.zero_grad()

            # 2. run epoch
            x = batch[0].to(self.device)
            batch_size = x.size(0)
            n_samples += batch_size
            outs = self.model(x) # (x, x_hat, mu, logvar, qz, z) for VAE; (x, x_hat, z) for AE
            if outs[-2] is None: # VAE: qz is None, when mu/logvar has nan
                return dict(total_loss=float('nan'))

            # 3. loss
            loss = self.loss(*outs)
            loss.backward()

            # 4. update param
            self.optimizer.step()

            # 5. Accumulate metrics
            for k, v in self.current_metrics.items():
                total_metrics[k] += v * batch_size
                
        # Calculate average metrics
        avg_metrics = {
            k: v/n_samples 
            for k, v in total_metrics.items()
        }
        # Store training metrics
        self.train_metrics.append(avg_metrics)
        return avg_metrics

    def _validate(self):
        """
        Validate model on validation set.
        
        Returns:
            dict: Dictionary containing average metrics for validation
        """
        self.model.eval()
        total_metrics = defaultdict(float)
        n_samples = 0

        with torch.no_grad():
            for batch in self.valid_loader:
                # 1. remove previous grad
                self.optimizer.zero_grad()

                # 2. run epoch
                x = batch[0].to(self.device)
                batch_size = x.size(0)
                n_samples += batch_size
                outs = self.model(x) # (x, x_hat, mu, logvar, qz, z) for VAE; (x, x_hat, z) for AE
                if outs[-2] is None: # VAE: qz is None, when mu/logvar has nan
                    return dict(total_loss=float('nan'))

                # 3. loss
                _ = self.loss(*outs)
                # self.current_metrics updates when computing loss

                # 4. Accumulate metrics
                for k, v in self.current_metrics.items():
                    total_metrics[k] += v * batch_size
        
        # Calculate average metrics
        avg_metrics = {
            k: v/n_samples 
            for k, v in total_metrics.items()
        }
        # Store validation metrics
        self.valid_metrics.append(avg_metrics)
        return avg_metrics

    def _check_tensor(self, tensor):
        return torch.isnan(tensor).any() or torch.isinf(tensor).any()

    def _check_nan(self, optimizer, loss=None):
        # check loss
        if loss is not None:
            if isinstance(loss, torch.Tensor):
                if self._check_tensor(loss):
                    return True
            if np.any(np.isnan(loss)) or np.any(np.isinf(loss)):
                return True

        # check param/grad
        for param_group in optimizer.param_groups:
            for param in filter(
                lambda p: p is not None, 
                param_group['params']
            ):
                if self._check_tensor(param):
                    return True
                if param.grad is not None and self._check_tensor(param.grad):
                    return True

        return False

    def train(self, use_tqdm=False, patience=45, min_delta=None, verbose=True, **kwargs):
        """
        Train the model with early stopping.

        Args:
            use_tqdm (bool, optional): Whether to use tqdm progress bar. Defaults to False
            patience (int, optional): Early stopping patience. Defaults to 45
            min_delta (float, optional): Minimum change in loss for early stopping. Defaults to 1e-4
        
        Returns:
            list: List of losses during training
        """
        if self.loss is None:
            raise ValueError("Loss function wasn't defined!")
            
        if min_delta is None:
            min_delta = self._eps

        # for logging
        if use_tqdm and verbose:
            pbar = tqdm(desc='train epoch', total=self.num_epochs)
            _log = lambda x,y : None
        else:
            pbar = DummyPbar()
            if verbose:
                logging_interval = max(1, self.num_epochs//100)
                def _log(epoch, loss):
                    if epoch==0 or (epoch + 1) % logging_interval == 0:
                        logger.info(f'Epoch {epoch + 1}/{self.num_epochs}, Loss: {loss:.4f}')
            else:
                _log = lambda x,y : None

        # for early stopping
        best_loss = float('inf')
        patience_counter = 0
        best_state_dict = None

        # for epochs
        losses = []
        set_seed(self.seed, device=self.device)

        # Reset metrics at the start of training
        self.train_metrics = []
        self.valid_metrics = []

        for epoch in range(self.num_epochs):
            self._epoch = epoch
            # train
            train_metrics = self._train_epoch()
            current_loss = train_metrics['total_loss']
            
            # valid if input
            if self.valid_loader is not None:
                valid_metrics = self._validate()
                current_loss = valid_metrics['total_loss']

            # save loss
            losses.append(current_loss)

            # logging
            _log(epoch, current_loss)

            if self._check_nan(optimizer=self.optimizer, loss=current_loss):
                pbar.close()
                if verbose:
                    logger.info(f'Found nan at epoch {epoch}')
                break
            if best_loss - current_loss > min_delta:
                best_loss = current_loss
                patience_counter = 0
                best_state_dict = {
                    'model': deepcopy(self.model.state_dict()),
                    'epoch': epoch,
                    'loss': best_loss
                }
            else:
                patience_counter += 1
            if patience_counter >= patience:
                pbar.close()
                if verbose:
                    logger.info(f'Early stopping at epoch {epoch}')
                break

            pbar.update(1)
        pbar.close()

        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict['model'])
            if verbose:
                logger.info(f"Restored best model from epoch {best_state_dict['epoch']}")

        self.Z, self.Xhat = self._get_reconstructions()

        self._clear()
        return losses