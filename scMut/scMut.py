import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from .baseVAE import (
    Decoder, 
    VAE, 
    AutoEncoderModel,
    DummyPbar,
    deepcopy,
    tqdm
)
from .utils import (
    set_seed, 
    DEVICE, 
    _input_tensor, 
    _softplus_inverse
)
from .typing import Tuple, Optional, Literal
from .log import logger


# R -> n 'x' p
def computeR_by_np(
    n: torch.Tensor,
    p: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Efficient and numerically stable computation of R = 1 - (1 - p)^n.

    Computes the probability of at least one mutation occurring across n generations,
    given site-specific mutation rate p. Uses multiple approximation strategies to
    avoid numerical overflow/underflow in extreme cases.

    This implements the core cumulative mutation model used throughout scMut.

    Args:
        n: Tensor of shape (batch_size, 1) or scalar representing generation count.
        p: Tensor of shape (feature_dim,) or scalar representing per-site mutation rate.
        eps: Small value for numerical stability.

    Returns:
        r: Probability tensor of same shape as broadcasted (n, p), clamped in [0,1].
           Computed as r = 1 - (1-p)^n using exact or approximate methods depending on regime.
    """

    log_safe_1mp = torch.log1p(-p + eps)          # log(1-p)
    r_exact  = -torch.expm1(n * log_safe_1mp)     # Exact computation: r = 1 - (1-p)^n

    # approx condition
    q = 1 - p
    np = n * p
    nq = n * q
    threshold = torch.minimum(torch.tensor(0.1), 1 / n)
    # p ~ 0
    use_linear_approx_0 = (np <= threshold)
    use_exp_approx_0 = (~use_linear_approx_0) & (p <= 0.01)
    # p ~ 1
    use_linear_approx_1 = (nq <= threshold)
    use_exp_approx_1 = (~use_linear_approx_1) & (q <= 0.01)

    # approx formula
    # p ~ 0
    r_linear1_0 = np                                      # Linear approximation: r ~ np
    r_linear2_0 = r_linear1_0 - (n * (n - 1) * p**2) / 2  # Second-order: r ~ np - n(n-1)p^2/2
    r_exp_0    = -torch.expm1(-np)                        # Exponential approximation: r ~ 1 - exp(-np)
    
    # p ~ 1
    r_linear1_1 = 1 - nq                                  # Linear approximation: r ~ 1 - nq
    r_linear2_1 = r_linear1_1 + (n * (n - 1) * q**2) / 2  # Second-order: r ~ 1 - nq + n(n-1)q^2/2
    r_exp_1    = -torch.expm1(-nq)                        # Exponential approximation: r ~ 1 - exp(-nq)

    # final computation
    # use second-order for better approximation
    r = torch.where(
        use_linear_approx_0,
        r_linear2_0,
        torch.where(
            use_exp_approx_0,
            r_exp_0,
            torch.where(
                use_linear_approx_1,
                r_linear2_1,
                torch.where(
                    use_exp_approx_1,
                    r_exp_1,
                    r_exact
                )
            )
        )
    ).clamp(min=0., max=1.)

    return r


# NMF
def decompose_R_to_np(
    R: np.ndarray,
    n: Optional[np.ndarray] = None,
    p: Optional[np.ndarray] = None,
    only_train_p: bool = False,
    miss_value: int = 3,
    lr: float = 1e-3,
    patience: int = 45,
    min_delta: float = 0,
    eps: float = 1e-8,
    max_epoch: Optional[int] = None,
    logging_interval: int = 100,
    verbose: bool = True,
    device: Union[str, torch.device] = DEVICE,
    dtype: torch.dtype = torch.float64,
    use_tqdm: bool = False,
    batch_size: Optional[int] = None,
    seed: Optional[int] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[float]]:
    """
    Decompose observed mutation matrix R into latent variables n (generation) and p (mutation rate).

    Solves an optimization problem to find optimal n and p such that R ≈ 1 - (1 - p)^n,
    using binary cross-entropy loss with masked missing values.

    Supports both full joint optimization and partial update (only train p or only train n).

    Args:
        R: Observed mutation matrix of shape (n_samples, n_sites).
        n: Initial guess for cell-wise generation vector (n_cells,). If None, initialized automatically.
        p: Initial guess for site-wise mutation rate vector (n_sites,). If None, initialized to zeros.
        only_train_p: If True, only optimize p while keeping n fixed.
        miss_value: Value indicating missing/masked entries in R.
        lr: Learning rate for Adam optimizer.
        patience: Early stopping patience.
        min_delta: Minimum improvement threshold.
        eps: Numerical stability epsilon.
        max_epoch: Maximum number of epochs; if None, runs until convergence.
        logging_interval: Interval for progress logging.
        verbose: Whether to print logs.
        device: Device to run on ('cpu' or 'cuda').
        dtype: Data type for computation.
        use_tqdm: Use tqdm progress bar.
        batch_size: Batch size for training; if None, uses full-batch.
        seed: Random seed.

    Returns:
        tuple: (best_n, best_p, losses)
            - best_n: Estimated n vector (None if not optimized)
            - best_p: Estimated p vector (None if not optimized)
            - losses: Training loss history
    """
    if batch_size is not None:
        assert isinstance(batch_size, int) and batch_size >= 100, \
            "batch_size must be None or an integer >= 100!"
    if max_epoch is None:
        assert not use_tqdm, 'Set max_epoch if use_tqdm=True!'
    if use_tqdm and verbose:
        pbar = tqdm(desc='train epoch', total=max_epoch)
    if isinstance(device, str):
        device = torch.device(device)
    cpu = torch.device('cpu')

    # Ensure R is a tensor on the specified device and dtype
    if batch_size is None:
        if isinstance(R, torch.Tensor) and R.device.type==device:
            pass
        else:
            R = _input_tensor(R, dtype=dtype, device=device)
    else:
        R = _input_tensor(R, dtype=dtype, device=cpu)
    n_size, p_size = R.shape
    mask = R != miss_value  # Mask out invalid values
    assert mask.sum()>0

    # Initialize n_logit and p_logit
    trainable_params = []
    if n is not None:
        n = _input_tensor(n, device=device, dtype=dtype)
        n_logit = _softplus_inverse(n)
        if not only_train_p:
            n_logit = nn.Parameter(n_logit)
            trainable_params.append(n_logit)
    else:
        assert not only_train_p, 'Please input initial n if only_train_p=True!'
        n_logit = nn.Parameter(
            torch.ones(n_size).to(device).to(dtype)
        )
        trainable_params.append(n_logit)

    if p is not None:
        p = _input_tensor(p, device=device, dtype=dtype)
        p_logit = nn.Parameter(torch.logit(p))
    else:
        p_logit = nn.Parameter(
            torch.zeros(p_size).to(device).to(dtype)
        )
    trainable_params.append(p_logit)

    # Define optimizer
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    # Early stopping variables
    best_loss = float("inf")
    best_n = None
    best_p = None
    best_epoch = 0
    epochs_no_improve = 0
    epoch = 0
    losses = []

    # Optimization loop
    condition = (lambda x: True) if max_epoch is None else (lambda x: x < max_epoch)

    while condition(epoch):
        epoch += 1
        optimizer.zero_grad()

        # Compute n and p from logits
        n = F.softplus(n_logit).unsqueeze(1)  # Shape: [n_size, 1]
        p = torch.sigmoid(p_logit)  # Shape: [p_size]

        if batch_size is None:

            # Predict R
            predicted_R = computeR_by_np(n, p, eps)

            # Check if predicted_R contains invalid values (out of range [0, 1])
            if torch.any(predicted_R < 0.0) or torch.any(predicted_R > 1.0):
                if verbose:
                    if use_tqdm:
                        pbar.close()
                    logger.warning(f"Invalid predicted_R detected at epoch {epoch}. Terminating early.")
                break

            # Compute the loss using binary cross entropy
            # model_logits = torch.logit(predicted_R) # avoid out of range ([0,1], bug from cuda)
            # loss = F.binary_cross_entropy_with_logits(model_logits[mask], R[mask], reduction="sum") / (mask.sum() + eps)
            loss = F.binary_cross_entropy(predicted_R[mask], R[mask], reduction="sum") / (mask.sum() + eps)
            current_loss = loss.item()
            loss.backward()

        else: # batch

            batch_losses = []
            generator = torch.Generator(device=device)
            if seed is not None:
                generator.manual_seed(seed + epoch)
            indices = torch.randperm(R.shape[0], generator=generator, device=device)

            invalid_flag = False
            accum_count = 0
            for start in range(0, R.shape[0], batch_size):
                end = min(start + batch_size, R.shape[0])
                batch_idx = indices[start:end]
                batch_idx_cpu = batch_idx.to(cpu)

                n_batch = n[batch_idx]  # [B, 1]
                R_batch = R[batch_idx_cpu].to(device)
                mask_batch = mask[batch_idx_cpu].to(device)
                if not mask_batch.any():
                    continue

                predicted_R_batch = computeR_by_np(n_batch, p, eps)
                if torch.any(predicted_R_batch < 0.0) or torch.any(predicted_R_batch > 1.0):
                    if verbose:
                        if use_tqdm:
                            pbar.close()
                        logger.warning(f"Invalid predicted_R detected at epoch {epoch}, batch [{start}:{end}]. Terminating early.")
                    invalid_flag = True
                    break

                loss_batch = F.binary_cross_entropy(predicted_R_batch[mask_batch], R_batch[mask_batch], reduction="sum") / (mask_batch.sum() + eps)
                batch_losses.append(loss_batch.item())
                loss_batch.backward(retain_graph=True)

            if invalid_flag:
                break

            if not batch_losses:
                loss = float('inf')
            else:
                loss = sum(batch_losses) / len(batch_losses)
            current_loss = loss

        # check grad
        everything_ok = True
        for param in optimizer.param_groups[0]['params']:
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    everything_ok=False
                    break
        if not everything_ok:
            if verbose:
                if use_tqdm:
                    pbar.close()
                logger.warning(f"Gradient contains NaN or Inf at epoch {epoch}:")
            break

        optimizer.step()

        # Check for improvement
        losses.append(current_loss)
        if current_loss < best_loss - min_delta:
            best_loss = current_loss
            best_n = n.squeeze(1).detach().clone() if n.requires_grad else n.squeeze(1).clone()
            best_p = p.detach().clone()
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Print progress
        if verbose:
            if use_tqdm:
                pbar.update(1)
            else:
                if epoch == 1 or epoch % logging_interval == 0:
                    logger.info(f"Iteration {epoch}: loss={current_loss:.6f}")

        # Early stopping condition
        if epochs_no_improve >= patience:
            if verbose:
                if use_tqdm:
                    pbar.close()
                logger.info(f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}")
            break

    # aviod error when epoch=1 and predicted_R contains invalid values
    if best_n is None and best_p is None: # epoch==1
        return best_n, best_p, losses

    return best_n.cpu().numpy(), best_p.cpu().numpy(), losses


# bayesian R
def compute_posterior(
    r: np.ndarray,
    r0: float,
    r1: np.ndarray,
    p_prior: np.ndarray,
    n: int = 1000,
    eps: float = 1e-8,
    mask_value: int = 3
) -> np.ndarray:
    """
    Compute posterior probability of true mutation status using Bayesian inference.

    Updates prior belief about mutation status based on observation and expected rates.

    Used for denoising and imputation of missing values.

    Args:
        r: Observed mutation frequency matrix.
        r0: Expected rate under null hypothesis (no mutation).
        r1: Expected rate under alternative hypothesis (mutation present).
        p_prior: Prior probability of mutation at each site.
        n: Number of trials (generations).
        eps: Small value for numerical stability.
        mask_value: Indicator for missing data.

    Returns:
        Posterior probability matrix of mutation presence.
    """
    r = np.asarray(r)
    r1 = np.asarray(r1)
    p_prior = np.asarray(p_prior)
    
    mask = (r == mask_value)
    r = np.where(mask, 0, r) # np.mean(r, axis=0, where=mask)
    
    r0 = np.broadcast_to(r0, p_prior.shape)
    r0 = np.clip(r0, eps, 1 - eps)
    r1 = np.clip(r1, eps, 1 - eps)
    p_prior = np.clip(p_prior, eps, 1 - eps)

    k = r * n
    log_like_M0 = k * np.log(r0) + (n - k) * np.log(1 - r0)
    log_like_M1 = k * np.log(r1) + (n - k) * np.log(1 - r1)
    log_weighted_M0 = log_like_M0 + np.log(1 - p_prior)
    log_weighted_M1 = log_like_M1 + np.log(p_prior)
    log_denominator = np.logaddexp(log_weighted_M0, log_weighted_M1)
    
    log_posterior = log_weighted_M1 - log_denominator
    posterior = np.exp(log_posterior)
    
    return posterior


# n (float) -> n (int)
def optimize_integer_scaling(
    n: np.ndarray,
    max_n: int,
    max_error_ratio: float = 0.1,
    tolerance_ratio: float = 0.01,
    max_iter: int = 1000,
    num_random: int = 50,
    seed: int = 42
) -> Tuple[np.ndarray, float, Dict[float, float]]:
    """
    Find optimal scaling factor k to convert real-valued generation scores to integers.

    Minimizes rounding error under constraint that maximum scaled value does not exceed max_n.

    Useful for converting continuous N estimates into discrete division counts.

    Args:
        n: Array of real-valued generation scores.
        max_n: Target maximum integer value after rounding.
        max_error_ratio: Search range around initial estimate.
        tolerance_ratio: Convergence threshold relative to scale.
        max_iter: Maximum refinement iterations.
        num_random: Number of random perturbations per step.
        seed: For reproducible sampling.

    Returns:
        tuple: (n_round_optimal, k_optimal, loss_records)
            - n_round_optimal: Rounded integer array
            - k_optimal: Optimal scaling factor
            - loss_records: History of tested k values and errors
    """
    len_n = len(n)
    
    # Initialize a default RNG if none is provided
    rng = np.random.default_rng(seed)
    
    # Step 1: Initial estimate of k
    k_init = max_n / np.max(n)
    k_low = (1 - max_error_ratio) * k_init
    k_high = (1 + max_error_ratio) * k_init
    tolerance_abs = tolerance_ratio * k_init  # Convert relative tolerance to absolute
    
    # Step 2: Initialize loss records
    loss_records = {}  # To store k and its corresponding rounding error
    
    # Helper function to compute objective and record loss
    def compute_loss(k):
        n_scaled = n * k
        n_round = np.round(n_scaled)
        rounding_error = np.sum((n_round - n_scaled) ** 2) / len_n
        loss_records[k] = rounding_error  # Record rounding error
        return rounding_error  # Objective function is just rounding error
    
    # Step 3: Grid search + Random perturbation for initial candidates
    num_samples = 50  # Number of samples in grid search
    k_candidates = np.linspace(k_low, k_high, num_samples)
    
    # Add random perturbations around grid points using the provided RNG
    k_candidates_random = []
    for k in k_candidates:
        k_candidates_random.extend([k + rng.uniform(-tolerance_abs, tolerance_abs) for _ in range(num_random)])
    k_candidates = np.unique(np.concatenate((k_candidates, k_candidates_random)))
    
    # Evaluate all candidates in a vectorized manner
    obj_values = []
    for k in k_candidates:
        obj = compute_loss(k)
        obj_values.append(obj)
    
    best_k_idx = np.argmin(obj_values)
    best_k = k_candidates[best_k_idx]
    best_obj = obj_values[best_k_idx]
    
    # Step 4: Local refinement using binary search + Random perturbation
    k_low, k_high = best_k - tolerance_abs, best_k + tolerance_abs
    for _ in range(max_iter):
        k_mid = (k_low + k_high) / 2
        
        # Evaluate k_mid and random perturbations around it
        k_perturbed = [k_mid + rng.uniform(-tolerance_abs, tolerance_abs) for _ in range(num_random)]
        k_candidates_local = np.unique([k_mid] + k_perturbed)
        
        obj_values_local = []
        for k in k_candidates_local:
            obj = compute_loss(k)
            obj_values_local.append(obj)
        
        best_k_local_idx = np.argmin(obj_values_local)
        best_k_local = k_candidates_local[best_k_local_idx]
        best_obj_local = obj_values_local[best_k_local_idx]
        
        if best_obj_local < best_obj:
            best_k = best_k_local
            best_obj = best_obj_local
        
        # Adjust search range
        n_round_mid = np.round(n * best_k)
        if np.max(n_round_mid) < max_n:
            k_low = best_k
        else:
            k_high = best_k
        
        # Check convergence
        if k_high - k_low < tolerance_abs:
            break
    
    # Final result
    k_optimal = best_k
    n_round_optimal = np.round(n * k_optimal).astype(int)
        
    return n_round_optimal, k_optimal, loss_records


# VAE model:

class MutDecoder_np(Decoder):
    """
    Decoder specialized for estimating scalar parameters (n or p) from latent space.

    Outputs a single logit value per sample, transformed via softplus (for n) or sigmoid (for p).
    Used in scMut to predict either:
        - cellular generation index (n), or
        - site-specific mutation rate (p)
    depending on training mode.
    """
    def __init__(
        self,
        z_dim: int,
        hidden_dims: List[int],
        output_dim: int = 1,
        activation: Literal["none"] = 'none',
        dropout_rate: float = 0,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True
    ):
        """
        Initialize the decoder.

        Args:
            z_dim: Dimension of latent input.
            hidden_dims: Hidden layer sizes (in reverse order of encoder).
            output_dim: Must be 1; outputs one parameter per sample.
            activation: Activation function between layers ('none' recommended).
            dropout_rate: Dropout probability (>0 enables dropout).
            use_batch_norm: Whether to apply BatchNorm1d.
            use_layer_norm: Whether to apply LayerNorm.

        Raises:
            ValueError: If output_dim != 1.
        """
        if output_dim != 1:
            raise ValueError("output_dim must be 1")
        super().__init__(
            z_dim=z_dim, hidden_dims=hidden_dims, output_dim=output_dim, activation=activation, 
            dropout_rate=dropout_rate, use_batch_norm=use_batch_norm, use_layer_norm=use_layer_norm
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the n/p decoder.

        Args:
            z: Latent tensor of shape (batch_size, z_dim)

        Returns:
            np_logit: Logit output of shape (batch_size, 1), representing:
                      - n_logit = softplus_inverse(n) in normal mode
                      - p_logit = logit(p) in reverse mode
        """
        h = self.hidden_layers(z)
        np_logit = self.output_layer(h) 
        # for n: n>0,     n=softplus(logit)
        # for p: 0<=p<=1, p=sigmoid(logit)
        return np_logit


class scMut(VAE):
    """
    Complete scMut model combining encoder and decoder for mutation data modeling.

    This model implements two distinct learning strategies based on the `reverse` flag:

    - Normal mode (reverse=False): 
        The latent variable Z is used to predict cellular generation (N).
        A fixed learnable parameter p_logit (one per site) stores the base mutation rate.
        This is the standard configuration for estimating N given known or initialized P.

    - Reverse mode (reverse=True):
        The latent variable Z is used to predict site-level mutation rates (P).
        A fixed learnable parameter n_logit (one per cell) stores the base generation count.
        This enables inference of P when N is assumed or pre-estimated.

    In both modes:
    - N (cellular generation) is always associated with cells
    - P (mutation rate) is always associated with sites
    - The role of Z determines what is learned dynamically from the data
    - The other quantity (N or P) is stored as a global parameter

    This design allows flexible integration with prior knowledge and multi-stage training.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        z_dim: int,
        activation: Literal["relu", "leakyrelu", "gelu"] = 'relu',
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        reverse: bool = False
    ):
        """
        Initialize the scMut model.

        Args:
            input_dim: Number of features (e.g., number of sites in normal mode).
            hidden_dims: Sizes of hidden layers in encoder/decoder.
            z_dim: Dimensionality of latent space.
            activation: Activation function for hidden layers.
            use_batch_norm: Where to apply BatchNorm1d.
            use_layer_norm: Where to apply LayerNorm.
            reverse: Controls the learning target:
                - If False (default): Z predicts N (generation per cell); p_logit is fixed/global.
                - If True: Z predicts P (mutation rate per site); n_logit is fixed/global.
                    
                     This does NOT swap biological roles:
                     - N is still the cellular generation count
                     - P is still the site-specific mutation probability
                    
                     Instead, it swaps which quantity is inferred from the latent space (Z),
                     and which is treated as a shared learnable parameter.
        """
        super().__init__(
            input_dim=input_dim, hidden_dims=hidden_dims, z_dim=z_dim, activation=activation, 
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm
        )
        self._reverse = reverse
        self.prob_layer = nn.Sigmoid()
        
        n_or_p = MutDecoder_np(
            # z_dim, hidden_dims[::-1], 1, 
            z_dim, [5], # 1,
            activation='none', 
            use_batch_norm=self.use_batch_norm_decoder, 
            use_layer_norm=self.use_layer_norm_decoder
        )
        if self._reverse:
            self.decoder_p = n_or_p
            self.n_logit = nn.Parameter(torch.ones(input_dim)) # generation for each cell(rev-feature)
        else:
            self.decoder_n = n_or_p
            self.p_logit = nn.Parameter(torch.zeros(input_dim)) # raw mutation rate for each gene(feature)

    def forward(
        self,
        x: torch.Tensor,
        train_np: bool = False
    ) -> Tuple[
        torch.Tensor,              # x_input
        torch.Tensor,              # n_logit or p_logit
        torch.Tensor,              # mu
        torch.Tensor,              # logvar
        Optional[torch.distributions.Normal],  # qz
        torch.Tensor               # z
    ]:
        """
        Forward pass of the model.
        
        Args:
            x: Input mutation matrix
            train_np: If True, returns n/p logits instead of reconstructed X

        Returns:
            tuple: Depending on mode, returns either (x, logits, ...) or (x, xhat, ...)
        """
        mu, logvar = self.encoder(x)
        qz, z = self.reparameterize(mu, logvar)

        if train_np:
            if self._reverse:
                p_logit = self.decoder_p(z)
                return x, p_logit, mu, logvar, qz, z
            else:
                n_logit = self.decoder_n(z)
                return x, n_logit, mu, logvar, qz, z
        else:
            xhat = self.prob_layer(self.decoder(z))
            return x, xhat, mu, logvar, qz, z

    @torch.no_grad()
    def set_mode1(self, mode: Literal["xhat", "np"]) -> None:
        """
        Set the model to a specific phase in mode1 training sequence.

        Mode1 Sequence: first train np (estimate N/P), then train xhat (reconstruct X).

        Args:
            mode: 'xhat' or 'np' — determines which parts to freeze/unfreeze.
                - "xhat": Freeze encoder / np-related decoders, unfreeze decoder.
                - "np": Freeze decoder, unfreeze encoder / np-related decoders.
        """
        if mode == "xhat":
            # Freeze np-related decoders
            if not self._reverse:
                for param in self.decoder_n.parameters():
                    param.requires_grad = False
            else:
                for param in self.decoder_p.parameters():
                    param.requires_grad = False
            
            # Freeze encoder and unfreeze decoder
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = True

        elif mode == "np":
            # Freeze decoder and unfreeze encoder
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = False
            
            # Unfreeze np-related decoders
            if not self._reverse:
                for param in self.decoder_n.parameters():
                    param.requires_grad = True
            else:
                for param in self.decoder_p.parameters():
                    param.requires_grad = True
        else:
            raise ValueError(f"Unsupported mode: {mode}. Choose from 'xhat' or 'np'.")

    @torch.no_grad()
    def set_mode2(self, mode: Literal["xhat", "np"]) -> None:
        """
        Set the model to a specific phase in mode2 training sequence.

        Mode2 Sequence: first train xhat (reconstruct X), then train np (estimate N/P).

        Args:
            mode: 'xhat' or 'np' — determines which parts to freeze/unfreeze.
                - "xhat": Freeze np-related decoders, unfreeze encoder/decoder.
                - "np": Freeze encoder/decoder, unfreeze np-related decoders.
        """
        if mode == "xhat":
            # Freeze np-related decoders
            if not self._reverse:
                for param in self.decoder_n.parameters():
                    param.requires_grad = False
            else:
                for param in self.decoder_p.parameters():
                    param.requires_grad = False
            
            # Unfreeze encoder and decoder
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = True

        elif mode == "np":
            # Freeze encoder and decoder
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            
            # Unfreeze np-related decoders
            if not self._reverse:
                for param in self.decoder_n.parameters():
                    param.requires_grad = True
            else:
                for param in self.decoder_p.parameters():
                    param.requires_grad = True
        else:
            raise ValueError(f"Unsupported mode: {mode}. Choose from 'xhat' or 'np'.")
        
        
class MutModel(AutoEncoderModel):
    """
    Model manager for scMut framework with multi-stage training and post-processing.

    Handles multi-stage training, evaluation, post-processing and denoising
    for single-cell somatic mutation data analysis.
    
    Key Features:
    - Two-stage training: first estimate N/P, then refine reconstruction
    - Optional transposition mode to swap sample/feature roles
    - Bayesian posterior-based denoising
    - Integer scaling for discrete division counts

    Example:
        model = MutModel(
            input_dim=500,
            hidden_dims=[128],
            z_dim=20,
            num_epochs=1000
        )
        model.load_data(X, batch_size=256)
        model.set_mode('np')
        model.train_np()
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        z_dim: int,
        num_epochs: int,
        num_epochs_nmf: Optional[int] = None,
        lr: float = 1e-3,
        beta_kl: float = 0.001,
        beta_best: float = 0.01,
        device: Union[str, torch.device] = DEVICE,
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
        eps: float = 1e-10,
        model_type: Optional[str] = None,
        mode_type: Literal["mode1", "mode2", None] = None,
        activation: Literal["relu", "leakyrelu", "gelu"] = 'relu',
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        miss_value: int = 3,
        wt_value: int = 0,
        edit_value: int = 1,
        edit_loss_weight: Optional[float] = None,
        wt_loss_weight: Optional[float] = None,
        use_weight: bool = False,
        train_transpose: bool = False,
        **kwargs
    ):
        """
        Initialize the MutModel.

        Args:
            input_dim: Number of input features (e.g., number of sites).
            hidden_dims: Dimensions of hidden layers in encoder/decoder.
            z_dim: Dimensionality of latent space.
            num_epochs: Maximum number of epochs for main training phase.
            num_epochs_nmf: Max epochs for NMF refinement step. If None, defaults to num_epochs.
            lr: Learning rate for Adam optimizer.
            beta_kl: Weight for KL divergence term during xhat-phase training.
            beta_best: Weight for consistency loss (aligning N/P estimates across stages).
            device: Device to run computations on ('cpu' or 'cuda').
            dtype: Data type for tensors (e.g., torch.float32).
            seed: Random seed for reproducibility.
            eps: Small epsilon for numerical stability.
            model_type: Ignored; forced to None to prevent AE/VAE fallback.
            mode_type: Training sequence type:
                - 'mode1': First train np, then train xhat
                - 'mode2': First train xhat, then train np
                - None: Use default set by set_mode()
            activation: Activation function for hidden layers ('relu', 'leakyrelu', 'gelu').
            use_batch_norm: Apply batch normalization in specified components.
            use_layer_norm: Apply layer normalization in specified components.
            miss_value: Value indicating missing/noisy entries in input matrix X.
            wt_value: Value representing wild-type (no mutation) status.
            edit_value: Value representing mutated (edited) status.
            edit_loss_weight: Weight for reconstruction loss on edited positions.
            wt_loss_weight: Weight for reconstruction loss on wild-type positions.
            use_weight: If True, applies weighted BCE loss using above weights.
            train_transpose: Controls both data layout and model learning target:

                - If False (default):
                    * Input X has shape (n_cells, n_sites)
                    * Model learns cellular generation (N) from latent space Z
                    * Site-level mutation rate (P) is globally shared and learned via fixed parameter p_logit

                - If True:
                    * Input X is transposed to (n_sites, n_cells)
                    * Model learns site-specific mutation rate (P) from latent space Z
                    * Cellular generation (N) is globally shared and learned via fixed parameter n_logit

                This flag synchronously controls:
                  1. Whether the input matrix X is transposed during data loading
                  2. The mode of the internal scMut model (via reverse=train_transpose)

                It enables dual analysis perspectives:
                  - Standard view: infer N per cell given P
                  - Transposed view: infer P per site given N

            **kwargs: Additional arguments passed to parent class AutoEncoderModel.

        Attributes:
            model (scMut): The core VAE model instance.
            optimizer1, optimizer2: Optimizers for different phases.
            N, P: Estimated cellular generations and site-level mutation rates.
            N_ft, P_ft: Finetuned (integer-scaled) versions of N and P.
            Xhat_np: Reconstructed mutation matrix from estimated N/P.
        """
        if model_type is not None:
            logger.info('model_type is forced to be set to None!')

        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            z_dim=z_dim,
            num_epochs=num_epochs,
            lr=lr,
            beta=beta_kl,
            device=device,
            dtype=dtype,
            seed=seed,
            eps=eps,
            activation=activation,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            model_type=None, # avoid raw AE/VAE model generation 
            **kwargs
        )
        self.num_epochs_nmf = num_epochs if num_epochs_nmf is None else num_epochs_nmf
        self.beta_best = beta_best
        self.loss = None # for baseVAE.train()
        self.miss_value = miss_value
        self.wt_value = wt_value
        self.edit_value = edit_value
        self.edit_loss_weight = 1. if edit_loss_weight is None else edit_loss_weight
        self.wt_loss_weight = 1. if wt_loss_weight is None else wt_loss_weight
        self.use_weight = use_weight 
        self.train_transpose = train_transpose
        self.feature_dim = 0 if self.train_transpose else 1

        #  X -> Encoder -> (μ_z, σ_z²) -> Z -> Decoder -> n -> r=1-(1-p)^n
        # │
        # └─ Global p（independent for each feature）
        # or to use non-mutation to replace mutation => p_rev = 1-p, r_rev = 1-r
        self.model = scMut(
            self.input_dim, self.hidden_dims, self.z_dim, 
            activation=self.activation,
            use_batch_norm=self.use_batch_norm,
            use_layer_norm=self.use_layer_norm,
            reverse=self.train_transpose,
        ).to(self.device).to(self.dtype)
        # self.initialize_weights(self.model)
        self.optimizer1 = torch.optim.Adam(
            [*self.model.decoder.parameters()], 
            lr=lr
        )
        self.optimizer2 = torch.optim.Adam(
            [*self.model.encoder.parameters(), *self.model.decoder.parameters()], 
            lr=lr
        )
        
        if self.train_transpose:
            self.n_logit = self.model.n_logit # get n_logit
            self.optimizer_np1 = torch.optim.Adam(
                [
                    *self.model.encoder.parameters(), 
                    *self.model.decoder_p.parameters(), 
                    self.model.n_logit
                ], 
                lr=lr
            )
            self.optimizer_np2 = torch.optim.Adam(
                [*self.model.decoder_p.parameters(), self.model.n_logit], 
                lr=lr
            )
        else:
            self.p_logit = self.model.p_logit # get p_logit
            self.optimizer_np1 = torch.optim.Adam(
                [
                    *self.model.encoder.parameters(), 
                    *self.model.decoder_n.parameters(), 
                    self.model.p_logit
                ], 
                lr=lr
            )
            self.optimizer_np2 = torch.optim.Adam(
                [*self.model.decoder_n.parameters(), self.model.p_logit], 
                lr=lr
            )

        self.N = None
        self.P = None
        self.N_ft = None
        self.P_ft = None
        self.N_nmf = None
        self.P_nmf = None
        self.Xhat_np = None
        self.mode_type = mode_type

        self.train_metrics_np = []
        self.valid_metrics_np = []
        self.current_metrics_np = {}
        # torch.autograd.set_detect_anomaly(True)

    def update_p(self, p_init: np.ndarray, requires_grad: bool = True) -> None:
        """
        Update the fixed mutation rate parameter (p_logit) in the model.

        Used to initialize or refine the site-level mutation rate prior before training.
        Only available when train_transpose=False.

        Args:
            p_init: Array of initial mutation probabilities per site, shape (n_sites,).
                    Must be in probability space [0,1], not logit space.
            requires_grad: Whether to enable gradient computation for this parameter during training.
                          If False, p_logit will remain frozen.

        Raises:
            AssertionError: If train_transpose=True (p_logit does not exist)
            AssertionError: If input shape does not match model's p_logit
        """
        assert not self.train_transpose, 'Parameter p_logit only exists in model with train_transpose=False!'
        p_init = _input_tensor(
            p_init, 
            device=self.model.p_logit.device, 
            dtype=self.dtype
        )
        assert p_init.shape == self.model.p_logit.shape, "p_init must match shape of model's p_logit!"
        
        with torch.no_grad():
            self.model.p_logit.copy_(torch.logit(p_init))

        self.model.p_logit.requires_grad_(requires_grad)

    def update_n(self, n_init: np.ndarray, requires_grad: bool = True) -> None:
        """
        Update the fixed generation count parameter (n_logit) in the model.

        Used to set a known or estimated cellular generation profile as prior.
        Only available when train_transpose=True.

        Args:
            n_init: Array of initial generation counts per cell, shape (n_cells,).
                    Values should not be in logit space.
            requires_grad: Whether to enable gradient computation for this parameter during training.
                          If False, n_logit will remain frozen.

        Raises:
            AssertionError: If train_transpose=False (n_logit does not exist)
            AssertionError: If input shape does not match model's n_logit
        """
        assert self.train_transpose, 'Parameter n only exists in model with train_transpose=True!'
        n_init = _input_tensor(
            n_init, 
            device=self.model.n.device, 
            dtype=self.dtype
        )
        assert n_init.shape == self.model.n.shape, "n_init must match shape of model's n!"
        
        with torch.no_grad():
            self.model.n_logit.copy_(
                _softplus_inverse(n_init)
            )

        self.model.n_logit.requires_grad_(requires_grad)

    def load_data(
        self,
        X: np.ndarray,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        dtype: Optional[torch.dtype] = None,
        shuffle: bool = True,
        best_n_or_p: Optional[np.ndarray] = None
    ) -> None:
        """
        Load training data into DataLoader with optional transposition and auxiliary variables.

        Handles both standard and transposed modes based on train_transpose flag.

        Args:
            X: Input mutation matrix of shape (n_cells, n_sites).
            batch_size: Size of each training batch.
            num_workers: Number of subprocesses for data loading.
            pin_memory: Whether to pin CPU memory for faster GPU transfer.
            dtype: Target tensor data type.
            shuffle: Whether to shuffle batches during training.
            best_n_or_p: Optional auxiliary variable used in some training phases:
                          - When train_transpose=False: provides initial guess for N (cell-wise)
                          - When train_transpose=True: provides initial guess for P (site-wise)

        Raises:
            ValueError: If input shape doesn't match expected dimensions.
        """
        if X.shape[self.feature_dim] != self.input_dim:
            raise ValueError(f"Input data must have {self.input_dim} features, but got {X.shape[self.feature_dim]} features.")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        if dtype is not None:
            if self.dtype is None:
                self.dtype = dtype
            elif dtype != self.dtype:
                logger.info(f"Try to load data as {dtype}, while model used {self.dtype}! Use model's one.")
        self.X = _input_tensor(X, dtype=self.dtype)
        X_data = self.X.transpose(0,1) if self.train_transpose else self.X

        if best_n_or_p is None:
            self._best_n_or_p = None
        else:
            if X_data.shape[0] != len(best_n_or_p):
                raise ValueError(f"Input best_n_or_p must have {X_data.shape[0]} features, but got {len(best_n_or_p)}.")
            self._best_n_or_p = _input_tensor(best_n_or_p, dtype=self.dtype)

        # load (shuffle) X and search optimal batch_size if out of memory
        try:
            self.data_loader = self._create_dataloader(
                X_data, self.batch_size, shuffle, sample_related_vars=self._best_n_or_p
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
                    X_data, optimal_batch, shuffle, sample_related_vars=self._best_n_or_p
                )
            else:
                raise e

        # load ordered X
        if shuffle:
            self.data_loader_order = self._create_dataloader(
                X_data, self.batch_size, shuffle=False, sample_related_vars=self._best_n_or_p
            )
        else:
            self.data_loader_order = self.data_loader

    def load_valid_data(self, Xvalid: np.ndarray) -> None:
        """
        Load validation data into DataLoader.

        Applies same transposition logic as load_data.

        Args:
            Xvalid: Validation mutation matrix of shape matching training data.

        Raises:
            ValueError: If input shape doesn't match expected dimensions.
        """
        if Xvalid.shape[self.feature_dim] != self.input_dim:
            raise ValueError(f"Input data must have {self.input_dim} features, but got {Xvalid.shape[self.feature_dim]} features.")
        self.Xvalid = _input_tensor(Xvalid, dtype=self.dtype)
        Xvalid_data = self.Xvalid.transpose(0,1) if self.train_transpose else self.Xvalid

        self.valid_loader = self._create_dataloader(
            Xvalid_data, self.batch_size, shuffle=False, sample_related_vars=self._best_n_or_p
        )

    def set_mode(self, mode: Literal["xhat", "np"], type: Optional[Literal["mode1", "mode2"]] = None) -> None:
        '''
        Set the operation mode and type for the model.
        
        Configures the model state, optimizer, and loss function for either:
        - Estimating latent parameters (N/P): mode='np'
        - Reconstructing the mutation matrix: mode='xhat'

        Supports two distinct training sequences:
        - mode1: first train np, then xhat
        - mode2: first train xhat, then np

        Parameters:
            mode (str): Specifies the current operation mode. Must be one of the following:
                        - 'xhat': Indicates running in xhat mode.
                        - 'np'  : Indicates running in np mode.
            type (str, optional): Specifies the operation type. Must be one of the following:
                        - 'mode1': Indicates mode1 sequence: first run mode='np', then mode='xhat'.
                        - 'mode2': Indicates mode2 sequence: first run mode='xhat', then mode='np'.
                        - None   : If not specified, defaults to self.mode_type.
                                   If self.mode_type is also None, defaults to 'mode1'.

        Notes:
            - If type is None and self.mode_type is None, type is set to 'mode1' by default.
            - Depending on the value of type ('mode1' or 'mode2'), the model's state, optimizer, and loss function are configured:
                - For mode1:
                    - Calls self.model.set_mode1(mode).
                    - Uses self.optimizer1 as the optimizer for baseVAE.
                    - Uses self._loss_mode1_xhat as the loss function for baseVAE.
                - For mode2:
                    - Calls self.model.set_mode2(mode).
                    - Uses self.optimizer2 as the optimizer for baseVAE.
                    - Uses self._loss_mode2_xhat as the loss function for baseVAE.
        '''
        assert mode in ['xhat','np']
        assert type in ['mode1','mode2',None]
        if type is None:
            if self.mode_type is None:
                logger.info("Use mode1 as default, where mode1: np(+z) -> xhat, mode2: xhat(+z) -> np")
                self.mode_type = 'mode1'
        else:
            self.mode_type = type

        if self.mode_type == 'mode1':
            self.model.set_mode1(mode)
            self.optimizer = self.optimizer1
            self.loss = self._loss_mode1_xhat
        else:
            self.model.set_mode2(mode)
            self.optimizer = self.optimizer2
            self.loss = self._loss_mode2_xhat

    def _reconstruct(self, n_or_p: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Generate reconstructed mutation matrix from predicted N or P.

        Computes R = 1 - (1-P)^N using numerically stable operations.

        Args:
            n_or_p: Predicted latent parameter from decoder:
                     - If train_transpose=False: softplus(n_logit), shape (batch, 1)
                     - If train_transpose=True: sigmoid(p_logit), shape (batch, 1)
            sample: If True, returns binary samples via Bernoulli draw;
                    if False, returns continuous probabilities.

        Returns:
            Reconstructed mutation matrix of shape (batch, n_features).
        """
        if self.train_transpose:
            n = F.softplus(self.n_logit)
            p = torch.sigmoid(n_or_p)
        else:
            n = F.softplus(n_or_p)
            p = torch.sigmoid(self.p_logit)
            self.p = p

        # Use Softplus activation for better gradient flow
        # p = F.softplus(p_logit) / (1 + F.softplus(p_logit))

        r = computeR_by_np(n=n, p=p, eps=self._eps)

        if sample:
            x = torch.bernoulli(r)
            return x

        return r
    
    def _loss_mode2_xhat(
        self,
        x: torch.Tensor,
        xhat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        qz: Optional[torch.distributions.Normal],
        z: torch.Tensor,
        *args, **kwargs
    ) -> torch.Tensor:
        """
        Loss function for xhat phase in mode2 training sequence.

        Combines reconstruction loss (BCE) and KL divergence.

        Args:
            x: Ground truth mutation matrix.
            xhat: Reconstructed matrix.
            mu: Latent mean.
            logvar: Latent log-variance.
            qz: Approximate posterior distribution.
            z: Sampled latent vector.

        Returns:
            Total loss scalar combining BCE and KL terms.
        """
        batch_size = x.size(0)
        mask = x != self.miss_value # True for non-missing

        # reconstruction X
        BCE = F.binary_cross_entropy(xhat[mask], x[mask], reduction='sum') / (mask.sum() + self._eps)
        # normal Z
        KLD = self._kl_loss(mu, logvar) / batch_size
        # KLD = self._kl_loss_by_qz(z, qz) / batch_size

        # Store individual losses for monitoring
        total_loss = BCE + self.beta * KLD
        self.current_metrics = {
            'reconstruction_loss': BCE.item(),
            'kl_loss': KLD.item(),
            'total_loss':  total_loss.item()
        }

        return total_loss

    def _loss_mode1_xhat(
        self,
        x: torch.Tensor,
        xhat: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        qz: Optional[torch.distributions.Normal],
        z: torch.Tensor,
        *args, **kwargs
    ) -> torch.Tensor:
        """
        Compute reconstruction loss for xhat phase in mode1 training sequence.

        In mode1, KL divergence is disabled during the xhat phase to allow flexible reconstruction
        without strong regularization on latent space structure.

        Args:
            x: Ground truth mutation matrix.
            xhat: Reconstructed matrix from decoder.
            mu: Latent mean vector.
            logvar: Log-variance of latent distribution.
            qz: Approximate posterior distribution object.
            z: Sampled latent vector.

        Returns:
            Total loss scalar based solely on reconstruction error (BCE).
        """
        batch_size = x.size(0)
        mask = x != self.miss_value # True for non-missing

        # reconstruction X
        BCE = F.binary_cross_entropy(xhat[mask], x[mask], reduction='sum') / (mask.sum() + self._eps)

        # Store individual losses for monitoring
        total_loss = BCE
        self.current_metrics = {
            'reconstruction_loss': BCE.item(),
            'total_loss':  total_loss.item()
        }

        return total_loss
    
    def _loss_mode2_np(
        self,
        x: torch.Tensor,
        n_or_p: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        qz: Optional[torch.distributions.Normal],
        z: torch.Tensor,
        xhat_logit: torch.Tensor,
        best_n_or_p: Optional[torch.Tensor] = None,
        *args, **kwargs
    ) -> torch.Tensor:
        """
        Loss function for np estimation phase in mode2 training sequence.

        Focuses on accurate prediction of N or P while keeping other components fixed.
        Does not include KL term since encoder is frozen.

        Args:
            x: Input mutation data.
            n_or_p: Predicted latent parameter (n_logit or p_logit).
            mu: Latent mean.
            logvar: Latent log-variance.
            qz: Posterior distribution.
            z: Sampled latent code.
            xhat_logit: Reconstructed values using current N/P estimate.
            best_n_or_p: Optional reference value for consistency loss (e.g., from previous stage).

        Returns:
            Total loss combining reconstruction and optional consistency terms.
        """
        if self.train_transpose:
            p = torch.sigmoid(n_or_p)
            n = F.softplus(self.n_logit)
        else:
            n = F.softplus(n_or_p)
            p = self.p

        batch_size = x.size(0)
        mask = x != self.miss_value # True for non-missing

        # reconstruction X
        BCE_np = F.binary_cross_entropy(xhat_logit[mask], x[mask], reduction='sum') / (mask.sum() + self._eps)
        
        # each type
        wt_loc = x == self.wt_value
        edit_loc = x == self.edit_value
        miss_loc = ~mask
        BCE1_np = F.binary_cross_entropy(xhat_logit[wt_loc], x[wt_loc], reduction='sum') / (wt_loc.sum() + self._eps)
        BCE2_np = F.binary_cross_entropy(xhat_logit[edit_loc], x[edit_loc], reduction='sum') / (edit_loc.sum() + self._eps)
        BCE3_np = F.binary_cross_entropy(xhat_logit[miss_loc], (x-self.miss_value+0.5)[miss_loc], reduction='sum') / (miss_loc.sum() + self._eps)

        if self.use_weight:
            BCE_np = BCE1_np * self.wt_loss_weight + BCE2_np * self.edit_loss_weight

        # Store individual losses for monitoring
        total_loss = BCE_np
        self.current_metrics_np = {
            'reconstruction_loss_np': BCE_np.item(),
            'reconstruction_loss_np_wt': BCE1_np.item(),
            'reconstruction_loss_np_edit': BCE2_np.item(),
            'reconstruction_loss_np_miss': BCE3_np.item(),
            'total_loss':  total_loss.item()
        }

        # best n or p
        if best_n_or_p is not None:
            if self.train_transpose: # best p
                _x = p
                _name = 'best_p_loss'
            else: # best n
                _x = n
                _name = 'best_n_loss'

            best_loss = torch.mean((best_n_or_p - _x) ** 2)
            total_loss = BCE_np + self.beta_best * best_loss
            self.current_metrics_np[_name] = best_loss.item()

        return total_loss
    
    def _loss_mode1_np(
        self,
        x: torch.Tensor,
        n_or_p: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        qz: Optional[torch.distributions.Normal],
        z: torch.Tensor,
        xhat_logit: torch.Tensor,
        best_n_or_p: Optional[torch.Tensor] = None,
        *args, **kwargs
    ) -> torch.Tensor:
        """
        Loss function for np estimation phase in mode1 training sequence.

        Includes both reconstruction loss and KL divergence, as encoder is trainable.
        Also supports consistency loss against prior estimates.

        Args:
            x: Input mutation data.
            n_or_p: Predicted latent parameter (n_logit or p_logit).
            mu: Latent mean.
            logvar: Latent log-variance.
            qz: Approximate posterior distribution.
            z: Sampled latent vector.
            xhat_logit: Reconstructed values using current N/P estimate.
            best_n_or_p: Optional reference value for consistency loss.

        Returns:
            Total loss scalar including reconstruction, KL, and optional consistency terms.
        """
        if self.train_transpose:
            p = torch.sigmoid(n_or_p)
            n = F.softplus(self.n_logit)
        else:
            n = F.softplus(n_or_p)
            p = self.p

        batch_size = x.size(0)
        mask = x != self.miss_value # True for non-missing

        # reconstruction X
        BCE_np = F.binary_cross_entropy(xhat_logit[mask], x[mask], reduction='sum') / (mask.sum() + self._eps)
        # normal Z
        KLD = self._kl_loss(mu, logvar) / batch_size
        # KLD = self._kl_loss_by_qz(z, qz) / batch_size
        
        # each type
        wt_loc = x == self.wt_value
        edit_loc = x == self.edit_value
        miss_loc = ~mask
        BCE1_np = F.binary_cross_entropy(xhat_logit[wt_loc], x[wt_loc], reduction='sum') / (wt_loc.sum() + self._eps)
        BCE2_np = F.binary_cross_entropy(xhat_logit[edit_loc], x[edit_loc], reduction='sum') / (edit_loc.sum() + self._eps)
        BCE3_np = F.binary_cross_entropy(xhat_logit[miss_loc], (x-self.miss_value+0.5)[miss_loc], reduction='sum') / (miss_loc.sum() + self._eps)

        if self.use_weight:
            BCE_np = BCE1_np * self.wt_loss_weight + BCE2_np * self.edit_loss_weight

        # Store individual losses for monitoring
        total_loss = BCE_np + self.beta * KLD
        self.current_metrics_np = {
            'reconstruction_loss_np': BCE_np.item(),
            'reconstruction_loss_np_wt': BCE1_np.item(),
            'reconstruction_loss_np_edit': BCE2_np.item(),
            'reconstruction_loss_np_miss': BCE3_np.item(),
            'kl_loss': KLD.item(),
            'total_loss':  total_loss.item()
        }

        # best n or p
        if best_n_or_p is not None:
            if self.train_transpose: # best p
                _x = p
                _name = 'best_p_loss'
            else: # best n
                _x = n
                _name = 'best_n_loss'

            best_loss = torch.mean((best_n_or_p - _x) ** 2)
            total_loss += best_loss * self.beta_best
            self.current_metrics_np[_name] = best_loss.item()

        return total_loss

    def _train_epoch_np(self) -> Dict[str, float]:
        """
        Train one epoch in np estimation mode.

        Updates either N or P depending on train_transpose setting.

        Returns:
            Dictionary of average metrics for this epoch (e.g., reconstruction_loss_np, kl_loss).
        """
        self.model.train()
        total_metrics = defaultdict(float)
        n_samples = 0

        loss_np_func = self._loss_mode1_np if self.mode_type=='mode1' else self._loss_mode2_np
        optimizer_np = self.optimizer_np1 if self.mode_type=='mode1' else self.optimizer_np2

        for batch in self.data_loader:#
            optimizer_np.zero_grad()
            x = batch[0].to(self.device)

            if len(batch) == 2:
                best_n_or_p = batch[1].to(self.device)
            else:
                best_n_or_p = None

            batch_size = x.size(0)
            n_samples += batch_size

            outs = self.model(x, train_np=True) # (x, n_logit/p_logit, mu, logvar, qz, z)
            if outs[-2] is None: # VAE: qz is None, when mu/logvar has nan
                return dict(total_loss=float('nan'))

            xhat_logit = self._reconstruct(n_or_p=outs[1], sample=False) # xhat by n and p
            loss = loss_np_func(*outs, xhat_logit=xhat_logit, best_n_or_p=best_n_or_p)
            loss.backward()
            optimizer_np.step()
            
            for k, v in self.current_metrics_np.items():
                total_metrics[k] += v * batch_size
        
        avg_metrics = {k: v/n_samples for k, v in total_metrics.items()}
        self.train_metrics_np.append(avg_metrics)
        return avg_metrics

    def _validate_np(self) -> Dict[str, float]:
        """
        Validate model performance in np estimation mode.

        Evaluates loss on validation set without updating parameters.

        Returns:
            Average metric values on validation data.
        """
        self.model.eval()
        total_metrics = defaultdict(float)
        n_samples = 0

        loss_np_func = self._loss_mode1_np if self.mode_type=='mode1' else self._loss_mode2_np
        optimizer_np = self.optimizer_np1 if self.mode_type=='mode1' else self.optimizer_np2

        with torch.no_grad():
            for batch in self.valid_loader:
                # 1. remove previous grad
                optimizer_np.zero_grad()

                # 2. run epoch
                x = batch[0].to(self.device)
                batch_size = x.size(0)
                n_samples += batch_size
                outs = self.model(x, train_np=True) # (x, n_logit/p_logit, mu, logvar, qz, z)
                if outs[-2] is None: # VAE: qz is None, when mu/logvar has nan
                    return dict(total_loss=float('nan'))

                xhat_logit = self._reconstruct(n_or_p=outs[1], sample=False) # xhat by n and p

                # 3. loss
                if len(batch) == 2:
                    best_n_or_p = batch[1].to(self.device)
                else:
                    best_n_or_p = None
                _ = loss_np_func(*outs, xhat_logit=xhat_logit, best_n_or_p=best_n_or_p)
                # self.current_metrics_np updates when computing loss

                # 4. Accumulate metrics
                for k, v in self.current_metrics_np.items():
                    total_metrics[k] += v * batch_size
        
        # Calculate average metrics
        avg_metrics = {
            k: v/n_samples 
            for k, v in total_metrics.items()
        }
        # Store validation metrics
        self.valid_metrics_np.append(avg_metrics)
        return avg_metrics

    def train_np(
        self,
        use_tqdm: bool = False,
        patience: int = 45,
        min_delta: Optional[float] = None,
        verbose: bool = True,
        n_init: Optional[np.ndarray] = None,
        p_init: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[float]:
        """
        Train the model to estimate latent parameters N or P.

        Executes multi-epoch training with early stopping.

        Args:
            use_tqdm: Show progress bar.
            patience: Early stopping epochs.
            min_delta: Minimum loss improvement threshold.
            verbose: Print logging messages.
            n_init: Initial guess for N (used if train_transpose=True).
            p_init: Initial guess for P (used if train_transpose=False).

        Returns:
            List of total loss values per epoch.
        """
        if self.mode_type is None:
            raise ValueError("Please run set_mode() first!")

        if min_delta is None:
            min_delta = self._eps

        # for initial params
        if self.train_transpose:
            if n_init is not None:
                self.update_n(n_init, requires_grad=True)
        else:
            if p_init is not None:
                self.update_p(p_init, requires_grad=True)

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
        self.train_metrics_np = []
        self.valid_metrics_np = []

        for epoch in range(self.num_epochs):
            self._epoch = epoch
            # train
            train_metrics = self._train_epoch_np()
            current_loss = train_metrics['total_loss']
            
            # valid if input
            if self.valid_loader is not None:
                valid_metrics = self._validate_np()
                current_loss = valid_metrics['total_loss']

            # save loss
            losses.append(current_loss)

            # logging
            _log(epoch, current_loss)

            if self._check_nan(optimizer=self.optimizer,loss=current_loss):
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

        self.Z, self.Xhat_np = self._get_reconstructions_np(sample=True)

        self._clear()
        return losses

    def _get_reconstructions(self, sample: bool = True) -> Tuple[np.ndarray, np.ndarray]: # rewrite for baseVAE
        """
        Get latent representations and reconstructed mutation matrix after full training.

        Uses trained decoder to generate Xhat from Z. Optionally returns sampled binary outputs.

        Args:
            sample: If True, applies Bernoulli sampling to reconstruct discrete mutations;
                    if False, returns continuous probabilities.

        Returns:
            tuple: (Z_latent, Xhat_reconstructed), each as numpy arrays.
        """
        if sample:
            set_seed(self.seed)

        self.model.eval()
        z_list, xhat_list = [], []
        
        with torch.no_grad():
            for batch in self.data_loader_order: # use right order but not shuffle!
                x = batch[0].to(self.device)
                outs = self.model(x, train_np=False) # (x, x_hat, mu, logvar, qz, z)
                z, x_hat = outs[2], outs[1] # z should be mu for VAE

                # use sample
                if sample:
                    x_hat = torch.bernoulli(x_hat)

                z_list.append(z.cpu())
                xhat_list.append(x_hat.cpu())

        Z = torch.cat(z_list, dim=0).numpy()
        Xhat = torch.cat(xhat_list, dim=0).numpy()
        if self.train_transpose:
            Xhat = Xhat.transpose(0,1)

        return Z, Xhat

    def _get_reconstructions_np(self, sample: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get latent representations and reconstructions during np-phase training.

        Computes R = 1 - (1-P)^N using final N/P estimates. Can also return decoded Z.

        Args:
            sample: Whether to apply Bernoulli sampling to output matrix.

        Returns:
            tuple: (Z, Xhat) arrays of shape (n_samples, z_dim) and (n_samples, input_dim)
                   Also sets self.N, self.P, self.Xhat_np attributes.
        """
        if sample:
            set_seed(self.seed)

        self.model.eval()
        z_list, xhat_list, n_or_p_list = [], [], []

        with torch.no_grad():
            for batch in self.data_loader_order:
                x = batch[0].to(self.device)
                
                outs = self.model(x, train_np=True) # (x, n_logit/p_logit, mu, logvar, qz, z)
                n_or_p, mu = outs[1:3]

                if self.mode_type == 'mode1':
                    xhat = self._reconstruct(n_or_p, sample=sample)
                else:
                    outs = self.model(x, train_np=False) # (x, xhat, mu, logvar, qz, z)
                    xhat = torch.bernoulli(outs[1]) if sample else outs[1]

                z_list.append(mu.cpu())
                xhat_list.append(xhat.cpu())
                n_or_p_list.append(n_or_p.cpu())
        
        # Convert to numpy arrays
        Z = torch.cat(z_list, dim=0).numpy()
        Xhat = torch.cat(xhat_list, dim=0).numpy()

        if self.train_transpose:
            self.P = torch.sigmoid(torch.cat(n_or_p_list, dim=0)).numpy().flatten()
            self.N = F.softplus(self.n_logit).cpu().detach().numpy().flatten()
            Xhat = Xhat.transpose(0,1)
        else:
            self.N = F.softplus(torch.cat(n_or_p_list, dim=0)).numpy().flatten()
            self.P = self.p.cpu().detach().numpy().flatten()

        return Z, Xhat

    def _get_n(self, Z: np.ndarray) -> np.ndarray:
        """
        Extract predicted cellular generation (N) from latent representation Z.

        Only valid when train_transpose=False, where the encoder was trained to predict N

        Args:
            Z: Latent matrix of shape (n_cells, z_dim)

        Returns:
            Array of N values (cell-wise generation) of shape (n_cells,)

        Raises:
            ValueError: If called in transposed mode (train_transpose=True).
        """
        if self.train_transpose:
            raise ValueError("Cannot compute n from Z when train_transpose=True!")
        Z = _input_tensor(Z, device=self.device)
        n_logit = self.model.decoder_n(Z)
        n = F.softplus(n_logit).cpu().detach().numpy().flatten()
        return n

    def finetune_n(
        self,
        max_n: int,
        use_nmf: bool = False,
        verbose: bool = True,
        **optim_kwargs
    ) -> Dict[float, float]:
        """
        Finely tune the estimated cellular generation (N) by optimizing scaling factor.

        Converts continuous N scores into integer division counts that minimize rounding error.

        Args:
            max_n: Target maximum integer value for scaled N.
            use_nmf: If True, uses N from NMF stage; otherwise uses VAE-estimated N.
            verbose: Print progress logs.
            **optim_kwargs: Additional arguments passed to optimize_integer_scaling:
                max_error_ratio=0.1, 
                tolerance_ratio=0.01, 
                max_iter=1000, 
                num_random=50, 
                seed=42,

        Returns:
            Dictionary recording tested scaling factors and their corresponding errors.
        """
        # get N
        if use_nmf:
            # use self.N_nmf by train()
            if self.N_nmf is None:
                raise ValueError('Please run train() first!')
            n = self.N_nmf
        else:
            if self.N is None:
                raise ValueError('Please run train() or train_nmf() first!')
            n = self.N

        # process
        n_round_optimal, k_optimal, loss_records = optimize_integer_scaling(
            n=n, 
            max_n=max_n, 
            seed=self.seed, 
            **optim_kwargs
        )

        self.N_ft = n_round_optimal
        self.k_optimal = k_optimal
        if verbose:
            logger.info('Finished! Get finetuned N by attribute .N_ft.')
        return loss_records

    def finetune_p(
        self,
        use_n: Optional[str] = None,
        lr: Optional[float] = None,
        patience: int = 45,
        min_delta: Optional[float] = None,
        unlimited_epoch: bool = True,
        logging_interval: int = 100,
        use_tqdm: bool = True,
        verbose: bool = True,
    ) -> List[float]:
        """
        Refine estimated mutation rates (P) using fixed N.

        Runs one-sided NMF optimization where only P is updated.

        Args:
            use_n: Which N to fix:
                   - None: use VAE-estimated N
                   - 'nmf': use N from NMF stage
                   - 'finetune': use integer-scaled N
            lr: Learning rate.
            patience: Early stopping patience.
            min_delta: Minimum improvement threshold.
            unlimited_epoch: Run until convergence.
            logging_interval: Logging frequency.
            use_tqdm: Show progress bar.
            verbose: Print logs.

        Returns:
            Training loss history.
        """
        lr = self.lr if lr is None else lr
        min_delta = self._eps if min_delta is None else min_delta
        max_epoch = None if unlimited_epoch else self.num_epochs_nmf
        if max_epoch is None and use_tqdm:
            use_tqdm = False
            if verbose:
                logger.warning("Set use_tqdm=False when max_epoch is None.")

        # get N
        if use_n is None:
            # use self.N by train()
            if self.N is None:
                raise ValueError('Please run train() first!')
            n = self.N
            p = self.P
        elif use_n=='nmf':
            # use self.N_nmf by train_nmf()
            if self.N_nmf is None:
                raise ValueError('Please run train() first!')
            n = self.N_nmf
            p = self.P_nmf
        elif use_n=='finetune':
            if self.N_ft is None:
                raise ValueError('Please run finetune_n() first!')
            n = self.N_ft
            p = self.P if self.P is not None else self.P_nmf
        else:
            raise ValueError('Only one of None, "nmf" and "finetune" is available for use_n!')
        
        # train
        _n, fine_p, losses = decompose_R_to_np(
            R=self.X, n=n, p=p, only_train_p=True, miss_value=self.miss_value, 
            lr=lr, patience=patience, min_delta=min_delta, eps=self._eps, 
            max_epoch=max_epoch, logging_interval=logging_interval, verbose=verbose,
            device=self.device, dtype=self.dtype, use_tqdm=use_tqdm
        )

        self.P_ft = fine_p
        if verbose:
            logger.info('Finished! Get finetuned P by attribute .P_ft.')
        return losses

    def train_nmf(
        self,
        X: Optional[np.ndarray] = None,
        n_init: Optional[np.ndarray] = None,
        p_init: Optional[np.ndarray] = None,
        lr: Optional[float] = None,
        patience: int = 45,
        min_delta: Optional[float] = None,
        unlimited_epoch: bool = True,
        logging_interval: int = 100,
        use_tqdm: bool = True,
        verbose: bool = True,
        use_batch: bool = True
    ) -> List[float]:
        """
        Train using pure NMF decomposition without VAE.

        Solves R ≈ 1 - (1-P)^N directly via gradient descent.

        Args:
            X: Input mutation matrix. If None, uses stored X.
            n_init: Initial guess for N.
            p_init: Initial guess for P.
            lr: Learning rate.
            patience: Early stopping patience.
            min_delta: Improvement threshold.
            unlimited_epoch: Run until convergence.
            logging_interval: Logging interval.
            use_tqdm: Show progress bar.
            verbose: Print logs.
            use_batch: Use mini-batch training.

        Returns:
            Loss history.
        """
        if X is None:
            X = self.X
        else:
            self.X = X
        lr = self.lr if lr is None else lr
        min_delta = self._eps if min_delta is None else min_delta
        max_epoch = None if unlimited_epoch else self.num_epochs_nmf
        if max_epoch is None and use_tqdm:
            use_tqdm = False
            if verbose:
                logger.warning("Set use_tqdm=False when max_epoch is None.")

        # train
        kwargs=dict(
            R=X, n=n_init, p=p_init, only_train_p=False, miss_value=self.miss_value, 
            lr=lr, patience=patience, min_delta=min_delta, eps=self._eps, 
            max_epoch=max_epoch, logging_interval=logging_interval, verbose=verbose,
            device=self.device, dtype=self.dtype, use_tqdm=use_tqdm,
            batch_size=self.batch_size if use_batch else None, seed=self.seed,
        )

        n, p, losses = decompose_R_to_np(**kwargs)
        
        self.N_nmf = n
        self.P_nmf = p
        return losses

    def compute_R(self, n: np.ndarray, p: np.ndarray, sample: bool = False) -> np.ndarray:
        """
        Compute expected mutation matrix R from given N and P.

        Evaluates R = 1 - (1-P)^N numerically stably.

        Args:
            n: Cellular generation array of length n_cells.
            p: Mutation rate array of length n_sites.
            sample: Whether to sample from Bernoulli(R) or return probability.

        Returns:
            Computed mutation matrix of shape (n_cells, n_sites).
        """
        for x in [n,p]:
            assert isinstance(x, (torch.Tensor, np.ndarray)) and x.ndim == 1
        n_tensor = torch.Tensor(n).reshape([n.shape[0], 1])
        p_tensor = torch.Tensor(p)

        R = computeR_by_np(n=n_tensor, p=p_tensor, eps=self._eps)

        if sample:
            g = torch.Generator()
            g.manual_seed(self.seed)
            R = torch.bernoulli(R, generator=g)

        return R

    def posterior_X(
        self,
        n: np.ndarray,
        p: np.ndarray,
        X: Optional[np.ndarray] = None,
        posterior_threshold: float = 0.95
    ) -> np.ndarray:
        """
        Denoise mutation matrix using Bayesian posterior probability.

        This method evaluates the confidence of each observed mutation call based on:
        - Estimated cellular generation (N)
        - Site-specific mutation rate (P)
        - Cumulative mutation model R = 1 - (1-P)^N

        It corrects low-confidence calls by reassigning them to wild-type status.

        Args:
            n: Estimated cellular generation array, shape (n_cells,).
            p: Estimated mutation rate array, shape (n_sites,).
            X: Observed mutation matrix; if None, uses stored X.
            posterior_threshold: Threshold for calling a mutation confident.
                                 Only posterior >= threshold will be kept as edit_value.

        Returns:
            Denoised mutation matrix with corrected calls:
                - high-confidence mutations -> edit_value
                - low-confidence mutations -> wt_value
                - missing values -> set to miss_value (unchanged)
        """
        if X is None:
            X = self.X
        mask = X==self.miss_value

        R = self.compute_R(n=n, p=p, sample=False)

        posterior_mtx = compute_posterior(
            r=X, 
            r0=0, 
            r1=R, 
            p_prior=p, 
            n=1000, 
            eps=self._eps, 
            mask_value=self.miss_value
        )

        denoise_X = np.where(
            mask,
            self.miss_value,
            np.where(
                posterior_mtx >= posterior_threshold, 
                self.edit_value, 
                self.wt_value
            )
        )

        return denoise_X


# X # change
# model = MutModel(
#     input_dim=X.shape[1],
#     hidden_dims=[128], # change 
#     z_dim=20,          # change
#     num_epochs=10000,
#     use_batch_norm='encoder',
#     use_layer_norm='both',
# )
# model.load_data(
#     X=X, 
#     batch_size=10000, 
#     num_workers=4, 
#     pin_memory=True, 
# )
# model.set_mode('np')
# losses_np = model.train_np(
#     patience=100,
#     use_tqdm=True
# )