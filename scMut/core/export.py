import warnings
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix, csr_matrix
from anndata import AnnData
import torch
from .scMut import MutModel
from .utils import _input_tensor
from .typing import Optional, Dict


def _asarray(values):
    if isinstance(values, (list, tuple, np.ndarray)):
        return np.asarray(values)
    elif isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    elif isinstance(values, (pd.DataFrame, pd.Series)):
        return values.values
    elif isinstance(values, spmatrix):
        return values.toarray()
    else:
        try:
            return np.asarray(values)
        except Exception as e:
            raise ValueError(f"Unsupported input type: {type(values)}:\n{e}")


def _exist_attr(obj, attr_name):
    if (
        hasattr(obj, attr_name) and 
        getattr(obj, attr_name) is not None
    ):
        return True, getattr(obj, attr_name)
    else:
        return False, None


def save_model_to_adata(
    model_n: Optional[MutModel] = None,
    model_p: Optional[MutModel] = None,
    sparse: bool = False,
    record_loss: bool = False
) -> AnnData:
    input_n = model_n is not None
    input_p = model_p is not None

    if not (input_n or input_p):
        raise ValueError("At least one model must be provided.")
    if input_p and not model_p.train_transpose:
        raise ValueError("model_p must be trained with train_transpose=True.")

    if input_n:
        X = _asarray(model_n.X)
        mask = X == model_n.miss_value
    elif input_p:
        X = _asarray(model_p.X)
        mask = X == model_p.miss_value

    if sparse:
        X = csr_matrix(X)
        mask = csr_matrix(mask)

    # collect outputs
    obs = {}
    var = {}
    obsm = {}
    varm = {}
    layers = { 'missing_mask': mask }
    uns = {
        'method': {
            "project": "scMut",
            "version": "0.1.0",
            "description": "Single-cell mutation parser",
            "note": (
                "Attributes suffixed with '_modelN' come from a model trained with train_transpose=False, "
                "where cells are observations and sites are features. "
                "Those with '_modelP' come from a model trained with train_transpose=True, "
                "where sites are treated as pseudo-observations. "
                "Suffixes are only added when both models contribute to the same field to avoid conflict."
            ),
            'input_n': input_n,
            'input_p': input_p,
            'data_sparse': sparse
        }
    }
    use_suffix = input_n and input_p

    for _attr in ['N', 'N_nmf', 'N_ft']:
        if input_n:
            _bool, _values = _exist_attr(model_n, _attr)
            if _bool:
                _key = f"{_attr}_modelN" if use_suffix else _attr
                obs[_key] = _asarray(_values)

        if input_p:
            _bool, _values = _exist_attr(model_p, _attr)
            if _bool:
                _key = f"{_attr}_modelP" if use_suffix else _attr
                obs[_key] = _asarray(_values)

    for _attr in ['P', 'P_nmf', 'P_ft']:
        if input_n:
            _bool, _values = _exist_attr(model_n, _attr)
            if _bool:
                _key = f"{_attr}_modelN" if use_suffix else _attr
                var[_key] = _asarray(_values)

        if input_p:
            _bool, _values = _exist_attr(model_p, _attr)
            if _bool:
                _key = f"{_attr}_modelP" if use_suffix else _attr
                var[_key] = _asarray(_values)

    for _attr in ['Z']:
        if input_n:
            _bool, _values = _exist_attr(model_n, _attr)
            if _bool:
                obsm[_attr] = _asarray(_values)

        if input_p:
            _bool, _values = _exist_attr(model_p, _attr)
            if _bool:
                varm[_attr] = _asarray(_values)

    for _attr in ['Xhat', 'Xhat_np']:
        if input_n:
            _bool, _values = _exist_attr(model_n, _attr)
            if _bool:
                _key = f"{_attr}_modelN" if use_suffix else _attr
                layers[_key] = csr_matrix(_asarray(_values)) if sparse else _asarray(_values)

        if input_p:
            _bool, _values = _exist_attr(model_p, _attr)
            if _bool:
                _key = f"{_attr}_modelP" if use_suffix else _attr
                layers[_key] = csr_matrix(_asarray(_values).T) if sparse else _asarray(_values).T

    if input_n:
        _dict = uns['method'].setdefault('model_n', {})

        if hasattr(model_n, 'k_optimal'):
            _dict["k_optimal"] = model_n.k_optimal

        _dict['miss_value'] = model_n.miss_value
        _dict['wt_value'] = model_n.wt_value
        _dict['edit_value'] = model_n.edit_value

        _dict['VAE'] = _exist_attr(model_n, 'N')[0] and _exist_attr(model_n, 'P')[0]
        _dict['NMF'] = _exist_attr(model_n, 'N_nmf')[0] and _exist_attr(model_n, 'P_nmf')[0]
        _dict['FT'] = _exist_attr(model_n, 'N_ft')[0] and _exist_attr(model_n, 'P_ft')[0]
        _dict['Xhat'] = _exist_attr(model_n, 'Xhat')[0]
        _dict['Xhat_np'] = _exist_attr(model_n, 'Xhat_np')[0]

        if record_loss:
            for _x in ['train_metrics', 'train_metrics_np']:
                _bool, _values = _exist_attr(model_n, _x)
                if _bool:
                    _dict[_x] = _values

    if input_p:
        _dict = uns['method'].setdefault('model_p', {})

        if hasattr(model_p, 'k_optimal'):
            _dict["k_optimal"] = model_p.k_optimal

        _dict['miss_value'] = model_p.miss_value
        _dict['wt_value'] = model_p.wt_value
        _dict['edit_value'] = model_p.edit_value

        _dict['VAE'] = _exist_attr(model_p, 'N')[0] and _exist_attr(model_p, 'P')[0]
        _dict['NMF'] = _exist_attr(model_p, 'N_nmf')[0] and _exist_attr(model_p, 'P_nmf')[0]
        _dict['FT'] = _exist_attr(model_p, 'N_ft')[0] and _exist_attr(model_p, 'P_ft')[0]
        _dict['Xhat'] = _exist_attr(model_p, 'Xhat')[0]
        _dict['Xhat_np'] = _exist_attr(model_p, 'Xhat_np')[0]

        if record_loss:
            for _x in ['train_metrics', 'train_metrics_np']:
                _bool, _values = _exist_attr(model_p, _x)
                if _bool:
                    _dict[_x] = _values

    # Use consistent names
    n_cells, n_sites = X.shape
    obs_names = pd.Index([f"cell_{i}" for i in range(n_cells)])
    var_names = pd.Index([f"site_{i}" for i in range(n_sites)])
    obs_df = pd.DataFrame(obs, index=obs_names)
    var_df = pd.DataFrame(var, index=var_names)

    # Create base AnnData
    adata = AnnData(
        X=X,
        obs=obs_df,
        var=var_df,
        obsm=obsm,
        varm=varm,
        uns=uns,
        layers=layers
    )

    return adata


def save_model_to_pickle(
    model: MutModel,
    output_path: str
):
    """
    Save a MutModel instance to disk using pickle.

    Args:
        model: Trained MutModel instance to be saved.
        output_path: Path (str or Path) where the model will be saved.
    
    Note:
        This function uses pickle which may cause compatibility issues across different Python versions or environments.
        Users should ensure environment consistency or manually handle patches if loading fails.
        Consider using torch.save() for better cross-version support, or save only key components (e.g., N, P, Z).
        Storing minimal state (rather than full model) is recommended for long-term reproducibility — a maybe future TODO.

    Example:
        save_model_to_pickle(model_n, "model_n.pkl")
    """

    try:
        with open(output_path, 'wb') as f:
            pickle.dump(model, f, protocol=4)

    except Exception as e:
        warnings.warn(f"Failed to save model to {output_path}:\n{e}")
        raise


def extract_latent_mu(
    model: MutModel,
    index_map: Dict[int, tuple],
    X: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    train_np: bool = True,
    output_path: Optional[str] = None,
) -> Dict[int, torch.Tensor]:
    """
    Extract latent space mean (mu) from a MutModel for each group defined by index_map.

    Args:
        model: Trained MutModel instance.
        index_map: Mapping from group key (e.g., time point t) to (start_idx, end_idx).
        X: Optional input tensor; if None, uses `model.X`. [default: None]
        device: Device to run inference on; defaults to model.device. [default: None]
        dtype: Data type for tensor; defaults to model.dtype. [default: None]
        train_np: Whether to enable NP module during forward pass. [default: True]
        output_path: Optional path to save the result via torch.save. [default: None]

    Returns:
        Dictionary mapping group key to latent mu tensor (on CPU).

    Example:
        z_dict = extract_latent_mu(
            model=mut_model,
            index_map=index_map,
            output_path="z_real_dict.pt"
        )
    """
    z_real_dict = {}

    # Prepare input tensor
    X_input = _input_tensor(
        model.X if X is None else X,
        device=model.device if device is None else device,
        dtype=model.dtype if dtype is None else dtype
    )

    model.model.eval()
    with torch.no_grad():
        for t in sorted(index_map.keys()):
            start, end = index_map[t]
            X_t = X_input[start:end]

            outs = model.model(X_t, train_np=train_np)
            mu = outs[2]  # Latent mu from VAE output
            z_real_dict[t] = mu.cpu()

    # Save if requested
    if output_path is not None:
        try:
            torch.save(z_real_dict, output_path)
        except Exception as e:
            warnings.warn(f"Failed to save z_real_dict to {output_path}:\n{e}")

    return z_real_dict
