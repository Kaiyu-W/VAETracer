import numpy as np
import pandas as pd
from scipy.sparse import spmatrix, csr_matrix
from anndata import AnnData
from torch import Tensor
from .scMut import MutModel
from .typing import Optional


def _asarray(values):
    if isinstance(values, (list, tuple, np.ndarray)):
        return np.asarray(values)
    elif isinstance(values, Tensor):
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

