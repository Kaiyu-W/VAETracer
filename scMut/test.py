import os
import numpy as np
import pickle
import tempfile
from datetime import datetime
from collections import defaultdict
from time import time, process_time
from scipy.stats import pearsonr, spearmanr, linregress
import matplotlib.pyplot as plt

import torch
from .scMut import MutModel
from .data import (
    simulate_data,
    simulate_lineage_data
)
from .utils import (
    DEVICE,
    set_seed,
    visualize_loss, 
    visualize_loss_k,
    plot_metrics,
    plot_latent_space,
    plot_regplot
)
from .log import (
    logger, 
    add_file_handler, 
    remove_file_handler
)


def setup_output_dir(save_dir=None, verbose=True, prefix=None):
    if prefix is None:
        prefix = 'temp_output'
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"./{prefix}_{timestamp}"
    
    save_dir = os.path.abspath(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    if verbose:
        logger.info(f"Set save_dir as: {save_dir}")

    return save_dir


def config_params(model_params, load_params, train_params, run_model_method, dtype):
    if dtype == 'float16':
        torch_dtype = torch.float16
    elif dtype == 'float32':
        torch_dtype = torch.float32
    elif dtype == 'float64':
        torch_dtype = torch.float64

    # model
    _model_params = model_params.copy()
    _model_params['dtype'] = torch_dtype

    # vae load 
    _load_params = load_params.copy()
    _load_params['dtype'] = torch_dtype
    if _load_params['num_workers'] > 1:
        os.environ['NUMEXPR_MAX_THREADS'] = str(_load_params['num_workers'])
    if _load_params['X'].shape[0] % _load_params['batch_size'] == 1 and not _model_params['train_transpose']:
        _load_params['batch_size'] += 1
    elif _load_params['X'].shape[1] % _load_params['batch_size'] == 1 and _model_params['train_transpose']:
        _load_params['batch_size'] += 1

    # train
    _train_nmf_params = dict(
        X=train_params['X'],
        lr=train_params['lr'],
        patience=train_params['patience'],
        min_delta=train_params['min_delta'],
        unlimited_epoch=train_params['unlimited_epoch'],
        logging_interval=train_params['logging_interval'],
        use_tqdm=train_params['use_tqdm'], 
        verbose=train_params['verbose'],
    )
    _train_vae_params = dict(
        use_tqdm=train_params['use_tqdm'], 
        patience=train_params['patience'],
        min_delta=train_params['min_delta'],
        verbose=train_params['verbose'],
    )
    for _x in ['n_init', 'p_init']:
        if _x in train_params:
            _train_nmf_params[_x] = _train_vae_params[_x] = train_params[_x]
    _train_ft_n_params = dict(
        max_n=train_params['max_n'], 
        use_nmf='nmf+ft' in run_model_method or 'vae' not in run_model_method,
        verbose=train_params['verbose'],
        **train_params['optim_kwargs']
    )
    _train_ft_p_params = dict( 
        use_n='finetune',
        lr=train_params['lr'],
        patience=train_params['patience'],
        min_delta=train_params['min_delta'],
        unlimited_epoch=train_params['unlimited_epoch'],
        logging_interval=train_params['logging_interval'],
        use_tqdm=train_params['use_tqdm'], 
        verbose=train_params['verbose'],
    )
    
    return (
        _model_params,
        _load_params,
        _train_nmf_params,
        _train_vae_params,
        _train_ft_n_params,
        _train_ft_p_params,
    )


def compare_data(data_dict):
    result = {}
    keys = list(data_dict.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            key1, key2 = keys[i], keys[j]
            data1, data2 = data_dict[key1], data_dict[key2]

            # Pearson correlation
            pearson_corr, pearson_p = pearsonr(data1, data2)

            # Spearman correlation
            spearman_corr, spearman_p = spearmanr(data1, data2)

            # Linear regression
            beta, _, r_value, linreg_p, _ = linregress(data1, data2)

            key = f'{key1}_{key2}'
            result[key] = dict(
                pearson_corr=pearson_corr, pearson_p=pearson_p,
                spearman_corr=spearman_corr, spearman_p=spearman_p,
                linreg_beta=beta, linreg_r=r_value, linreg_p=linreg_p,
            )
    return result


def plot_data(data_dict, save_dir=None, prefix=None):
    if save_dir is not None:
        if not os.path.exists(save_dir):
            save_dir = None
    prefix = '' if prefix is None else f'{prefix}_'

    keys = list(data_dict.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            key1, key2 = keys[i], keys[j]
            key = f'{key1}_{key2}'
            data1, data2 = data_dict[key1], data_dict[key2]

            plot_regplot(
                data_dict, key1, key2, 
                lowess=False, 
                show=False, 
                save=None if save_dir is None else f'{save_dir}/regplot_{prefix}{key}.png'
            )


def run_once(
    X,
    run_model_method,
    max_n = None,
    model_params = dict(),
    load_params = dict(),
    train_params = dict(),
    train_transpose = False,
    dtype: str = 'float32',
    seed = 42,
    save_dir = None,
    verbose = True,
    real_n = None,
    real_p = None,
    vae_type = 'mode1'
):
    assert dtype in ['float16', 'float32', 'float64']
    assert run_model_method in ['nmf','vae','nmf+vae','nmf+ft','vae+ft','nmf+vae+ft','nmf+ft+vae']
    assert 'train_transpose' not in model_params, 'Set train_transpose directly rather than in model_params!'
    assert vae_type in ['mode1', 'mode2']
    save = save_dir is not None
    if save:
        assert os.path.exists(save_dir)

    # parameters config
    _model_params = dict(
        input_dim=X.shape[0 if train_transpose else 1],
        hidden_dims=[128],
        z_dim=20,
        num_epochs=1000,
        num_epochs_nmf=10000,
        lr=5e-3,
        beta_kl=0.001,
        beta_best=0.01, # unuse
        device=DEVICE, 
        dtype=None,
        seed=seed,
        eps=1e-10,
        model_type = None, # force
        activation='relu',
        use_batch_norm='encoder',
        use_layer_norm='both',
        miss_value=3,
        wt_value=0,
        edit_value=1,
        edit_loss_weight = 3, 
        wt_loss_weight = 1, 
        use_weight=False,
        train_transpose=train_transpose
    )
    _train_params = dict(
        X=X,
        n_init=None,
        p_init=None,
        lr=None,
        patience=45,
        min_delta=None,
        use_tqdm=True,
        unlimited_epoch=False, 
        logging_interval=100, 
        verbose=verbose,
        max_n=max_n,
        optim_kwargs=dict()
    )
    _load_params = dict(
        X=X, 
        batch_size=1000, 
        num_workers=15, 
        pin_memory=True, 
        dtype=None, 
        shuffle=True, 
        best_n_or_p=None
    )

    _model_params.update(model_params)
    _train_params.update(train_params)
    _load_params.update(load_params)
    (
        _model_params,
        _load_params,
        _train_nmf_params,
        _train_vae_params,
        _train_ft_n_params,
        _train_ft_p_params,
    ) = config_params(
        model_params=_model_params, 
        load_params=_load_params, 
        train_params=_train_params,
        run_model_method=run_model_method,
        dtype=dtype
    )

    # Initialize and train model
    model = MutModel(**_model_params)
    model._nmf = False
    model._vae = False
    model._ft = False
    loss_dict = defaultdict()

    _time_start = process_time()
    n_dict, p_dict, xz_dict = {'real': real_n}, {'real': real_p}, {'x': X}

    def _nmf():
        loss_dict['nmf'] = model.train_nmf(**_train_nmf_params)
        model._nmf = True

        if save:
            visualize_loss(loss_dict['nmf'], yscale=None, save=f'{save_dir}/nmf_loss.png', show=False)
            visualize_loss(loss_dict['nmf'], yscale='log', save=f'{save_dir}/nmf_loss_log.png', show=False)
            plt.close('all')

    def _vae(type='mode1'):
        model.load_data(**_load_params)
        if type=='mode1':
            if verbose:
                logger.info(f'Now run vae-{type}-np...')
            model.set_mode('np', type='mode1')
            loss_dict['vae_np'] = model.train_np(**_train_vae_params)

            if verbose:
                logger.info(f'Now run vae-{type}-xhat...')
            model.set_mode('xhat', type='mode1')
            loss_dict['vae_xhat'] = model.train(**_train_vae_params)

            model._vae = True
        else:
            if verbose:
                logger.info(f'Now run vae-{type}-xhat...')
            model.set_mode('xhat', type='mode2')
            loss_dict['vae_xhat'] = model.train(**_train_vae_params)

            if verbose:
                logger.info(f'Now run vae-{type}-np...')
            model.set_mode('np', type='mode2')
            loss_dict['vae_np'] = model.train_np(**_train_vae_params)

            model._vae = True

        if save:
            visualize_loss(loss_dict['vae_np'], yscale=None, save=f'{save_dir}/vae_np_loss.png', show=False)
            visualize_loss(loss_dict['vae_np'], yscale='log', save=f'{save_dir}/vae_np_loss_log.png', show=False)
            visualize_loss(loss_dict['vae_xhat'], yscale=None, save=f'{save_dir}/vae_xhat_loss.png', show=False)
            visualize_loss(loss_dict['vae_xhat'], yscale='log', save=f'{save_dir}/vae_xhat_loss_log.png', show=False)

            model.train_metrics_xhat, model.valid_metrics_xhat = model.train_metrics, model.valid_metrics
            model.train_metrics, model.valid_metrics = model.train_metrics_np, model.valid_metrics_np
            plot_metrics(model, yscale=None, nrow=2, save=f'{save_dir}/vae_np_losses.png', show=False)
            plot_metrics(model, yscale='log', nrow=2, save=f'{save_dir}/vae_np_losses_log.png', show=False)
            model.train_metrics, model.valid_metrics = model.train_metrics_xhat, model.valid_metrics_xhat

            logger.info('Now represent Z by tSNE & UMAP...')
            z_tsne = plot_latent_space(model, labels=None if train_transpose else real_n, reduction='tsne', return_z=True, save=f'{save_dir}/vae_z_tsne.png', show=False)
            z_umap = plot_latent_space(model, labels=None if train_transpose else real_n, reduction='umap', return_z=True, save=f'{save_dir}/vae_z_umap.png', show=False)
            xz_dict['vae_z_tsne'] = z_tsne
            xz_dict['vae_z_umap'] = z_umap
            plt.close('all')

    def _ft():
        loss_dict['ft_n'] = model.finetune_n(**_train_ft_n_params)
        loss_dict['ft_p'] = model.finetune_p(**_train_ft_p_params)
        model._ft = True

        if save:
            visualize_loss_k(loss_dict['ft_n'], k_optimal=model.k_optimal, smooth=False, save=f'{save_dir}/finetune_n_loss.png', show=False)
            visualize_loss_k(loss_dict['ft_n'], k_optimal=model.k_optimal, smooth=True, save=f'{save_dir}/finetune_n_loss_smooth.png', show=False)
            visualize_loss(loss_dict['ft_p'], yscale=None, save=f'{save_dir}/finetune_p_loss.png', show=False)
            visualize_loss(loss_dict['ft_p'], yscale='log', save=f'{save_dir}/finetune_p_loss_log.png', show=False)
            plt.close('all')

    # nmf
    if run_model_method in ['nmf','nmf+vae','nmf+ft','nmf+vae+ft','nmf+ft+vae']: 
        if verbose:
            logger.info('Now run nmf...')

        _nmf()
        
    # vae
    if run_model_method in ['vae','vae+ft','nmf+vae','nmf+vae+ft']: 
        if verbose:
            logger.info('Now run vae...')

        if 'nmf+vae' in run_model_method:
            if train_transpose:
                model.update_n(model.N_nmf, requires_grad=False)
                _load_params['best_n_or_p'] = model.P_nmf
                _train_vae_params['p_init'] = None # avoid p_init updated again when train_np()
            else:
                model.update_p(model.P_nmf, requires_grad=False)
                _load_params['best_n_or_p'] = model.N_nmf
                _train_vae_params['n_init'] = None # avoid n_init updated again when train_np()

        _vae(vae_type)
        

    # finetune (nmf)
    if run_model_method in ['nmf+ft','vae+ft','nmf+vae+ft','nmf+ft+vae']:
        if verbose:
            logger.info('Now run ft...')

        _ft()

        # ft + vae
        if 'ft+vae' in run_model_method:
            if verbose:
                logger.info('Now run vae...')

            if train_transpose:
                model.update_n(model.N_ft, requires_grad=False)
                # _load_params['best_n_or_p'] = model.P_ft
                _train_vae_params['p_init'] = None # avoid p_init updated again when train_np()
            else:
                model.update_p(model.P_ft, requires_grad=False)
                # _load_params['best_n_or_p'] = model.N_ft
                _train_vae_params['n_init'] = None # avoid n_init updated again when train_np()

            _vae(vae_type)

    # time
    _time_end = process_time()
    _seconds = _time_end - _time_start
    if verbose:
        _m, _s = divmod(_seconds, 60)
        _h, _m = divmod(_m, 60)
        _format = "%1dh%2dm%2ds" % (_h, _m, _s)
        logger.info(f"Done by {_format}")

    # data
    if model._nmf:
        n_dict['nmf'] = model.N_nmf
        p_dict['nmf'] = model.P_nmf
    if model._vae:
        n_dict['vae'] = model.N
        p_dict['vae'] = model.P
        xz_dict['vae_xhat'] = model.Xhat
        xz_dict['vae_xhat_np'] = model.Xhat_np
        xz_dict['vae_z'] = model.Z
    if model._ft:
        n_dict['ft'] = model.N_ft
        p_dict['ft'] = model.P_ft

    # stat & visualize
    stat_n = compare_data(n_dict)
    stat_p = compare_data(p_dict)
    stat_dict = dict(n=stat_n, p=stat_p)

    if save:
        plot_data(n_dict, save_dir=save_dir, prefix='n')
        plot_data(p_dict, save_dir=save_dir, prefix='p')
        plt.close('all')

    return model, loss_dict, _seconds, n_dict, p_dict, xz_dict, stat_dict


def run_pipe(
    run_model_method, n_repeat=100, save_dir=None, verbose=True, vae_type='mode1', beta_pairs=None, # this line could change
    n_cells = 1000, n_sites = 2000, max_n = 50, ratio2 = 0.5, survival_rate = 0.6, noise_level = 0.1, # this line should fix
    **pipe_kwargs # changes about how to run model 
):
    assert n_repeat > 0
    if 'train_transpose' in pipe_kwargs and pipe_kwargs['train_transpose']:
        prefix=f'trans_{run_model_method}'
    else:
        prefix=run_model_method
    root_save_dir = setup_output_dir(save_dir, verbose=verbose, prefix=prefix)
    log_file_handler = add_file_handler(logger, f'{root_save_dir}/run.log')
    logger.info(f"Logging configured. Logs will be saved to {root_save_dir}/run.log")

    # Generate data:
    if beta_pairs is None:
        beta_pairs = [
            (0.1, 0.5, None, None),  # two peak of 0/1
            (1  , 32 , None, None),  # ~0
            (8  , 4  , None, None),  # ~0.67
            (2  , 38 , 16  , 4   ),  # two peak of 0.05/0.8
        ]
    seeds = range(1, 1 + n_repeat)
    params_dict = dict(
        n_cells=n_cells, 
        n_sites=n_sites, 
        max_n=max_n,
        beta_pairs=beta_pairs, 
        seeds=seeds,
        survival_rate=survival_rate,
        noise_level=noise_level,
    )
    final_result = dict(params=params_dict, simple={}, lineage={})

    def _run_simple(beta_pair, seed, run_model_method, save_dir, **kwargs):
        # Simulate data
        silicon_datas = simulate_data(
            n_cells, n_sites, *beta_pair,
            ratio2=ratio2,
            generation_min=0,
            generation_max=max_n,
            noise_level=noise_level,
            seed=seed,
        )
        M_p, M_n, M, M_mask = silicon_datas
        M_sample = (np.random.rand(*M.shape) < M).astype(int)
        X = np.where(M_mask, np.ones_like(M_sample) * 3, M_sample)

        # Run the model and collect results
        model, loss_dict, run_seconds, n_dict, p_dict, xz_dict, stat_dict = run_once(
            X=X,
            run_model_method=run_model_method,
            max_n=max_n,
            seed=seed,
            verbose=verbose,
            save_dir=save_dir,
            real_n=M_n,
            real_p=M_p,
            vae_type=vae_type,
            **kwargs
        )

        # clear model for space
        model.to_cpu()
        del model

        # Return the combined results
        return {
            'data': [M_sample, M_mask],      # Simulated data
            'loss_dict': loss_dict,          # Training losses
            'run_seconds': run_seconds,      # Execution cpu time
            'n_dict': n_dict,                # Estimated n values
            'p_dict': p_dict,                # Estimated p values
            'xz_dict': xz_dict,              # Placeholder for future use
            'stat_dict': stat_dict           # Statistical comparisons
        }

    def _run_lineage(beta_pair, seed, run_model_method, save_dir, **kwargs):
        # Simulate lineage data
        silicon_datas = simulate_lineage_data(
            n_cells, n_sites, *beta_pair,
            ratio2=ratio2,
            generation_max=max_n,
            survival_rate=survival_rate,
            noise_level=noise_level,
            seed=seed,
            keep_whole_lineage=True
        )
        M_p, M_n, M_sample, M_mask, M_newick, M_id = silicon_datas
        X = np.where(M_mask, np.ones_like(M_sample) * 3, M_sample)

        # Run the model and collect results
        model, loss_dict, run_seconds, n_dict, p_dict, xz_dict, stat_dict = run_once(
            X=X,
            run_model_method=run_model_method,
            max_n=max_n,
            seed=seed,
            verbose=verbose,
            save_dir=save_dir,
            real_n=M_n,
            real_p=M_p,
            vae_type=vae_type,
            **kwargs
        )

        # clear model for space
        model.to_cpu()
        del model

        # Return the combined results
        return {
            'data': [M_sample, M_mask, M_newick, M_id],      # Simulated data
            'loss_dict': loss_dict,                          # Training losses
            'run_seconds': run_seconds,                      # Execution cpu time
            'n_dict': n_dict,                                # Estimated n values
            'p_dict': p_dict,                                # Estimated p values
            'xz_dict': xz_dict,                              # Placeholder for future use
            'stat_dict': stat_dict,                          # Statistical comparisons
        }

    try:
        # Run simple data experiments
        if verbose:
            logger.info("Now run simple data...")
        simple_results = final_result['simple']
        for i, beta_pair in enumerate(beta_pairs):
            for seed in seeds:
                if verbose:
                    logger.info(f"Now run simple data, {i}:{beta_pair} seed={seed}")

                _save_dir = setup_output_dir(f'{root_save_dir}/simple/{i}_{seed}', verbose=verbose)
                simple_results.setdefault(i, {})[seed] = _run_simple(beta_pair, seed, run_model_method, _save_dir, **pipe_kwargs)

                if verbose:
                    logger.info(f"Statistics: {simple_results[i][seed]['stat_dict']}")

        # Run lineage data experiments
        if verbose:
            logger.info("Now run lineage data...")
        lineage_results = final_result['lineage']
        for i, beta_pair in enumerate(beta_pairs):
            for seed in seeds:
                if verbose:
                    logger.info(f"Now run lineage data, {i}:{beta_pair} seed={seed}")

                _save_dir = setup_output_dir(f'{root_save_dir}/lineage/{i}_{seed}', verbose=verbose)
                lineage_results.setdefault(i, {})[seed] = _run_lineage(beta_pair, seed, run_model_method, _save_dir, **pipe_kwargs)

                if verbose:
                    logger.info(f"Statistics: {lineage_results[i][seed]['stat_dict']}")

        # Get final results and save
        final_result = dict(params=params_dict, simple=simple_results, lineage=lineage_results)
        if verbose:
            logger.info(f"Now save all results into {root_save_dir}/result.pkl...")
        with open(f'{root_save_dir}/result.pkl', 'wb') as f:
            pickle.dump(final_result, f)

        return final_result

    except Exception as e:
        logger.error(f"Error occurred: {e}", exc_info=True)
        return final_result

    finally:
        # Clean up the handlers
        remove_file_handler(logger, log_file_handler)


