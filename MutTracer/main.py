import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict
from random import random
from itertools import combinations
import traceback
import random
from scipy.spatial.distance import euclidean
import math
import torch.nn.functional as F
from MutTracer import *
from tqdm import tqdm
import time
import copy
import argparse
from scMut import scMut 
import pickle
import time



"""
Main script for MutTracer: temporal latent state prediction and visualization.

Workflow:

1. **Argument Parsing**
    - Parses command-line arguments including paths to trained models, input/output times,
      number of training epochs, data files, and plotting options.
    - Arguments control which time points to use for training, prediction, and visualization.

2. **Device Setup**
    - Determines whether to use GPU or CPU for computations.

3. **Data Loading**
    - Loads pre-trained model, latent representations (`zmt_dict` and `zxt_dict`) 
      for real and transcriptional latent states.
    - Filters latent features according to user-specified input time points.
    - Moves latent features to the selected device and converts to float tensors.

4. **Feature Alignment**
    - Uses `MaskedDimensionAdapter` to align latent vectors across time points,
      handling missing dimensions or masked entries.
    - Prepares training (`zmt_train`, `zxt_train`) datasets.

5. **System Initialization**
    - Initializes `TrainingSystem` with the dimensions of zt and zxt.
    - Moves predictor model to the device.
    - Initializes benchmarking utility for baseline evaluation.

6. **Data Loader Creation**
    - Converts aligned latent features into PyTorch data loaders for batched training.

7. **Batch Statistics**
    - Optionally computes basic statistics (mean, std, min, max) of the first few batches
      for quick sanity check of the data.

8. **Training Loop**
    - Iterates over the specified number of epochs.
    - Performs a forward/backward step using `system.train_step`.
    - Computes epoch-level loss and optionally evaluates on test data every 10 epochs.
    - Tracks learning rate history, training and test losses, and saves best model based on test MSE.
    - Optionally triggers early stopping.

9. **Visualization**
    - `plot_full_timeline`: visualizes predicted and real latent states across all time points.
    - `plot_enhanced_loss`: plots smoothed training loss curves.
    - `visualize_predictions`: detailed comparison between real and predicted latent states.
    - `analyze_and_visualize`: identifies the most likely ancestor time point and visualizes lineage.
    - `analyze_n_distribution`: examines distribution of cellular generations.
    - `plot_distribution_comparison`: compares predicted vs real latent distributions, computes metrics.
    - `analyze_zxt`: evaluates transcriptional latent features, aligns predictions, and generates plots.

10. **Final Predictions and Metrics**
    - Generates final predicted latent states from zt and zxt components.
    - Computes evaluation metrics and saves results to disk.
    - Optionally updates ancestor time points if `--auto_ancestor` is specified.

11. **Saving Results**
    - Saves all training results, final predictions, metrics, and plots in a specified directory
      for downstream analysis.

This main script integrates model training, evaluation, and visualization into a single pipeline,
allowing systematic exploration of temporal latent dynamics in single-cell datasets.
"""



def parse_arguments():
    parser = argparse.ArgumentParser(description='MutTracer')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the trained model .pkl file')
    parser.add_argument('--zmt_path', type=str, required=True, help='Path to z_real dictionary file')
    parser.add_argument('--zxt_path', type=str, required=True, help='Path to zxt dictionary file')
    parser.add_argument('--input_times', type=int, nargs='+', required=True, help='Input time points')
    parser.add_argument('--predict_times', type=int, nargs='+', required=True, help='Time points to predict')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs (default: 500)')
    parser.add_argument('--real_times_keep', type=int, nargs='+', default=None,
                      help='Real time points to include in filtered UMAP')
    parser.add_argument('--pred_times_keep', type=int, nargs='+', default=None,
                      help='Predicted time points to include in filtered UMAP')
    parser.add_argument('--plot_format', type=str, default='pdf',
                      choices=['pdf', 'png', 'svg'], help='Output plot format')
    parser.add_argument('--auto_ancestor', action='store_true',
                      help='Automatically select most compact time as ancestor')
    parser.add_argument('--adata_path', type=str, required=True,
                    help='Path to the original .h5ad file')
    parser.add_argument('--scvi_model_path', type=str, required=True,
                    help='Path to the trained scVI model .pkl file')
    parser.add_argument(
        '--save_dir', type=str, default='results',
        help='Directory to save training results and plots (default: ./results)'
    )


    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time_mapping = {
        1: 1,   
        2: 2, 
        3: 3,   
    }
    results = TrainingResults()
    input_time_points = args.input_times  
    predict_time_points = args.predict_times  
    
    
    model, original_z_real_dict, original_zxt_dict = load_model_and_data(
        args.model_path,
        args.zmt_path,
        args.zxt_path
    )
    original_z_real_dict = torch.load(args.zmt_path)  
    original_zxt_dict = torch.load(args.zxt_path)  

    original_z_real_dict = dict(sorted(original_z_real_dict.items()))
    original_z_real_dict = dict(sorted(original_z_real_dict.items()))
    
    z_real_dict = {time_mapping[k]: v for k, v in original_z_real_dict.items() if time_mapping[k] in input_time_points}
    zxt_dict = {time_mapping[k]: v for k, v in original_zxt_dict.items() if time_mapping[k] in input_time_points}
    z_real_dict = {k: v.float().to(device) for k, v in z_real_dict.items()}
    zxt_dict = {k: v.float().to(device) for k, v in zxt_dict.items()}
    adapter = MaskedDimensionAdapter()
    aligned_z_real_dict, masks = adapter.align_input(z_real_dict) 
    aligned_zxt_dict, masks = adapter.align_input(zxt_dict) 
    zt_train, zxt_train = aligned_z_real_dict, aligned_zxt_dict
    zt_test, zxt_test = aligned_z_real_dict, aligned_zxt_dict

    print(f"\nTime: {sorted(zt_train.keys())}")
    
    zt_dim = aligned_z_real_dict[next(iter(aligned_z_real_dict))].shape[-1] 
    zxt_dim = aligned_zxt_dict[next(iter(aligned_zxt_dict))].shape[-1]
    system = TrainingSystem(zt_dim, zxt_dim, hidden_dim=128)
    
    system.predictor = system.predictor.to(device).float()  
    best_mse = float('inf')

    benchmark = Benchmark(device, zt_dim, zxt_dim)
    
    baseline_results = benchmark.evaluate(aligned_z_real_dict, aligned_zxt_dict)
    plot_benchmark1(baseline_results)
    train_loader = create_data_loader(zt_train, zxt_train, batch_size=32, seq_length=3)
    test_loader = create_data_loader(zt_test, zxt_test, batch_size=32, seq_length=3)
    

    batch_stats = []
    for i, batch in enumerate(train_loader):
        zt_batch = batch['z_real'][list(batch['z_real'].keys())[0]]  
        batch_stats.append({
            'batch': i,
            'mean': zt_batch.mean().item(),
            'std': zt_batch.std().item(),
            'max': zt_batch.max().item(),
            'min': zt_batch.min().item()
        })
        if i >= 10:  
           break
    
    zx_train_means = {t: torch.mean(zxt_train[t]) for t in zt_train.keys()}
    zx_train_stds = {t: torch.std(zxt_train[t]) + 1e-6 for t in zt_train.keys()}
    z_train_means = {t: torch.mean(zt_train[t]) for t in zt_train.keys()}
    z_train_stds = {t: torch.std(zt_train[t]) + 1e-6 for t in zt_train.keys()}
    epochs = args.epochs 
    best_mse = float('inf')
    with tqdm(total=epochs, desc="Training Progress", unit="epoch") as pbar:
        for epoch in range(epochs):
            loss = system.train_step(zt_train, zxt_train)
            system.predictor.train()  
            epoch_loss = 0.0
            batch_count = 0
            start_time = time.time()
        
            for batch in train_loader:
                loss = system.train_step(batch['z_real'], batch['zxt'], masks_dict=batch['masks'] )
                epoch_loss += loss
                batch_count += 1
        
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
            if not hasattr(system, 'lr_history'):
                system.lr_history = []
            system.lr_history.append(system.optimizer.param_groups[0]['lr'])
            results.epoch_losses.append(avg_epoch_loss)
            if epoch % 10 == 0:
                test_loss = 0.0
                test_batches = 0
                epoch_preds = {}
                pred_results = {}
                with torch.no_grad():
                    for test_batch in test_loader:

                        pred_dict, _ = system.predict_sequence(test_batch['z_real'], test_batch['zxt'])
                        batch_loss = 0
                        valid_count = 0
                        for t in pred_dict:
                        
                            if t in test_batch['z_real']:
                                pred = pred_dict[t]['prediction'][:, :zt_dim]
                                true = test_batch['z_real'][t]
                                batch_loss += F.mse_loss(pred, true).item()
                                pred_results[t] = {
                                    'pred': pred_dict[t]['prediction'],
                                    'true': z_real_dict[t].to(device)  
                                }
                                epoch_preds[t] = {
                                    'zt_pred': pred_dict[t]['prediction'][:, :zt_dim].cpu().numpy(),
                                    'zxt_pred': pred_dict[t]['prediction'][:, zt_dim:].cpu().numpy(),
                                    'zt_true': test_batch['z_real'][t].cpu().numpy(),
                                    'zxt_true': test_batch['zxt'][t].cpu().numpy()
                                }
                                valid_count += 1
                        if valid_count > 0:
                            test_loss += batch_loss / valid_count
                        test_batches += 1
                results.predictions[epoch] = epoch_preds     
                avg_test_loss = test_loss / test_batches if test_batches > 0 else 0
                results.test_losses.append(avg_test_loss)
                if avg_test_loss < best_mse:
                    best_mse = avg_test_loss
                    torch.save(system.predictor.state_dict(), 'best_model.pth')
                    results.best_model = copy.deepcopy(system.predictor.state_dict())
                    results.best_epoch = epoch
                    results.best_prediction = epoch_preds
                    results.best_metrics = best_mse
                with torch.no_grad():
                    system.predictor.eval()
                    pred_dict, _ = system.predict_sequence(zt_test, zxt_test)
                    pred_list = []
                    true_list = []
                    common_times = sorted(set(pred_dict.keys()) & set(zt_test.keys()))
                    for t in common_times:
                        if isinstance(pred_dict[t], dict):
                            pred = pred_dict[t]['prediction']
                        else:
                            pred = pred_dict[t]
                    
                        if pred.shape[-1] > zt_dim:
                            pred = pred[..., :zt_dim]
                    
                        pred_list.append(pred)
                        true_list.append(zt_test[t])
               
                    pred_tensor = torch.stack(pred_list)
                    true_tensor = torch.stack(true_list)
                    
                
                    your_mse = F.mse_loss(pred_tensor, true_tensor).item()
                    if your_mse < best_mse:
                        best_mse = your_mse
                    metrics = system.evaluate_predictions(zt_test, zxt_test)

                if system.early_stopper(loss):
                    break

            pbar.update(1)
                
    plot_full_timeline(system, aligned_z_real_dict, aligned_zxt_dict, device, args)

    plot_enhanced_loss(system, args)
    final_pred, _ = system.predict_sequence(zt_test, zxt_test)
    final_metrics = system.evaluate_predictions(zt_test, zxt_test)
    results.final_prediction = final_pred  
    results.final_metrics = final_metrics  

    zt_dim = aligned_z_real_dict[next(iter(aligned_z_real_dict))].shape[-1]
    zxt_dim = aligned_zxt_dict[next(iter(aligned_zxt_dict))].shape[-1]

    visualize_predictions(
        original_z_real_dict=original_z_real_dict,
        original_zxt_dict=original_zxt_dict,
        pred_dict=final_pred,  
        zt_dim=zt_dim,
        zxt_dim=zxt_dim,
        real_times_keep=args.real_times_keep,
        pred_times_keep=args.pred_times_keep,
        args=args  
    )
    
    
    ancestor_time=analyze_and_visualize(
        original_z_real_dict=original_z_real_dict,
        original_zxt_dict=original_zxt_dict,
        pred_dict=final_pred,
        zt_dim=aligned_z_real_dict[next(iter(aligned_z_real_dict))].shape[-1],
        zxt_dim=aligned_zxt_dict[next(iter(aligned_zxt_dict))].shape[-1],
        args=args,
        real_times_keep=args.real_times_keep,
        pred_times_keep=args.pred_times_keep
    )
    if args.auto_ancestor and ancestor_time is not None:
        if args.pred_times_keep is None:
            args.pred_times_keep = []
        if ancestor_time not in args.pred_times_keep:
            args.pred_times_keep.append(ancestor_time)

    zt_dim = aligned_z_real_dict[next(iter(aligned_z_real_dict))].shape[-1]
    zxt_dim = aligned_zxt_dict[next(iter(aligned_zxt_dict))].shape[-1]
    pred_from_zt = {
        t: data['prediction'][:, :zt_dim].detach().cpu().numpy()
        for t, data in final_pred.items()
    }
    pred_from_zxt = {
        t: data['prediction'][:, zt_dim:zt_dim + zxt_dim].detach().cpu().numpy()
        for t, data in final_pred.items()
    }

    analyze_n_distribution(
        model=model, 
        original_z_real_dict=original_z_real_dict,
        split_pred_dict=pred_from_zt,
        real_times_keep=args.real_times_keep,
        pred_times_keep=args.pred_times_keep,
        args=args
    )  

    for predict_time in args.predict_times:  
        try:
            r2, wd = plot_distribution_comparison(
                original_z_real_dict,
                pred_from_zt,  
                model.model.decoder_n,
                predict_time=predict_time,
                args=args
            )

            if not hasattr(results, 'comparison_metrics'):
                results.comparison_metrics = {}
            results.comparison_metrics[predict_time] = {'R2': r2, 'WD': wd}
        
        except ValueError as e:
            print(f"{predict_time}: {str(e)}") 
    aligned_preds = align_zxt_features(
        pred_dict=pred_from_zxt, 
        ref_dict=original_zxt_dict,
        method='quantile'
    )
    original_adata = sc.read_h5ad(args.adata_path)

    with open(args.scvi_model_path, "rb") as f: 
        model_exp = pickle.load(f)
    analyze_zxt(
        model=model_exp, 
        adata=original_adata,      
        original_zxt_dict=original_zxt_dict,  
        split_preds=aligned_preds, 
        real_times_keep=args.real_times_keep,  
        pred_times_keep=args.pred_times_keep,   
        output_dir="training_plots",
        plot_format="pdf",
        ancestor_time=ancestor_time
    )
    

   
    plot_training_loss(system, args, filename="full_loss")


    save_results(results, args, epochs, input_time_points, predict_time_points, filename='training_results.pkl')

