import os
import pickle
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for constructing fixed-length temporal sequences from
    latent representations across multiple time points.

    This dataset aligns two types of latent embeddings (z_real and zxt)
    based on shared time indices, and returns sliding-window sequences
    suitable for training time-series models (e.g., RNNs, Transformers).

    Each sample contains:
        - A dictionary of z_real embeddings indexed by time
        - A dictionary of zxt embeddings indexed by time
        - The corresponding list of time points

    Parameters
    ----------
    z_real_dict : dict
        Dictionary mapping time points to real latent representations
        (shape: [batch_size, latent_dim]).
    zxt_dict : dict
        Dictionary mapping time points to transcriptional latent representations
        (shape: [batch_size, latent_dim]).
    seq_length : int, optional
        Length of the temporal sequence (number of consecutive time points),
        by default 5.
    """

    def __init__(self, z_real_dict, zxt_dict, seq_length=5):
        self.seq_length = seq_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        common_times = sorted(set(z_real_dict.keys()) & set(zxt_dict.keys()))
        if not common_times:
            raise ValueError("not common_times")
        self.times = common_times
        self.z_real = [z.to(self.device) for z in [z_real_dict[t] for t in common_times]]
        self.zxt = [z.to(self.device) for z in [zxt_dict[t] for t in common_times]]
        batch_size = self.z_real[0].shape[0]
        for z in self.z_real + self.zxt:
            assert z.shape[0] == batch_size,"all time must be same size" 

    def __len__(self):
        return len(self.times) - self.seq_length + 1
    
    def __getitem__(self, idx):
        end_idx = idx + self.seq_length
        times = self.times[idx:end_idx]
        z_real_seq = torch.stack(self.z_real[idx:end_idx])
        zxt_seq = torch.stack(self.zxt[idx:end_idx])
        sample = {
            'z_real': {t: z for t, z in zip(times, z_real_seq)},
            'zxt': {t: z for t, z in zip(times, zxt_seq)},
            'times': times
        }
        
        return sample


def collate_fn(batch):
    """
    Custom collate function for batching variable-length temporal sequences.

    This function aligns samples in a batch by the union of all time points,
    pads missing time points with zero-valued dummy tensors, and generates
    corresponding binary masks to indicate valid observations. The output
    is organized as time-indexed dictionaries for latent representations
    and masks, facilitating time-aware downstream modeling.
    """
    all_times = sorted(set(t for sample in batch for t in sample['times']))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    batch_z_real = defaultdict(list)
    batch_zxt = defaultdict(list)
    batch_masks = defaultdict(list)  
    for t in all_times:
        for sample in batch:
            if t in sample['z_real']:
                batch_z_real[t].append(sample['z_real'][t])
                batch_zxt[t].append(sample['zxt'][t])
                mask = torch.ones(sample['z_real'][t].shape[0], 1, device=device)
                batch_masks[t].append(mask)
            else:
                for ref_sample in batch:
                    if t in ref_sample['z_real']:
                        dummy_shape = ref_sample['z_real'][t].shape
                        dummy = torch.zeros(dummy_shape, device=device)
                        break
                else:
                    raise ValueError(f"can not generate dummy for {t}")  
                batch_z_real[t].append(dummy)
                batch_zxt[t].append(dummy)
                batch_masks[t].append(torch.zeros(dummy.shape[0], 1, device=device))
    return {
        'z_real': {t: torch.cat(batch_z_real[t]) for t in all_times},
        'zxt': {t: torch.cat(batch_zxt[t]) for t in all_times},
        'masks': {t: torch.cat(batch_masks[t]) for t in all_times}  
    }


def create_data_loader(z_real_dict, zxt_dict, batch_size=32, seq_length=3):
    """
    Utility function to construct a DataLoader for time-series latent data.

    This function initializes a TimeSeriesDataset from aligned latent
    representations and wraps it in a PyTorch DataLoader using a custom
    collate function to handle variable temporal coverage across samples.
    """
    dataset = TimeSeriesDataset(z_real_dict, zxt_dict, seq_length)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=False  
    )
    return loader


class DimensionAdapter:
    """
    Utility class for aligning and recovering tensor dimensions across time points.

    This class provides functionality to normalize the number of cells per
    time point by either padding with zeros or truncating to a fixed reference
    size, and to subsequently recover predictions back to their original
    cell counts. It is mainly used to ensure dimensional consistency during
    model inference while preserving original data shapes.
    """
    def __init__(self, ref_num_cells=1000):
        self.ref_num = ref_num_cells  

    def align_input(self, raw_dict):
        aligned = {}
        for day, data in raw_dict.items():
            n_cells = data.shape[0]
            if n_cells < self.ref_num:
                padding = torch.zeros(self.ref_num - n_cells, data.shape[1],
                                    dtype=data.dtype, device=data.device)
                aligned[day] = torch.cat([data, padding], dim=0)
            else:
                aligned[day] = data[:self.ref_num]
        return aligned
    
    def recover_output(self, pred_dict, original_dict):
        recovered = {}
        for day, data in pred_dict.items():
            if day in original_dict:
                original_num = original_dict[day].shape[0]
                recovered[day] = data[:original_num]
        return recovered


class MaskedDimensionAdapter_most(DimensionAdapter):
    """
    Dimension adapter that aligns inputs by padding to the maximum cell count
    across time points while generating corresponding validity masks.

    For each time point, tensors with fewer cells are zero-padded to match
    the maximum observed cell number, and binary masks are created to indicate
    real versus padded entries. This strategy preserves all available cells
    and is suitable when retaining maximal information is preferred.
    """
    def align_input(self, raw_dict):
        max_cells = max(x.shape[0] for x in raw_dict.values())
        aligned = {}
        masks = {} 
        
        for t, data in raw_dict.items():

            if data.shape[0] < max_cells:
                aligned[t] = F.pad(data, (0,0,0,max_cells-data.shape[0]))
                masks[t] = torch.cat([torch.ones(data.shape[0]), 
                                    torch.zeros(max_cells-data.shape[0])])
            else:
                aligned[t] = data[:max_cells]
                masks[t] = torch.ones(max_cells)
                
        return aligned, masks  


class MaskedDimensionAdapter:
    """
    Dimension adapter that aligns inputs by subsampling to the minimum cell count
    across time points and generating uniform validity masks.

    For time points with more cells than the minimum, random subsampling is
    applied to enforce dimensional consistency. This strategy avoids padding
    and is useful when strict alignment without artificial values is required.
    """
    def align_input(self, raw_dict):

        min_cells = min(x.shape[0] for x in raw_dict.values())
        aligned = {}
        masks = {}
        
        for t, data in raw_dict.items():

            if data.shape[0] > min_cells:

                indices = torch.randperm(data.shape[0])[:min_cells]
                aligned[t] = data[indices]
            else:
                aligned[t] = data
            
            masks[t] = torch.ones(min_cells)
                
        return aligned, masks


class TrainingResults:
    """
    Container class for tracking training and evaluation outcomes.

    This class stores loss trajectories, performance metrics, model checkpoints,
    predictions at different epochs, and timing information. It provides a
    centralized structure for organizing results during model training,
    validation, and final evaluation.
    """
    def __init__(self):
        self.epoch_losses = []
        self.test_losses = []
        self.predictions = {}  # {epoch: {time: (pred, true)}}
        self.best_model = None
        self.best_epoch = -1
        self.training_time = 0
        self.metrics = {
            'mse': [],
            'mae': [],
            'r2': []
        }
        self.final_prediction = None    
        self.final_metrics = None        
        self.best_prediction = None     
        self.best_metrics = None 


class Benchmark:
    """
    Benchmark framework for evaluating baseline temporal models.

    This class provides a unified interface to compare simple predictive
    architectures (Linear, LSTM, and GRU) on concatenated latent representations.
    Models are evaluated using mean squared error against ground-truth latent
    states across shared time points.
    """
    def __init__(self, device, zt_dim=64, zxt_dim=64):
        self.device = device
        self.zt_dim = zt_dim
        self.zxt_dim = zxt_dim
        self.models = {
            'Linear': nn.Sequential(
                nn.Linear(zt_dim + zxt_dim, zt_dim),
                nn.ReLU()
            ).to(device),
            'LSTM': nn.LSTM(
                input_size=zt_dim + zxt_dim,
                hidden_size=zt_dim,
                batch_first=True
            ).to(device),
            'GRU': nn.GRU(
                input_size=zt_dim + zxt_dim,
                hidden_size=zt_dim,
                batch_first=True
            ).to(device)
        }
        
    def evaluate(self, z_real_dict, zxt_dict):

        results = {}
        common_times = sorted(set(z_real_dict.keys()) & set(zxt_dict.keys()))
        if not common_times:
            raise ValueError("no common times")

        input_data = torch.stack([
            torch.cat([z_real_dict[t], zxt_dict[t]], dim=-1)
            for t in common_times
        ])
        target_data = torch.stack([z_real_dict[t] for t in common_times])
        for name, model in self.models.items():
            with torch.no_grad():
                try:
                    if name == 'Linear':

                        flat_input = input_data.flatten(0, 1)
                        pred = model(flat_input).view_as(target_data)
                    else:  
                        pred, _ = model(input_data.transpose(0, 1))  
                        pred = pred.transpose(0, 1)  
                    results[name] = F.mse_loss(pred, target_data).item()
                except Exception as e:
                    print(f"{name} failed: {str(e)}")
                    results[name] = float('nan')
        
        return results


def load_model_and_data(model_path, zt_path, zxt_path):
    """
    Load trained models and corresponding latent representations from disk.

    This function restores serialized model objects, switches them to evaluation
    mode, and loads aligned latent dictionaries for downstream inference or
    benchmarking.
    """
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    
    for key in model_dict:
        if hasattr(model_dict[key], 'eval'):
            model_dict[key].eval()
    original_z_real_dict = torch.load(zt_path)
    original_zxt_dict = torch.load(zxt_path)    
    original_z_real_dict = dict(sorted(original_z_real_dict.items()))
    original_zxt_dict = dict(sorted(original_zxt_dict.items()))
    
    return model_dict, original_z_real_dict, original_zxt_dict


def save_results(results, args, epochs, input_time_points, predict_time_points, filename='training_results.pkl'):
    """
    Serialize training configurations and results to disk.

    This function saves model outputs together with experiment metadata,
    including training epochs, time-point settings, and relevant runtime
    parameters, enabling reproducible analysis and post hoc evaluation.
    """
    save_dir = getattr(args, 'save_dir', '.') 
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)

    with open(path, 'wb') as f:
        pickle.dump({
            'config': {  
                'epochs': epochs,
                'input_time_points': input_time_points,
                'predict_time_points': predict_time_points,
                'real_times_keep': getattr(args, 'real_times_keep', None),
                'pred_times_keep': getattr(args, 'pred_times_keep', None),
                'plot_format': getattr(args, 'plot_format', 'pdf'),
                'auto_ancestor': getattr(args, 'auto_ancestor', True),
                'adata_path': getattr(args, 'adata_path', None),
                'scvi_model_path': getattr(args, 'scvi_model_path', None)
             },
            'results': results
        }, f)


def split_samples(z_real_dict, zxt_dict, test_ratio=0.3):
    """
    Split latent representations into training and testing subsets.

    For each time point, samples are randomly partitioned into train and test
    sets according to the specified ratio, while preserving alignment between
    transcriptional and auxiliary latent spaces.
    """
    train_real, test_real = {}, {}
    train_zxt, test_zxt = {}, {}
    
    for t in z_real_dict:
        zt = z_real_dict[t]  
        zxt = zxt_dict[t]    
        n_samples = zt.shape[0]
        indices = torch.randperm(n_samples)
        split = int(n_samples * (1 - test_ratio))
        train_real[t] = zt[indices[:split]]
        test_real[t] = zt[indices[split:]]
        train_zxt[t] = zxt[indices[:split]]
        test_zxt[t] = zxt[indices[split:]]
    
    for t in z_real_dict:
        print(f"Time {t}:")
        print(f"  sample counts: {len(train_real[t])}  ")
    
    return train_real, train_zxt, test_real, test_zxt
    
