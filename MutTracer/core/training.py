import torch
import os
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
from .data_utils import *
from .models import *  

class TrainingSystem:
    """
    Training and inference system for bidirectional temporal prediction of latent states.

    This class implements the optimization framework for learning a bidirectional
    temporal predictor that models the evolution of mutation-related latent states
    (zt) and transcriptional auxiliary latent states (zxt) across discrete cellular
    generations. It integrates supervised reconstruction losses, trajectory-level
    consistency constraints, temporal smoothness regularization, and ancestor state
    constraints to enforce biologically plausible lineage dynamics. The system
    supports teacher forcing during training, adaptive loss weighting, and both
    forward and backward temporal prediction to infer unobserved intermediate and
    ancestral cellular states.
    """
    def __init__(self, zt_dim, zxt_dim, hidden_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.zt_dim = zt_dim
        self.zxt_dim = zxt_dim
        self.total_dim = zt_dim + zxt_dim 
        #self.aligner = FeatureAligner(zt_dim, zxt_dim, hidden_dim).to(self.device)
        self.predictor = BidirectionalPredictor(zt_dim, zxt_dim, hidden_dim).to(self.device)
        for param in self.predictor.parameters():
            param.requires_grad = True
        self.similarity_threshold = 0.98 
        self.valid_steps_cache = defaultdict(list)
        self.loss_history = defaultdict(list)
        self.teacher_forcing_prob = 0.8
        self.best_loss = float('inf')
        self.tf_decay = 0.985
        self.z0_constraints = []
        self.pred_dict = {}  
        self.min_tf_prob = 0.1
        self.early_stopper = EarlyStopper(patience=5)
        self.optimizer = optim.AdamW(self.predictor.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min',   
            factor=0.5,  
            patience=2, 
        )
        self.eval_metrics = {
            'zt_mae': [],
            'zxt_mae': [],
            'traj_diff': []
        }
        self.predictor.train() 
    def euclidean_distance(z1, z2):
        return torch.norm(z1 - z2, p=2, dim=-1)
    def _calculate_continuity_loss(self, pred_zt_dict, pred_zxt_dict, sorted_times):

        loss = 0.0
        for i in range(len(sorted_times)-1):
            t_curr, t_next = sorted_times[i], sorted_times[i+1]
            
            delta_zt = pred_zt_dict[t_next] - pred_zt_dict[t_curr]
            loss += torch.mean(delta_zt.pow(2))

            delta_zxt = pred_zxt_dict[t_next] - pred_zxt_dict[t_curr]
            loss += 0.5 * torch.mean(delta_zxt.pow(2)) 
    
        return loss / (len(sorted_times)-1)

    def _calculate_smoothness_loss(self, pred_zt_dict, pred_zxt_dict, sorted_times):

        loss = 0.0
        if len(sorted_times) >= 3:
            for i in range(1, len(sorted_times)-1):
                t_prev, t_curr, t_next = sorted_times[i-1], sorted_times[i], sorted_times[i+1]
            
                second_diff_zt = pred_zt_dict[t_next] - 2*pred_zt_dict[t_curr] + pred_zt_dict[t_prev]
                loss += torch.mean(second_diff_zt.pow(2))
            
                second_diff_zxt = pred_zxt_dict[t_next] - 2*pred_zxt_dict[t_curr] + pred_zxt_dict[t_prev]
                loss += 0.5 * torch.mean(second_diff_zxt.pow(2))
    
        return loss / max(1, len(sorted_times)-2)

    def _calculate_velocity_consistency(self, pred_zt_dict, z_real_dict, sorted_times):

        loss = 0.0
        valid_pairs = 0
        
        for i in range(len(sorted_times)-1):
            t_curr, t_next = sorted_times[i], sorted_times[i+1]
        
            if t_curr in z_real_dict and t_next in z_real_dict:
                pred_velocity = pred_zt_dict[t_next] - pred_zt_dict[t_curr]
                real_velocity = z_real_dict[t_next] - z_real_dict[t_curr]
                loss += 1 - F.cosine_similarity(pred_velocity, real_velocity, dim=-1).mean()
                valid_pairs += 1
    
        return loss / valid_pairs if valid_pairs > 0 else 0.0

    def composite_loss(self, pred_zt_dict, pred_zxt_dict, z_real_dict, zxt_dict, z0_candidates,pred_output=None):
        losses = []
        device = self.device
        zt_supervised = torch.tensor(0.0, device=device)
        zxt_supervised = torch.tensor(0.0, device=device)
        all_times = sorted(list(pred_zt_dict.keys()))
        valid_count = 0
        for t in pred_zt_dict:
            if t in z_real_dict:
                batch_size, zt_dim = pred_zt_dict[t].shape
                zt_mse = F.mse_loss(pred_zt_dict[t], z_real_dict[t], reduction='sum') / (batch_size * zt_dim)
            
                _, zxt_dim = pred_zxt_dict[t].shape
                zxt_mse = F.mse_loss(pred_zxt_dict[t], zxt_dict[t], reduction='sum') / (batch_size * zxt_dim)
            
                zt_supervised += zt_mse
                zxt_supervised += zxt_mse
                valid_count += 1

        if valid_count > 0:
            losses.append(1.2 * zt_supervised / valid_count) 
            losses.append(0.8 * zxt_supervised / valid_count)

        if len(all_times) >= 3:  
            for i in range(1, len(all_times)-1):
                t_prev = all_times[i-1]
                t_curr = all_times[i]
                t_next = all_times[i+1]
            
                pred_prev = torch.cat([pred_zt_dict[t_prev], pred_zxt_dict[t_prev]], dim=-1)
                pred_curr = torch.cat([pred_zt_dict[t_curr], pred_zxt_dict[t_curr]], dim=-1)
                pred_next = torch.cat([pred_zt_dict[t_next], pred_zxt_dict[t_next]], dim=-1)
            
                linear_interp = 0.5 * (pred_prev + pred_next)
                interp_loss = F.mse_loss(pred_curr, linear_interp)
            
                delta_prev = pred_curr - pred_prev
                delta_next = pred_next - pred_curr
                smoothness_loss = F.mse_loss(delta_prev, delta_next)

                losses.append(0.8 * interp_loss)  
                losses.append(0.2 * smoothness_loss)

        if len(z0_candidates) >= 1:  

           candidates = torch.stack(z0_candidates)  # [N, D]
           sim_matrix = F.cosine_similarity(
               candidates.unsqueeze(1),  # [N, 1, D]
               candidates.unsqueeze(0),  # [1, N, D]
               dim=-1
           )

           ancestor_loss = F.relu(0.8 - sim_matrix).mean()  
           losses.append(0.3 * ancestor_loss)  

        sorted_times = sorted(pred_zt_dict.keys())
        if len(sorted_times) >= 1:

            zt_seq = torch.stack([pred_zt_dict[t] for t in sorted_times])  # [T, B, D_zt]
            zxt_seq = torch.stack([pred_zxt_dict[t] for t in sorted_times]) # [T, B, D_zxt]
        
            time_weights = 1.0 / (1 + torch.arange(len(sorted_times)-1, device=device)) # [T-1]
        
            zt_diffs = zt_seq[:-1] - zt_seq[1:]  # [T-1, B, D_zt]
            zt_time_loss = torch.mean(time_weights.view(-1,1,1) * zt_diffs.pow(2))
        
            zxt_diffs = zxt_seq[:-1] - zxt_seq[1:] # [T-1, B, D_zxt]
            zxt_time_loss = torch.mean(time_weights.view(-1,1,1) * zxt_diffs.pow(2))
        
            time_loss = 0.8 * zt_time_loss + 0.4 * zxt_time_loss  # zt权重稍高
            losses.append(time_loss)

        td_loss = torch.tensor(0.0, device=device)
        sorted_t = sorted(pred_zt_dict.keys())
        if len(sorted_t) > 1:
            for i in range(1, len(sorted_t)):
                delta_pred = pred_zt_dict[sorted_t[i]] - pred_zt_dict[sorted_t[i-1]]
                if sorted_t[i] in z_real_dict and sorted_t[i-1] in z_real_dict:
                    delta_real = z_real_dict[sorted_t[i]] - z_real_dict[sorted_t[i-1]]
                    td_loss += F.mse_loss(delta_pred, delta_real)
                    direction_loss = 1 - F.cosine_similarity(delta_pred, delta_real, dim=-1).mean()
                    td_loss += direction_loss * 0.5
            td_loss /= (len(sorted_t) - 1)

        sorted_times = sorted(pred_zt_dict.keys())
        if len(sorted_times) > 1:
            zt_seq = torch.stack([pred_zt_dict[t] for t in sorted_times])
        
            forward_diffs = zt_seq[1:] - zt_seq[:-1]
            forward_loss = torch.mean(forward_diffs.pow(2))
        
            backward_diffs = zt_seq[:-1] - zt_seq[1:]
            backward_loss = torch.mean((backward_diffs + 0.5).pow(2))  
        
            losses.append(1.2 * backward_loss+0.8*forward_loss)

        if len(pred_zt_dict) >= 1:  

            real_times = sorted([t for t in z_real_dict.keys() if t in pred_zt_dict])
            real_traj = torch.stack([z_real_dict[t] for t in real_times])  # [T_real, B, D]

            pred_traj = torch.stack([pred_zt_dict[t] for t in real_times])  # [T_real, B, D]

            traj_loss = F.mse_loss(
                pred_traj.diff(dim=0), 
                real_traj.diff(dim=0),  
                reduction='mean'
            )
            real_traj_zxt = torch.stack([zxt_dict[t] for t in real_times])  # [T_real, B, D]
        
            pred_traj_zxt = torch.stack([pred_zxt_dict[t] for t in real_times])  # [T_real, B, D]
        
            traj_loss_zxt = F.mse_loss(
                pred_traj_zxt.diff(dim=0), 
                real_traj_zxt.diff(dim=0), 
                reduction='mean'
            )
            pred_zt = torch.stack([pred_zt_dict[t] for t in real_times])
            real_zt = torch.stack([z_real_dict[t] for t in real_times])
            endpoint_loss = 0.5 * (F.mse_loss(pred_zt[0], real_zt[0]) + 
                              F.mse_loss(pred_zt[-1], real_zt[-1]))
            losses.append(0.3 * traj_loss+endpoint_loss)#+0.1*traj_loss_zxt)
        for t in pred_zt_dict:
            if t in z_real_dict:
                zt_var = torch.var(pred_zt_dict[t], dim=0).mean()  
                zxt_var = torch.var(pred_zxt_dict[t], dim=0).mean()

        if len(pred_zt_dict) >= 2:
            sorted_times = sorted(pred_zt_dict.keys())
            continuity_loss = self._calculate_continuity_loss(
                pred_zt_dict, pred_zxt_dict, sorted_times
            )
            losses.append(0.5 * continuity_loss) 
            smoothness_loss = self._calculate_smoothness_loss(
                pred_zt_dict, pred_zxt_dict, sorted_times
            )
            velocity_loss = self._calculate_velocity_consistency(
                pred_zt_dict, z_real_dict, sorted_times
            )
            losses.append(0.3 * velocity_loss)

        total_loss = sum(losses) if losses else torch.tensor(0.0, device=device)
    

        self.loss_history['total'].append(total_loss.item())
        if valid_count > 0:
            self.loss_history['zt_supervised'].append((zt_supervised / valid_count).item())
            self.loss_history['zxt_supervised'].append((zxt_supervised / valid_count).item())
            
        if len(sorted_t) > 1:
            self.loss_history['traj'].append(traj_loss.item())
            self.loss_history['ancestor'].append(ancestor_loss.item())
            self.loss_history['continuity'].append(continuity_loss.item())
            self.loss_history['time'].append(time_loss.item())

        return total_loss

    def combined_similarity(self, z1, z2):

        cosine_sim = F.cosine_similarity(z1, z2, dim=-1)
        euclidean_dist = torch.norm(z1 - z2, p=2, dim=-1)
        return 0.5 * (1 - cosine_sim) + 0.5 * euclidean_dist


    def predict_sequence_past(self, z_real_dict, zxt_dict):

        all_preds = defaultdict(list)
        z0_candidates = []
        input_times = sorted(z_real_dict.keys())
        time_indices = torch.tensor(input_times, device=self.device)
    
        t_values = sorted(z_real_dict.keys())
        t_min_pred = 0
        t_max_pred = (max(t_values) + 1) if t_values else 0
        delta_t = time_indices.diff() if len(time_indices) > 1 else torch.zeros(1, device=self.device)
        delta_t_dict = {}
        for i in range(1, len(input_times)):
            delta_t_dict[input_times[i]] = delta_t[i-1]
        for t_current in t_values:

            t_tensor = torch.tensor([t_current], device=self.device)
            current_zt = z_real_dict[t_current].clone()
            current_zxt = zxt_dict[t_current].clone()
        
            for i, t in enumerate(range(t_current-1, t_min_pred-1, -1)):
                current_delta = torch.tensor([t_current - t], device=self.device).float()
                zt_pred, zxt_pred, _, _, _, _ = self.predictor(
                    current_zt.unsqueeze(0),
                    current_zxt.unsqueeze(0),
                    torch.tensor([t], device=self.device),
                    delta_t=current_delta.unsqueeze(0) 
                )

                pred_entry = torch.cat([zt_pred.squeeze(0), zxt_pred.squeeze(0)], dim=-1)
                all_preds[t].append(pred_entry)
            
                if t in z_real_dict and random.random() < self.teacher_forcing_prob:
                    current_zt = z_real_dict[t]
                    current_zxt = zxt_dict[t]
                else:
                    current_zt = zt_pred.squeeze(0)
                    current_zxt = zxt_pred.squeeze(0)
            
                if t == 0:
                    z0_candidates.append(current_zt.detach())
        
            current_zt = z_real_dict[t_current].clone()
            current_zxt = zxt_dict[t_current].clone()
        
            for i, t in enumerate(range(t_current+1, t_max_pred+1)):
                zt_pred, zxt_pred, _, _, _, _ = self.predictor(
                    current_zt.unsqueeze(0),
                    current_zxt.unsqueeze(0),
                    torch.tensor([t], device=self.device),
                    delta_t=current_delta.unsqueeze(0) 
                )
                             
                all_preds[t].append(torch.cat([zt_pred.squeeze(0), zxt_pred.squeeze(0)], dim=-1))
            
                if t in z_real_dict and random.random() < self.teacher_forcing_prob:
                    current_zt = z_real_dict[t]
                    current_zxt = zxt_dict[t]
                else:
                    current_zt = zt_pred.squeeze(0)
                    current_zxt = zxt_pred.squeeze(0)
                if t == t_max_pred:  
                    z0_candidates.append(zt_pred.squeeze(0).detach())
    
        merged_preds = {
            t: {
                "prediction": torch.stack(pred_list).mean(dim=0),
                "ground_truth": torch.cat([z_real_dict[t], zxt_dict[t]], dim=-1) if t in z_real_dict else None
            }
            for t, pred_list in all_preds.items()
        }
    
        return merged_preds, z0_candidates

    def mean_pairwise_cosine_similarity(self, embeddings):
        sims = []
        n = embeddings.size(0)
        for i in range(n):
            for j in range(i + 1, n):
                sim = torch.nn.functional.cosine_similarity(
                    embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)
                ).item()
                sims.append(sim)
        return sum(sims) / len(sims) if sims else 0.0

    def predict_sequence(self, z_real_dict, zxt_dict, sim_threshold=0.9):
        TIME_OFFSET = 10 
        device = next(iter(z_real_dict.values())).device
        all_preds = defaultdict(list)
        z0_candidates = []
        min_variance = float('inf')
        best_pred0 = None 

        t_values = sorted([t + TIME_OFFSET for t in z_real_dict.keys()])
        t_max_pred = max(t_values) + 1 if t_values else 0
        t_min = min(t_values) if t_values else 0
        time_decay = lambda t: 1.0 / (1 + abs(t))  

        def is_converged(features):
            return features.std(dim=0).mean() < 0.1  

        for t_current in t_values:

            current_zt = z_real_dict[t_current - TIME_OFFSET].clone()
            current_zxt = zxt_dict[t_current - TIME_OFFSET].clone()
        
            for t in range(t_current-1, -1, -1):  

                delta_t = torch.tensor([t_current - t], device=self.device).float() * time_decay(t)
                zt_pred, zxt_pred, *_ = self.predictor(
                    current_zt.unsqueeze(0),
                    current_zxt.unsqueeze(0),
                    torch.tensor([t], device=device).float(),  
                    delta_t=delta_t
                )

                combined = torch.cat([zt_pred.squeeze(0), zxt_pred.squeeze(0)], dim=-1)
                current_variance = combined.var(dim=0).mean().item()
                if t <= 0 and current_variance < min_variance:
                    min_variance = current_variance
                    best_pred0 = combined.detach().clone()
            
                all_preds[t].append(combined)
                current_zt, current_zxt = zt_pred.squeeze(0), zxt_pred.squeeze(0)

            current_zt = z_real_dict[t_current - TIME_OFFSET].clone()
            current_zxt = zxt_dict[t_current - TIME_OFFSET].clone()
        
            for t in range(t_current+1, t_max_pred+1):
                delta_t = torch.tensor([t - t_current], 
                                     dtype=torch.float32,
                                     device=device).unsqueeze(0)
            
                zt_pred, zxt_pred, *_ = self.predictor(
                    current_zt.unsqueeze(0),
                    current_zxt.unsqueeze(0),
                    torch.tensor([t], device=device).float(),
                    delta_t=delta_t
                )
                all_preds[t].append(torch.cat([
                    zt_pred.squeeze(0), 
                    zxt_pred.squeeze(0)
                ], dim=-1))
        if best_pred0 is not None:
            z0_candidates.append(best_pred0)

        final_preds = {
            t - TIME_OFFSET: {
                "prediction": torch.stack(pred_list).mean(dim=0),
                "ground_truth": torch.cat([
                    z_real_dict[t - TIME_OFFSET], 
                    zxt_dict[t - TIME_OFFSET]
                ], dim=-1) if (t - TIME_OFFSET) in z_real_dict else None
            }
            for t, pred_list in all_preds.items()
        }
    
        return final_preds, z0_candidates


    def train_step(self, z_real_dict, zxt_dict,masks_dict=None):
        
        try:
            self.predictor.train()  
            self.optimizer.zero_grad()

            z_real_dict = {t: (z - z.mean()) / (z.std() + 1e-6) for t, z in z_real_dict.items()}
            zxt_dict = {t: (z - z.mean()) / (z.std() + 1e-6) for t, z in zxt_dict.items()}
            z_real_dict = {t: z.clone().detach().requires_grad_(True) for t, z in z_real_dict.items()}
            zxt_dict = {t: z.clone().detach().requires_grad_(True) for t, z in zxt_dict.items()}

            input_times = sorted(z_real_dict.keys())
            time_indices = torch.tensor(input_times, device=self.device)

            batch_size = next(iter(z_real_dict.values())).shape[0]
            zt_stack = torch.stack([z_real_dict[t] for t in input_times], dim=1)  # [B,T,D]
            zxt_stack = torch.stack([zxt_dict[t] for t in input_times], dim=1)     # [B,T,D]
        
            zt_pred, zxt_pred, mu, logvar, _, _ = self.predictor(
                zt_stack,
                zxt_stack,
                time_indices.unsqueeze(0).expand(batch_size, -1)  # [B,T]
            )
        
            pred_output, z0_candidates = self.predict_sequence(z_real_dict, zxt_dict)
            pred_zt_dict = {
                t: data["prediction"][:, :self.zt_dim].requires_grad_(True)
                for t, data in pred_output.items()
            }
            pred_zxt_dict = {
                t: data["prediction"][:, self.zt_dim:].requires_grad_(True)
                for t, data in pred_output.items()
            }

            z_real_dict_loss = {
                t: data["ground_truth"][:, :self.zt_dim]
                for t, data in pred_output.items() 
                if data["ground_truth"] is not None
            }
            zxt_dict_loss = {
                t: data["ground_truth"][:, self.zt_dim:]
                for t, data in pred_output.items()
                if data["ground_truth"] is not None
            }
        
            loss = self.composite_loss(pred_zt_dict, pred_zxt_dict, 
                                 z_real_dict_loss, zxt_dict_loss, 
                                 z0_candidates,pred_output=pred_output)

            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 5.0)
                self.optimizer.step()
            else:
                raise RuntimeError("Loss does not require gradients")

            self.update_teacher_forcing(loss.item())
            self.pred_output = {t: z for t, z in pred_output.items()} 
            self.z0_candidates = [z for z in z0_candidates]
            
            grad_norms = [p.grad.norm().item() for p in self.predictor.parameters() if p.grad is not None]

            if self.pred_output:
                pred_values = torch.stack([v['prediction'] for v in self.pred_output.values()])   
                return loss.item()
        except Exception as e:
            print(f"error : {str(e)}")
            traceback.print_exc()
            return None

    def train_loop(self, z_real_dict, zxt_dict, epochs):

        val_mae = {
            'total': total_mae.item(),
            'zt': zt_mae.item(),
            'zxt': zxt_mae.item()
        }
        times = torch.tensor(list(z_real_dict.keys())).to(device)
        if not hasattr(system, 'val_mae_history'):
            system.val_mae_history = {k: [] for k in val_mae}

        for k in val_mae:
            system.val_mae_history[k].append(val_mae[k])
        
        for epoch in range(epochs):
            verbose = (epoch % 10 == 0)
            loss = self.train_step(z_real_dict, zxt_dict)
        
    def update_teacher_forcing(self, current_loss):

        if current_loss < self.best_loss:
            self.best_loss = current_loss
            grad_norm = sum(p.grad.norm() for p in self.predictor.parameters() if p.grad is not None)
            self.teacher_forcing_prob = max(
                self.min_tf_prob,
                0.5 * (1 + torch.sigmoid(torch.tensor(grad_norm / 1000. - 1.)))
            )
        else:
            self.teacher_forcing_prob = min(
                0.7, 
                self.teacher_forcing_prob * 1.05
            )  

    def evaluate_predictions(self, z_real_dict, zxt_dict):

        self.predictor.eval()  
        metrics = {
            'zt_mae': [],
            'zxt_mae': [],
            'zt_cosine': [],
            'zxt_cosine': [],
            'traj_mae': [],
            'traj_cosine': []
        }
    
        pred_dict = {t: z.to(self.device) for t, z in self.pred_dict.items()}
        z_real_dict = {t: z.to(self.device) for t, z in z_real_dict.items()}
        zxt_dict = {t: z.to(self.device) for t, z in zxt_dict.items()}

        common_times = sorted(set(pred_dict.keys()) & set(z_real_dict.keys()))
        for t in common_times:

            metrics['zt_mae'].append(F.l1_loss(pred_dict[t][:, :self.zt_dim], z_real_dict[t]).item())
            metrics['zt_cosine'].append(F.cosine_similarity(
                pred_dict[t][:, :self.zt_dim].flatten(),
                z_real_dict[t].flatten(),
                dim=0
            ).item())
        
            metrics['zxt_mae'].append(F.l1_loss(pred_dict[t][:, self.zt_dim:], zxt_dict[t]).item())
            metrics['zxt_cosine'].append(F.cosine_similarity(
                pred_dict[t][:, self.zt_dim:].flatten(),
                zxt_dict[t].flatten(),
                dim=0
            ).item())
    
        if len(common_times) > 1:
            for i in range(1, len(common_times)):
                t_curr, t_prev = common_times[i], common_times[i-1]

                delta_pred = pred_dict[t_curr] - pred_dict[t_prev]

                delta_real = torch.cat([
                    z_real_dict[t_curr] - z_real_dict[t_prev],
                    zxt_dict[t_curr] - zxt_dict[t_prev]
                ], dim=-1)
            
                metrics['traj_mae'].append(F.l1_loss(delta_pred, delta_real).item())
                metrics['traj_cosine'].append(F.cosine_similarity(
                    delta_pred.flatten(),
                    delta_real.flatten(),
                    dim=0
                ).item())
    
        return {
            k: np.mean(v) if v else float('nan') 
            for k, v in metrics.items()
        }

