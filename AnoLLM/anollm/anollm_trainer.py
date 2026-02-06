

import os
import random
import json
import numpy as np
import torch
import typing as tp
import torch.distributed as dist
import torch.nn as nn
import math
import itertools

from typing import List
from typing import Dict, Any, Union
from pathlib import Path
from transformers import Trainer
from sklearn import metrics
from anollm.anollm_dataset import AnoLLMDataCollator, AnoLLMDataLoader
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F 
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset, DistributedSampler
from collections import defaultdict
from typing import Dict, Any
from torch.nn.functional import log_softmax
from transformers.utils import is_sagemaker_mp_enabled
from transformers.utils.import_utils import (
    is_torch_xpu_available,
    is_torch_mlu_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_mps_available,
)

from typing import Iterator

class RatioBatchSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        batch_size: int = 16,
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = False,
        drop_last: bool = False,
        abnormal_ratio: float = None
    ):
        super().__init__(
            dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            drop_last=drop_last
        )
        self.batch_size = batch_size
        
        abnormal: int = int(self.batch_size * abnormal_ratio)  # Number of anomaly samples per batch
        self.n_normal = self.batch_size - abnormal
        self.n_anomaly = abnormal

        # Extract labels
        labels = dataset.labels
        normal_idx = [i for i, y in enumerate(labels) if y == 0]
        anomaly_idx = [i for i, y in enumerate(labels) if y == 1]

        print(f"Rank {self.rank}: Normal samples: {len(normal_idx)}, Anomaly samples: {len(anomaly_idx)}")

        # Distributed sample assignment
        self.rank_normal = normal_idx[self.rank :: self.num_replicas]
        self.rank_anomaly = anomaly_idx[self.rank :: self.num_replicas]

        print(f"Rank {self.rank}: After distribution - Normal: {len(self.rank_normal)}, Anomaly: {len(self.rank_anomaly)}")

        # Compute minimum available cycles
        normal_batches = len(self.rank_normal) / self.n_normal
        anomaly_batches = len(self.rank_anomaly) / self.n_anomaly
        
        # If any class has zero samples, fall back to old logic
        if len(self.rank_normal) < self.n_normal or len(self.rank_anomaly) < self.n_anomaly:
            self.cycles = 0
        else:
            # Round up so the larger class can be fully used
            self.cycles = max(math.ceil(normal_batches), math.ceil(anomaly_batches))

        # If full batches cannot be formed, use all available samples
        if self.cycles == 0:
            print(f"Rank {self.rank}: Warning - Not enough samples for complete batches")
            print(f"Rank {self.rank}: Using all available samples instead")
            
            # Use all available samples without guaranteeing ratio
            self.indices = self.rank_normal + self.rank_anomaly
            
            # If samples are still too few, reduce batch size
            if len(self.indices) < self.batch_size:
                print(f"Rank {self.rank}: Very few samples available")
        else:
            # Build ratio-based batches
            self.indices = []
            # Use itertools.cycle for infinite looping
            norm_stream = itertools.cycle(self.rank_normal)
            anom_stream = itertools.cycle(self.rank_anomaly)

            for _ in range(self.cycles):
                # Take normal
                self.indices.extend(
                    list(itertools.islice(norm_stream, self.n_normal))
                )
                # Take anomalies
                self.indices.extend(
                    list(itertools.islice(anom_stream, self.n_anomaly))
                )

        print(f"Rank {self.rank}: Final indices count: {len(self.indices)}")

        if len(self.indices) == 0:
            raise ValueError(
                f"Rank {self.rank} received empty sampler. "
                f"Normal: {len(self.rank_normal)}, Anomaly: {len(self.rank_anomaly)}"
            )

    def __iter__(self) -> Iterator[int]:
        if self.shuffle:
            # Shuffle the two lists separately
            random.shuffle(self.rank_normal)
            random.shuffle(self.rank_anomaly)
            # Rebuild indices
            self._rebuild_indices()
        yield from self.indices

    def _rebuild_indices(self):
        """Rebuild indices."""
        # If full batches cannot be formed, use all available samples
        if self.cycles == 0:
            print(f"Rank {self.rank}: Warning - Not enough samples for complete batches")
            print(f"Rank {self.rank}: Using all available samples instead")
            
            # Use all available samples without guaranteeing ratio
            self.indices = self.rank_normal + self.rank_anomaly
            
            # If samples are still too few, reduce batch size
            if len(self.indices) < self.batch_size:
                print(f"Rank {self.rank}: Very few samples available")
        else:
            # Build ratio-based batches
            self.indices = []
            # Use itertools.cycle for infinite looping
            norm_stream = itertools.cycle(self.rank_normal)
            anom_stream = itertools.cycle(self.rank_anomaly)

            for _ in range(self.cycles):
                # Take normal
                self.indices.extend(
                    list(itertools.islice(norm_stream, self.n_normal))
                )
                # Take anomalies
                self.indices.extend(
                    list(itertools.islice(anom_stream, self.n_anomaly))
                )

    def __len__(self) -> int:
        return len(self.indices)



def _seed_worker(_):
	"""
	Helper function to set worker seed during Dataloader initialization.
	"""
	worker_seed = torch.initial_seed() % 2**32
	random.seed(worker_seed)
	np.random.seed(worker_seed)
	torch.manual_seed(worker_seed)
	torch.cuda.manual_seed_all(worker_seed)
                

class AnoLLMTrainer(Trainer):
    
    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        data_collator = self.data_collator
        train_dataset = (
			self.train_dataset
		)  # self._remove_unused_columns(self.train_dataset, description="training")

        # Propagate permutation-control settings to the dataset (if supported).
        graph_based_rank = getattr(self, "graph_based_rank", "no")
        sorted_set = getattr(self, "sorted_set", None)
        if hasattr(train_dataset, "set_graph_based_rank") and callable(getattr(train_dataset, "set_graph_based_rank")):
            train_dataset.set_graph_based_rank(graph_based_rank=graph_based_rank, sorted_set=sorted_set)
        else:
            # fallback: attach attributes dynamically
            setattr(train_dataset, "graph_based_rank", graph_based_rank)
            setattr(train_dataset, "sorted_set", sorted_set)

        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        batch_sampler = RatioBatchSampler(
            dataset = train_dataset,
            batch_size   = self._train_batch_size,
            num_replicas = world_size,
            rank         = local_rank,
            shuffle      = True,
            drop_last    = False,
            abnormal_ratio = self.abnormal_ratio,
            )
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=False, drop_last=True)

        return DataLoader(
			train_dataset,
			batch_size=self._train_batch_size,
			sampler=batch_sampler,
			collate_fn=data_collator,
			drop_last=self.args.dataloader_drop_last,
			num_workers=self.args.dataloader_num_workers,
			pin_memory=self.args.dataloader_pin_memory,
			worker_init_fn=_seed_worker,
		)
    def __init__(
        self,
        *args,
        log_file: str = "data",
        abnormal_ratio = None,
        graph_based_rank: str = "no",
        sorted_set = None,
        cos = False,
        weights_path: str = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        log_dir = Path(log_file)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = log_dir / "training_log.csv"
        self.eval_log_file = log_dir / "eval_log.csv"
        self.abnormal_ratio = abnormal_ratio
        self.graph_based_rank = graph_based_rank
        self.sorted_set = sorted_set
        self.cos = cos
        
        # Load weights for evaluation if provided
        self.weights_map = {}
        if weights_path:
            try:
                with open(weights_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, dict) and isinstance(payload.get("weights"), dict):
                    self.weights_map = payload.get("weights", {})
                else:
                    print(f"Warning: weights_path {weights_path} content format incorrect. Expected top-level 'weights' dict.")
            except Exception as e:
                 print(f"Warning: Failed to load weights from {weights_path}: {e}")

        with open(self.log_file, "w") as f:
            f.write("step,loss,learning_rate,epoch\n")

        with open(self.eval_log_file, "w") as f:
            f.write(
                "step,epoch,eval_auc_roc,eval_loss,eval_perplexity,"
                "eval_loss_normal,eval_loss_anomaly,"
                "eval_perplexity_normal,eval_perplexity_anomaly,"
                "eval_weighted_auc_roc,eval_weighted_perplexity_normal,eval_weighted_perplexity_anomaly\n"
            )

    def training_step(
        self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None
    ) -> torch.Tensor:
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        if is_sagemaker_mp_enabled():
            from transformers.trainer_pt_utils import smp_forward_backward
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}
        # Get current step log info
        logs = {
        "step": self.state.global_step,
        "loss": loss.item(),
        "learning_rate": self._get_learning_rate(),
        "epoch": self.state.epoch,
        }
        
        # Write to log file
        with open(self.log_file, "a") as f:
            f.write(f"{logs['step']},{logs['loss']:.6f},{logs['learning_rate']:.8f},{logs['epoch']:.2f}\n")

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss, **kwargs)
            # Finally we need to normalize the loss for reporting
        if num_items_in_batch is None:
            return loss.detach() / self.args.gradient_accumulation_steps
        
        return loss.detach()


    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    # # ---------- Label smoothing strength (tune as needed) ----------
    #     eps = 0.1  # 0.1 is a common value
    # # --------------------------------------

    #     normal, anomaly = inputs
    #     loss = torch.tensor(0.0, device=model.device, requires_grad=True)
    #     loss = loss * next(model.parameters()).sum() * 0.0   # Ensure graph exists

    #     def _nll(batch):
    #         if batch is None or batch["input_ids"].size(0) == 0:
    #             return None
    #         out = model(**batch)
    #         logits = out.logits.contiguous()          # [B, L, V]
    #         labels = batch["labels"].contiguous()     # [B, L]

    #     # 1. Build smoothed label distribution
    #         B, L, V = logits.shape
    #         confidence = 1.0 - eps
    #         low_conf = eps / (V - 1)  # Spread remaining prob uniformly to non-true classes
    #         soft_labels = torch.full_like(logits, low_conf)  # [B, L, V]
    #     # Set correct positions to confidence
    #         soft_labels.scatter_(2, labels.unsqueeze(2), confidence)

    #     # 2. Ignore -100: set corresponding prob to 0 and exclude from sums
    #         ignore_mask = (labels == -100).unsqueeze(2)  # [B, L, 1]
    #         soft_labels = soft_labels.masked_fill(ignore_mask, 0.0)

    #     # 3. Manual log-softmax + KL(soft_labels || probs)
    #         log_probs = log_softmax(logits, dim=-1)
    #     # KL = sum( p * log(p/q) ) = sum( p * (log p - log q) )
    #     # soft_labels already has no log, so directly:
    #     # loss = sum( soft_labels * (-log_probs) )
    #     # Then divide by valid token count to get mean token loss
    #         nll = torch.sum(soft_labels * (-log_probs))
    #         normalizer = (labels != -100).sum()  # Valid token count
    #         if normalizer > 0:
    #             nll = nll / normalizer
    #         else:
    #             nll = torch.tensor(0.0, device=logits.device, requires_grad=True)
    #         return nll

    # # ------------ Normal samples -----------
    #     nll_n = _nll(normal) if normal is not None else None
    #     out_normal = model(**normal) if normal is not None else None

    # # ------------ Anomaly samples ------------
    #     nll_a = _nll(anomaly) if anomaly is not None else None
    #     out_anomaly = model(**anomaly) if anomaly is not None else None

    # # ------------ Contrastive loss ------------
    #     if nll_n is not None and nll_a is not None:
    #         loss = nll_n - nll_a
    #     elif nll_n is not None:
    #         loss = nll_n
    #     elif nll_a is not None:
    #         loss = -nll_a
    # # else: keep loss as 0

    #     outputs = out_normal or out_anomaly
    #     return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self.ce_loss = CrossEntropyLoss(ignore_index=-100)
        normal, anomaly = inputs
        emb_n = emb_a = None # Fixed: Initialize embedding variables
        outputs_normal = outputs_anomaly = None
        loss = None
        if normal is None or anomaly is None:
            print("normal or anomaly is None")
        
        loss = torch.tensor(0.0, device=model.device, requires_grad=True)
        loss = loss * next(model.parameters()).sum() * 0.0

        def _nll(batch):
            if batch is None or batch["input_ids"].size(0) == 0:
                return None
            out = model(**batch)
            logits = out.logits.contiguous()           # [B, L, V]
            labels = batch["labels"].contiguous()      # [B, L]
            
            # Shift so that tokens < n predict n
            # Shift to align predictions and labels
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute loss on shifted tensors
            myloss = self.ce_loss(shift_logits.view(-1, shift_logits.size(-1)),
                       shift_labels.view(-1))
            
            return myloss
            
            return myloss
        # # ===== Debug function =====
        # def _dbg(batch, name):
        #     if batch is None or len(batch["input_ids"]) == 0:
        #         print(f"[{name}] batch is empty")
        #         return
        #     for i in range(len(batch["input_ids"])):
        #         seq_len = batch["input_ids"].size(1)          # Original length
        #         logits_keep = seq_len - 1                     # Kept length
        #         last_tok_id = batch["input_ids"][i, -1].item()
        #         last_label  = batch["labels"][i, -1].item()
        #         print(f"[{name}] seq_len={seq_len}, "
        #             f"logits_keep={logits_keep}, "
        #             f"last_token_id={last_tok_id}, "
        #             f"last_label={last_label}  <-- masked")

        def _cosine_similarity_loss(batch):
            if batch is None or batch["input_ids"].size(0) == 0:
                return None, None

            with torch.no_grad():                       # Remove this line if you want gradients
                out = model(**batch, output_hidden_states=True)
                # Take last hidden_state: [B, L, H]
                last_hidden = out.hidden_states[-1]
                mask = batch["attention_mask"].unsqueeze(-1)  # [B, L, 1]
                # Mask pad positions, then average to get sentence embedding
                emb = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)  # [B, H]

            # Pairwise cosine similarity between sentence embeddings
            emb_norm = F.normalize(emb, p=2, dim=1)      # [B, H]
            sim_mat = torch.mm(emb_norm, emb_norm.t())   # [B, B]
            # Average only upper triangle (exclude diagonal) to avoid self-similarity
            b = sim_mat.size(0)
            if b > 1:
                upper = sim_mat.triu(diagonal=1)
                cnt = b * (b - 1) / 2
                avg_sim = upper.sum() / cnt
            else:
                avg_sim = sim_mat[0, 0] * 0.0            # Single sample returns 0

            return avg_sim, emb

        
        # ------------ Normal samples -----------
        #_dbg(normal, "NORMAL")
        if normal is not None and normal["input_ids"].size(0) > 0:
        # Compute embeddings for normal samples
            if self.cos == True:
                cos_sim_n, emb_n = _cosine_similarity_loss(normal)
                print("cos_sim_n:", cos_sim_n)
                loss_n = _nll(normal)-cos_sim_n.mean() if cos_sim_n is not None else None
        # Normal sample loss should maximize similarity
            loss_n = _nll(normal) 
            out_normal = model(**normal)
        else:
            loss_n = None
            out_normal = None
            emb_n = None
    
        # ------------ Anomaly samples ------------
        #_dbg(anomaly, "ANORMAL")
        if anomaly is not None and anomaly["input_ids"].size(0) > 0:
        # Compute embeddings for anomaly samples
            if self.cos == True:
                cos_sim_a, emb_a = _cosine_similarity_loss(anomaly)
                print("cos_sim_a:", cos_sim_a)
                loss_a = _nll(anomaly)+cos_sim_a.mean() if cos_sim_a is not None else None
        # Anomaly sample loss should minimize similarity
            loss_a = _nll(anomaly) 
            out_anomaly = model(**anomaly)
        else:
            loss_a = None
            out_anomaly = None
            emb_a = None

        
        # ------------ Compute contrastive loss ------------
        print(f"loss_n: {loss_n}, loss_a: {loss_a}")
        if loss_n is not None and loss_a is not None:
            # Both normal and anomaly samples exist
            loss = loss_n - loss_a
        elif loss_n is not None:
            # Only normal samples
            loss = loss_n
        elif loss_a is not None:
            # Only anomaly samples
            loss = -loss_a
        else:
            # No samples; keep loss as 0
            pass

        if emb_n is not None and emb_a is not None:
            # Compute center of normal samples
            center_n = emb_n.mean(dim=0)
        
            # Compute distance from anomaly samples to normal center (extra regularization)
            dist_a = F.mse_loss(emb_a, center_n.unsqueeze(0).expand(emb_a.size(0), -1), reduction='none').mean(dim=1)
            loss = loss + 0.1 * dist_a.mean()  # Add a weight coefficient

        
        outputs = outputs_normal or outputs_anomaly
        return (loss, outputs) if return_outputs else loss
    
    def set_eval_setting(self, n_permutations):
        self.n_permutations = n_permutations
         
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        # Propagate permutation-control settings to the eval dataset (if supported).
        graph_based_rank = getattr(self, "graph_based_rank", "no")
        sorted_set = getattr(self, "sorted_set", None)
        if hasattr(eval_dataset, "set_graph_based_rank") and callable(getattr(eval_dataset, "set_graph_based_rank")):
            eval_dataset.set_graph_based_rank(graph_based_rank=graph_based_rank, sorted_set=sorted_set)
        else:
            setattr(eval_dataset, "graph_based_rank", graph_based_rank)
            setattr(eval_dataset, "sorted_set", sorted_set)

        # Simple debug: print sorted_set and one eval sample's column names (only from rank 0)
        try:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        except Exception:
            local_rank = 0
        if local_rank == 0:
            print("DEBUG: sorted_set=", sorted_set)
            try:
                print("DEBUG: eval column names=", eval_dataset.get_column_names())
            except Exception:
                print("DEBUG: could not get eval column names")
            try:
                sample = eval_dataset[0]
                print("DEBUG: one eval sample col_indices=", sample.get('col_indices'))
                # Print the left-to-right column order as it appears in the constructed text
                try:
                    col_indices = sample.get('col_indices')
                    if col_indices is not None:
                        col_names = eval_dataset.get_column_names()
                        col_order = [col_names[i] for i in col_indices]
                        print("DEBUG: eval sample col order (left->right)=", col_order)
                except Exception:
                    print("DEBUG: could not map col_indices to names")

                # Try to decode the assembled input_ids to human text (truncated)
                try:
                    decoded = eval_dataset.tokenizer.decode(sample.get('input_ids', []), skip_special_tokens=False)
                    print("DEBUG: decoded sample text (truncated)=", decoded[:1000])
                except Exception:
                    print("DEBUG: could not decode sample input_ids")

                # If sorted_set exists, show how each token-list would map to column indices/names
                try:
                    if isinstance(sorted_set, list) and len(sorted_set) > 0 and hasattr(eval_dataset, '_resolve_sorted_set_indices'):
                        for si, tokens in enumerate(sorted_set[:5]):
                            try:
                                resolved = eval_dataset._resolve_sorted_set_indices(tokens, eval_dataset.get_column_names())
                                resolved_names = [eval_dataset.get_column_names()[i] for i in resolved]
                                print(f"DEBUG: sorted_set[{si}] tokens={tokens} -> indices={resolved} -> names={resolved_names}")
                            except Exception as e:
                                print(f"DEBUG: could not resolve sorted_set[{si}]: {e}")
                except Exception:
                    print("DEBUG: could not inspect sorted_set mappings")
            except Exception:
                print("DEBUG: could not get one eval sample")
		# do not use distributed sampler
        dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=self.args.eval_batch_size, shuffle = False, 
												collate_fn = AnoLLMDataCollator(self.tokenizer, is_eval=True),)

		
        perplexities = np.zeros((len(eval_dataset), self.n_permutations))
        weighted_perplexities = np.zeros((len(eval_dataset), self.n_permutations)) if self.weights_map else None
        eval_losses = np.zeros((len(eval_dataset), self.n_permutations))

        loss_fct = CrossEntropyLoss(reduction="none")
		
		# for conditional columns
        comma_id =  eval_dataset.tokenizer.convert_tokens_to_ids(',')
        n_col = eval_dataset.get_n_columns()
        column_names = eval_dataset.get_column_names()

        for perm_idx in range(self.n_permutations):
            start_idx = 0
			# Fix a single order for this permutation (graph-based or random).
            if hasattr(eval_dataset, "shuffle_column_order_with_perm") and callable(getattr(eval_dataset, "shuffle_column_order_with_perm")):
                eval_dataset.shuffle_column_order_with_perm(perm_idx=perm_idx)
            else:
                eval_dataset.shuffle_column_order()
            for data in dataloader:
                encoded_batch = data["input_ids"].to(self.model.device)
                attn_mask = data["attention_mask"].to(self.model.device)
                end_idx = start_idx + len(encoded_batch)
                labels = encoded_batch 
				
                start_pos_batch = data["feature_value_start"]
                end_pos_batch = data["feature_value_end"]
                col_indices_batch = data["col_indices"]

                with torch.no_grad():
                    out_logits = self.model(encoded_batch, attention_mask=attn_mask).logits
                shift_logits = out_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_attention_mask_batch = attn_mask[..., 1:].contiguous()
                eval_loss_batch = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1) / shift_attention_mask_batch.sum(1)
				
                if len(eval_dataset.textual_columns) > 0 or self.weights_map:
                    perplexity_batch = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).cpu().numpy() # batch * (ori_seq_len -1)
                    
                    for i in range(len(encoded_batch)):
                        perplexity_single = 0
                        weighted_perplexity_single = 0
                        for j in range(n_col): 
                            start_pos = start_pos_batch[i][j]
                            end_pos = end_pos_batch[i][j]
                            col_idx = col_indices_batch[i][j]
                            col_name = column_names[col_idx]
                            
                            col_score = perplexity_batch[i, start_pos:end_pos].sum()
                            
                            # Normal perplexity calculation
                            if len(eval_dataset.textual_columns) > 0 and col_name in eval_dataset.textual_columns:
                                col_score_norm = col_score / (end_pos - start_pos)
                            else:
                                col_score_norm = col_score
                                
                            perplexity_single += col_score_norm
                            
                            if np.isnan(perplexity_single):
                                print(start_pos, end_pos, col_score)
                                print(col_score / (end_pos - start_pos))
                                print(perplexity_single)

                            # Weighted perplexity calculation
                            if self.weights_map:
                                w_val = float(self.weights_map.get(col_name, 0))
                                wrapper_mult = w_val + 1.0
                                # Follow logic in decision_function: apply weight to the normalized score
                                weighted_perplexity_single += col_score_norm * wrapper_mult

                        perplexities[start_idx+i, perm_idx] = perplexity_single
                        if weighted_perplexities is not None:
                            weighted_perplexities[start_idx+i, perm_idx] = weighted_perplexity_single
                else:
                    perplexity_batch = (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1) 
                    perplexities[start_idx:end_idx, perm_idx] = perplexity_batch.cpu().numpy()
				
                eval_losses[start_idx:end_idx, perm_idx] = eval_loss_batch.cpu().numpy()
                start_idx = end_idx

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if dist.is_initialized():
             world_size = dist.get_world_size() 
        else:
             world_size = 1
		
        all_perplexity = [None for _ in range(world_size)]
        if dist.is_initialized():
             dist.all_gather_object(all_perplexity, perplexities)
             perplexities = np.concatenate(all_perplexity, axis = 1)
		
        weighted_auc_roc = 0.0
        if weighted_perplexities is not None:
            all_weighted_perplexity = [None for _ in range(world_size)]
            if dist.is_initialized():
                dist.all_gather_object(all_weighted_perplexity, weighted_perplexities)
                weighted_perplexities = np.concatenate(all_weighted_perplexity, axis = 1)
		
        all_eval_loss = [None for _ in range(world_size)]
        if dist.is_initialized():
            dist.all_gather_object(all_eval_loss, eval_losses)
            eval_losses = np.concatenate(all_eval_loss, axis = 1)

		
        labels = eval_dataset.anomaly_labels
		
        mean_perplexity = np.mean(perplexities)
        normal_indices = np.where(labels == 0)[0]
        anomaly_indices = np.where(labels == 1)[0]
        perplexity_normal = np.mean(perplexities[normal_indices])
        eval_loss_normal = np.mean(eval_losses[normal_indices])
        perplexity_anomaly = np.mean(perplexities[anomaly_indices])
        eval_loss_anomaly = np.mean(eval_losses[anomaly_indices])

		#print("is nan:", np.isnan(eval_dataset.anomaly_labels).sum(), np.isnan(perplexities).sum())
        auc_roc = metrics.roc_auc_score(eval_dataset.anomaly_labels, np.mean(perplexities, axis = 1))
        
        metric = {"eval_loss": np.mean(eval_losses), "eval_perplexity": mean_perplexity, "eval_auc_roc": auc_roc, \
						"eval_loss_normal": eval_loss_normal, "eval_perplexity_normal": perplexity_normal,
						"eval_loss_anomaly": eval_loss_anomaly, "eval_perplexity_anomaly": perplexity_anomaly}

        if weighted_perplexities is not None:
            weighted_mean_perplexities = np.mean(weighted_perplexities, axis=1)
            weighted_auc_roc = metrics.roc_auc_score(eval_dataset.anomaly_labels, weighted_mean_perplexities)
            metric["eval_weighted_auc_roc"] = weighted_auc_roc
            metric["eval_weighted_perplexity_normal"] = np.mean(weighted_perplexities[normal_indices])
            metric["eval_weighted_perplexity_anomaly"] = np.mean(weighted_perplexities[anomaly_indices])
		
        if local_rank == 0:
            self.log(metric)
            self._memory_tracker.stop_and_update_metrics(metric)

            step = int(getattr(self.state, "global_step", 0) or 0)
            epoch = getattr(self.state, "epoch", 0.0)
            if epoch is None: epoch = 0.0
            
            w_auc_str = f"{weighted_auc_roc:.6f}" if weighted_perplexities is not None else "N/A"
            w_ppl_norm_str = f"{metric.get('eval_weighted_perplexity_normal', 0.0):.6f}" if weighted_perplexities is not None else "N/A"
            w_ppl_anom_str = f"{metric.get('eval_weighted_perplexity_anomaly', 0.0):.6f}" if weighted_perplexities is not None else "N/A"
            
            with open(self.eval_log_file, "a") as f:
                f.write(
                    f"{step},{epoch:.2f},"
                    f"{metric['eval_auc_roc']:.6f},{metric['eval_loss']:.6f},{metric['eval_perplexity']:.6f},"
                    f"{metric['eval_loss_normal']:.6f},{metric['eval_loss_anomaly']:.6f},"
                    f"{metric['eval_perplexity_normal']:.6f},{metric['eval_perplexity_anomaly']:.6f}," 
                    f"{w_auc_str},{w_ppl_norm_str},{w_ppl_anom_str}\n" 
                ) # Note: CSV format changed, header mismatch if not updated. But keeps it simple.
            print("==============TEST PERFORMANCE=================")
            if weighted_perplexities is not None:
                print(f"weighted_auc_roc: {weighted_auc_roc}")

        return metric

def _seed_worker(_):
	"""
	Helper function to set worker seed during Dataloader initialization.
	"""
	worker_seed = torch.initial_seed() % 2**32
	random.seed(worker_seed)
	np.random.seed(worker_seed)
	torch.manual_seed(worker_seed)
	torch.cuda.manual_seed_all(worker_seed)

'''
	Overwrites the get_train_dataloader methode of the HuggingFace Trainer to not remove the "unused" columns -
	they are needed later!

	def get_train_dataloader(self) -> DataLoader:
		if self.train_dataset is None:
			raise ValueError("Trainer: training requires a train_dataset.")

		data_collator = self.data_collator
		train_dataset = (
			self.train_dataset
		)  # self._remove_unused_columns(self.train_dataset, description="training")
		local_rank = int(os.environ["LOCAL_RANK"])
		world_size = dist.get_world_size()
		train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=False, drop_last=True)

		return DataLoader(
			train_dataset,
			batch_size=self._train_batch_size,
			sampler=train_sampler,
			collate_fn=data_collator,
			drop_last=self.args.dataloader_drop_last,
			num_workers=self.args.dataloader_num_workers,
			pin_memory=self.args.dataloader_pin_memory,
			worker_init_fn=_seed_worker,
		)'''
	

	

	

	


