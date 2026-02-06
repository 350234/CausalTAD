import os
from pathlib import Path
import argparse
import typing as tp

import numpy as np
import torch
import time
import json
import pickle as pkl

import torch.distributed as dist

from anollm import AnoLLM
from src.data_utils import DATA_MAP, get_text_columns, get_max_length_dict,my_load_data
from train_anollm import get_run_name


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", type = str, default='wine', choices = [d.lower() for d in DATA_MAP.keys()],
					help="Name of datasets in the ODDS benchmark")
	parser.add_argument("--exp_dir", type = str, default=None)
	parser.add_argument("--setting", type = str, default='semi_supervised', choices = ['semi_supervised', 'unsupervised'], help="semi_supervised:an uncontaminated, unsupervised setting; unsupervised:a contaminated, unsupervised setting")
	
	#dataset hyperparameters
	parser.add_argument("--data_dir", type = str, default='data')
	parser.add_argument("--n_splits", type = int, default=5)
	parser.add_argument("--split_idx", type = int, default=None) # 0 to n_split-1
	# binning
	parser.add_argument("--binning", type = str, choices=['quantile', 'equal_width', 'language', 'none', 'standard'], default='standard')
	parser.add_argument("--n_buckets", type = int, default=10)
	parser.add_argument("--remove_feature_name", action = 'store_true')
	
	# model hyperparameters (for getting the model name)
	parser.add_argument("--model", type = str, choices = ['gpt2', 'distilgpt2', 'smol', 'smol-360', 'smol-1.7b'], default='smol')
	parser.add_argument("--lora", action='store_true', default=False)
	parser.add_argument("--lr", type = float, default=5e-5)
	parser.add_argument("--random_init", action='store_true', default=False)
	parser.add_argument("--no_random_permutation", action='store_true', default=False)
	
	#testing
	parser.add_argument("--max_steps", type = int, default=2000)
	parser.add_argument("--batch_size", type = int, default=128) # per gpu
	parser.add_argument("--n_permutations", type = int, default=100) # per gpu

	parser.add_argument("--weights", type = int, choices=[0,1], default=0, help="whether to use column weights (0 or 1)")
	parser.add_argument("--weights_path", type = str, default=None)
	parser.add_argument("--augmentation", type = str, default='no') # per gpu
	parser.add_argument("--abnormal_ratio", type = float, default=0.5)
	parser.add_argument("--graph_based_rank", type = str, default='no', choices=['yes', 'no'])
	parser.add_argument("--score_weight", type = str, default='no', choices=['yes', 'no'])
	parser.add_argument(
		"--sorted_set_path",
		type=str,
		default=None,
		help="When --graph_based_rank yes, load sorted_set (2D list) from a .json or .pkl file.",
	)
	parser.add_argument("--train_cos", type = bool, default=False)
	parser.add_argument("--eval_steps", type = int, default = 1000)
	args = parser.parse_args()
	
	if args.model == 'smol':
		args.model = 'models/SmolLM-135M'
	elif args.model == 'smol-360':
		args.model = 'models/SmolLM-360M'
	elif args.model == 'smol-1.7b':	
		args.model = 'models/SmolLM-1.7B'
	
	return args


def _load_sorted_set(path: Path):
	if not path.exists():
		raise FileNotFoundError(f"sorted_set_path not found: {path}")
	suffix = path.suffix.lower()
	if suffix == ".json":
		with open(path, "r", encoding="utf-8") as f:
			obj = json.load(f)
	elif suffix == ".pkl":
		with open(path, "rb") as f:
			obj = pkl.load(f)
	else:
		raise ValueError("sorted_set_path must end with .json or .pkl")
	if not isinstance(obj, list):
		raise ValueError("sorted_set must be a list or list of lists")
	# If it's a 1D list (elements are not lists), wrap into a 2D list
	if len(obj) > 0 and not isinstance(obj[0], list):
		obj = [obj]
	else:
		# ensure inner elements are lists
		if len(obj) > 0 and any(not isinstance(i, list) for i in obj):
			raise ValueError("sorted_set must be a 2D list (list of lists)")
	return obj


def _split_contiguous(total: int, world_size: int, rank: int) -> tp.Tuple[int, int]:
	"""Return [start, end) indices for rank's shard."""
	base = total // world_size
	rem = total % world_size
	start = rank * base + min(rank, rem)
	end = start + base + (1 if rank < rem else 0)
	return start, end

def main():
	# Set CUDA devices for each process
	local_rank = int(os.environ["LOCAL_RANK"])
	world_size = dist.get_world_size()
	rank = dist.get_rank()
	torch.cuda.set_device(local_rank)
	
	args = get_args()
	if args.exp_dir is not None:
		args.exp_dir = Path(args.exp_dir)

	sorted_set = None
	if args.graph_based_rank == "yes":
		if args.sorted_set_path is None:
			raise ValueError("--sorted_set_path is required when --graph_based_rank yes")
		sorted_set = _load_sorted_set(Path(args.sorted_set_path))
		total_perms = len(sorted_set)
	else:
		total_perms = args.n_permutations

	if args.exp_dir is None:
		args.exp_dir = Path('exp') / args.dataset / args.setting / "split{}".format(args.n_splits) / "split{}".format(args.split_idx)
	
	if not os.path.exists(args.exp_dir):
		raise ValueError("Experiment directory {} does not exist".format(args.exp_dir))
		
	score_dir = args.exp_dir / 'scores'
	run_name = get_run_name(args)

	score_path = score_dir / "{}.npy".format(run_name)
	print("score_path:",  score_path)	
	os.makedirs(score_dir, exist_ok = True)

	start_perm, end_perm = _split_contiguous(total_perms, world_size, rank)
	local_n_perm = end_perm - start_perm
	
	X_train, X_test, y_train, y_test = my_load_data(args)
	
	if not os.path.exists(score_path):
		model_dir = args.exp_dir / 'models'
		model_path = model_dir / '{}.pt'.format(run_name)
		
		efficient_finetuning = 'lora' if args.lora else ''
		max_length_dict = get_max_length_dict(args.dataset)
		text_columns = get_text_columns(args.dataset)
		model = AnoLLM(args.model,
						efficient_finetuning = efficient_finetuning,
						model_path = str(model_path),
						max_length_dict=max_length_dict, 
						textual_columns = text_columns,
						no_random_permutation=args.no_random_permutation,
						bp16=True,
				)
		print(text_columns, max_length_dict)
		
		model.load_from_state_dict(str(model_path))
		model.model.to(local_rank)  
			
		# Move the model to the appropriate GPU
		# Wrap the model for distributed training
		model.model = torch.nn.parallel.DistributedDataParallel(
			model.model, device_ids=[local_rank], output_device=local_rank
		)
		start_time = time.time()	
		if local_n_perm == 0:
			# keep shape consistent for gather/concat
			scores = np.zeros((len(X_test), 0), dtype=np.float32)
		else:
			scores = model.decision_function(
				X_test,
				n_permutations=local_n_perm,
				batch_size=args.batch_size,
				device="cuda",
				weights = args.weights,
				weights_path = args.weights_path,
				graph_based_rank=args.graph_based_rank,
				sorted_set=(sorted_set[start_perm:end_perm] if sorted_set is not None else None),
			)
		end_time = time.time()

		# Avoid dist.all_gather_object: it serializes to a byte tensor on GPU (NCCL)
		# and can OOM for large arrays. Instead, write each rank's shard to disk and
		# let rank0 merge via a .npy memmap.
		part_path = score_dir / f"partial_{run_name}.rank{rank}.npy"
		np.save(part_path, scores)
		dist.barrier()

		if rank == 0:
			print("Inference time:", end_time - start_time)
			
			run_time_dir = args.exp_dir / "run_time" / "test"
			os.makedirs(run_time_dir, exist_ok = True)
			run_time_path = run_time_dir / "{}.txt".format(run_name)
			with open(run_time_path, 'w') as f:
				f.write(str(end_time - start_time))

			n_test = len(X_test)
			raw_score_path = score_dir / "raw_{}.npy".format(run_name)
			raw_mm = np.lib.format.open_memmap(
				raw_score_path,
				mode="w+",
				dtype=np.float32,
				shape=(n_test, total_perms),
			)

			sum_scores = np.zeros((n_test,), dtype=np.float64)
			total_count = 0

			for r in range(world_size):
				r_start, r_end = _split_contiguous(total_perms, world_size, r)
				r_part_path = score_dir / f"partial_{run_name}.rank{r}.npy"
				if not r_part_path.exists():
					raise FileNotFoundError(f"Missing partial scores from rank {r}: {r_part_path}")
				part = np.load(r_part_path)
				if part.shape != (n_test, r_end - r_start):
					raise ValueError(
						f"Partial scores shape mismatch for rank {r}: got {part.shape}, expected {(n_test, r_end - r_start)}"
					)
				raw_mm[:, r_start:r_end] = part.astype(np.float32, copy=False)
				if (r_end - r_start) > 0:
					sum_scores += part.sum(axis=1, dtype=np.float64)
					total_count += (r_end - r_start)
				try:
					os.remove(r_part_path)
				except OSError:
					pass

			if total_count == 0:
				raise ValueError("total_perms is 0; cannot compute mean scores.")
			mean_scores = (sum_scores / float(total_count)).astype(np.float32)
			np.save(score_path, mean_scores)
			# Ensure memmap flush
			raw_mm.flush()
		dist.barrier()
	
	dist.destroy_process_group()
	
if __name__ == '__main__':
	dist.init_process_group(backend="nccl") 
	main()

	