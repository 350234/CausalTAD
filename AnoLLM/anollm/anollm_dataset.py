import random
import typing as tp
import os 

from datasets import Dataset
from dataclasses import dataclass
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle as pkl
MAX_COL_LENGTH = 128

class AnoLLMDataset(Dataset):
	"""AnoLLM Dataset

	The AnoLLM overwrites the _getitem function of the HuggingFace Dataset Class to include the permutation step.

	Attributes:
		tokenizer (AutoTokenizer): Tokenizer from HuggingFace
	"""

	def set_tokenizer(self, tokenizer):
		"""Set the Tokenizer

		Args:
			tokenizer: Tokenizer from HuggingFace
		"""
		self.tokenizer = tokenizer 
	
	def set_anomaly_label(self, labels):
		assert len(labels) == len(self._data)
		self.anomaly_labels = labels

	def set_textual_columns(self, columns: tp.List[str]):
		col_list = self.get_column_names()
		for col in columns:
			if col not in col_list:
				raise ValueError("Column {} not in the dataset.".format(col))
		self.textual_columns = columns

	def set_graph_based_rank(
		self,
		graph_based_rank: str = "no",
		sorted_set: tp.Optional[tp.List[tp.List[str]]] = None,
	) -> None:
		"""Control how column permutation order is sampled.

		If graph_based_rank == 'yes', the permutation order is sampled from a
		deterministic candidate set (sorted_set) instead of all d! permutations.
		"""
		self.graph_based_rank = graph_based_rank
		self.sorted_set = sorted_set

	def _resolve_sorted_set_indices(
		self,
		order_tokens: tp.List[str],
		column_names: tp.List[str],
	) -> tp.List[int]:
		used: tp.Set[int] = set()
		indices: tp.List[int] = []
		for token in order_tokens:
			needle = str(token).strip().casefold()
			if not needle:
				continue
			match_idx = None
			for i, name in enumerate(column_names):
				if i in used:
					continue
				name_cf = str(name).casefold()
				template_cf = f" {name} is ".casefold()
				if needle in name_cf or needle in template_cf or name_cf in needle:
					match_idx = i
					break
			if match_idx is None:
				raise KeyError(
					f"sorted_set token '{token}' cannot be matched to any column name. "
					f"Available columns: {column_names}"
				)
			used.add(match_idx)
			indices.append(match_idx)

		# Append remaining columns (deterministic) so every row still contains all features.
		for i in range(len(column_names)):
			if i not in used:
				indices.append(i)
		return indices
	
	def get_n_columns(self):
		row = self._data.fast_slice(0, 1)
		return row.num_columns

	def get_column_names(self):
		row = self._data.fast_slice(0, 1)
		return row.column_names
	
	def shuffle_column_order(self):
		return self.shuffle_column_order_with_perm(perm_idx=None)

	def shuffle_column_order_with_perm(self, perm_idx: tp.Optional[int] = None):
		"""Used in evaluation.

		Pick a single column order and then keep it fixed for all rows.
		If graph_based_rank == 'yes', choose the order from sorted_set.
		Otherwise, use a random permutation.
		"""
		row = self._data.fast_slice(0, 1)
		column_names = row.column_names

		if getattr(self, "graph_based_rank", "no") == "yes":
			sorted_set = getattr(self, "sorted_set", None)
			if not sorted_set or not isinstance(sorted_set, list):
				raise ValueError(
					"graph_based_rank=='yes' requires sorted_set (2D list of token lists)."
				)
			if perm_idx is None:
				order_tokens = random.choice(sorted_set)
			else:
				order_tokens = sorted_set[int(perm_idx) % len(sorted_set)]
			if not isinstance(order_tokens, list):
				raise ValueError("sorted_set must be a 2D list (list of lists).")
			self.shuffle_idx = self._resolve_sorted_set_indices(order_tokens, column_names)
		else:
			self.shuffle_idx = list(range(row.num_columns))
			random.shuffle(self.shuffle_idx)
		return self.shuffle_idx
	
	def fix_column_order(self):
		# set the column order to be default column order. Do not shuffle the columns.
		row = self._data.fast_slice(0, 1)
		self.shuffle_idx = list(range(row.num_columns))
	
	def prepare(
		self,
		is_eval: bool = True, 
		max_length_dict: tp.Optional[tp.Dict[str, int]] = {},
		data_path = None,
		thesupervised = None,
		):
		'''
		Preprocess the data by tokenizing each column and truncating the columns to max_length
		Inputs:
		max_length_dict specifies the maximum length of each column. If None, all columns are truncated to max length
		pad_columns specifies whether to pad the columns to the same length according to max_length of a each column
		'''

		self.is_eval = is_eval
		
		column_names = self.get_column_names()
		print("column_names:",column_names)
		self.processed_data = [] 
		self.tokenized_feature_names = []
		bos_token_id = self.tokenizer.bos_token_id

		if self.is_eval == True:
			thesupervised = "eval"

		# Take the last column
		last_col_name = column_names[-1]
		raw_labels = self._data[last_col_name].to_pylist()
		if self.is_eval == False:
			print("raw_labels:",raw_labels)
		
		# Check label column name
		label_col = []
		label_col.append("fraud found")
		label_col.append("fraudulent")
		label_col.append("class")

		if last_col_name not in label_col:
			if self.is_eval == False:
				raise KeyError(f"Last column should be one of {label_col}, but it is not")
			
		my_column_names = self.get_column_names()
		
		if self.is_eval == False:
			self._data = self._data.drop(last_col_name)
			my_last_col_name = my_column_names[-1]
			if my_last_col_name not in label_col:
				print(f"Last column {label_col} has been removed")
				

		n_col = self.get_n_columns()

		print(thesupervised)
		
		# Unsupervised label encoding
		if thesupervised == "unsupervised":
			if last_col_name == "fraud found":
				self.labels = [0 if str(x).strip().lower() == "yes" else 0 for x in raw_labels]
				unique = set(self.labels)
				assert unique == {0}, f"Labels must be exactly {0}, got {sorted(unique)}"
			elif last_col_name == "fraudulent":
				self.labels = [0 if str(x).strip().lower() == "yes" else 0 for x in raw_labels]
				unique = set(self.labels)
				assert unique == {0}, f"Labels must be exactly {0}, got {sorted(unique)}"
			elif last_col_name == "class":
				self.labels = [0 if str(x).strip().lower() == "yes" else 0 for x in raw_labels]
				unique = set(self.labels)
				assert unique == {0}, f"Labels must be exactly {0}, got {sorted(unique)}"

		# Semi-supervised label encoding
		if thesupervised == "semi_supervised":
			if last_col_name == "fraud found":
				self.labels = [1 if str(x).strip().lower() == "yes" else 0 for x in raw_labels]
				unique = set(self.labels)
				assert unique == {0, 1}, f"Labels must be exactly {0, 1}, got {sorted(unique)}"
			elif last_col_name == "fraudulent":
				self.labels = [1 if str(x).strip().lower() == "yes" else 0 for x in raw_labels]
				unique = set(self.labels)
				assert unique == {0, 1}, f"Labels must be exactly {0, 1}, got {sorted(unique)}"
			elif last_col_name == "class":
				self.labels = [1 if str(x).strip().lower() == "yes" else 0 for x in raw_labels]
				unique = set(self.labels)
				assert unique == {0, 1}, f"Labels must be exactly {0, 1}, got {sorted(unique)}"

		for col_idx in range(n_col):
			feature_names = ' ' + my_column_names[col_idx] + ' '
			tokenized_feature_names = self.tokenizer(feature_names)
			tokenized_is = self.tokenizer('is ')
			if bos_token_id and tokenized_feature_names['input_ids'][0] == bos_token_id:
				tokenized_feature_names['input_ids'] = tokenized_feature_names['input_ids'][1:]
				tokenized_is['input_ids'] = tokenized_is['input_ids'][1:]

			self.tokenized_feature_names.append(tokenized_feature_names["input_ids"] + tokenized_is["input_ids"])
		
		if data_path is not None and os.path.exists(data_path):
			print("Data was processed before and saved at:", data_path)
			self.processed_data = pkl.load(open(data_path, 'rb'))
			# Quick sanity check: every row should have n_col entries
			for i, r in enumerate(self.processed_data):
				if len(r) != n_col:
					print(f"DEBUG_PROCESSED_MISMATCH: idx={i} len={len(r)} expected={n_col} sample_cols={min(5, len(r))}")
			assert all(len(r) == n_col for r in self.processed_data), "Processed_data row length mismatch"
		else:
			for key in tqdm(range(len(self._data))):
				row = self._data.fast_slice(key, 1)
				tokenized_texts = []
				for col_idx in range(n_col):
					feature_values = str(row.columns[col_idx].to_pylist()[0]).strip()
					if len(feature_values) == 0:
						feature_values = "None"
					data = self.tokenizer(feature_values)
					if bos_token_id and data['input_ids'][0] == bos_token_id:
						data['input_ids'] = data['input_ids'][1:]

					tokenized_texts.append(data["input_ids"])
					if len(data["input_ids"]) == 0:
						print("Warning: tokenized text is empty.", my_column_names[col_idx],len( feature_values),feature_values)
				self.processed_data.append(tokenized_texts)
			
			# truncate the columns that are too long	
			for col_idx in range(n_col):
				name = my_column_names[col_idx]
				if name not in max_length_dict:
					max_length = MAX_COL_LENGTH
				else:
					max_length = max_length_dict[name]
				assert isinstance(max_length, int)
				
				for data_idx in range(len(self.processed_data)):
					length = len(self.processed_data[data_idx][col_idx]) + len(self.tokenized_feature_names[col_idx])
					if length >= max_length:
						self.processed_data[data_idx][col_idx] = self.processed_data[data_idx][col_idx][:max_length - len(self.tokenized_feature_names[col_idx])]
			if data_path is not None:
				pkl.dump(self.processed_data, open(data_path, 'wb'))
			# Post-build sanity check (short and easy to remove)
			for i, r in enumerate(self.processed_data):
				if len(r) != n_col:
					print(f"DEBUG_PROCESSED_MISMATCH_AFTER_BUILD: idx={i} len={len(r)} expected={n_col} sample_cols={min(5, len(r))}")
			assert all(len(r) == n_col for r in self.processed_data), "Processed_data row length mismatch after build"
		print("Data processing completed and saved at:", data_path)

	def _getitem(
		self, 
		key: tp.Union[int, slice, str], 
		decoded: bool = True, 
		**kwargs
	) -> tp.Union[tp.Dict, tp.List]:
		"""
		Get one instance of the tabular data, permuted, converted to text and tokenized.
		"""
		row = self._data.fast_slice(key, 1)
		

		# get shuffle_idx
		if "shuffle_idx" in self.__dict__:
			# evaluation-time fixed shuffle
			shuffle_idx = self.shuffle_idx
		else:
			# training-time shuffle
			if getattr(self, "graph_based_rank", "no") == "yes":
				sorted_set = getattr(self, "sorted_set", None)
				if not sorted_set or not isinstance(sorted_set, list):
					raise ValueError(
						"graph_based_rank=='yes' requires sorted_set (2D list of token lists)."
					)
				order_tokens = random.choice(sorted_set)
				if not isinstance(order_tokens, list):
					raise ValueError("sorted_set must be a 2D list (list of lists).")
				shuffle_idx = self._resolve_sorted_set_indices(order_tokens, row.column_names)
			else:
				shuffle_idx = list(range(row.num_columns))
				random.shuffle(shuffle_idx)
		
		# get tokenized text
		comma_id =  self.tokenizer.convert_tokens_to_ids(',')
		eos_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)
		bos_token_id = self.tokenizer.bos_token_id
		if self.is_eval:
			tokenized_text = {"input_ids": [], "attention_mask": [], "feature_value_start":[],
							"feature_value_end":[],'col_indices':shuffle_idx}
		else:
			tokenized_text = {"input_ids": [], "attention_mask": []}
		if bos_token_id:
			tokenized_text["input_ids"] = [bos_token_id]

		if hasattr(self, "processed_data"):
			start_idx = 0
			for idx, col_idx in enumerate(shuffle_idx):
				tokenized_feature_names = self.tokenized_feature_names[col_idx]
				# Quick boundary check to help locate index errors (short and removable)
				if key >= len(self.processed_data) or key < 0:
					print(f"DEBUG_KEY_OOB: key={key} processed_len={len(self.processed_data)}")
				if col_idx >= len(self.processed_data[key]):
					print(f"DEBUG_INDEX_ERROR: key={key} col_idx={col_idx} len_row={len(self.processed_data[key])} shuffle_idx={shuffle_idx} row_columns={row.column_names}")
					raise IndexError(f"col_idx out of range: key={key} col_idx={col_idx} len_row={len(self.processed_data[key])}")
				tokenized_feature_values = self.processed_data[key][col_idx]
				tokenized_col = tokenized_feature_names + tokenized_feature_values 
				if idx == len(shuffle_idx) - 1:
					tokenized_text["input_ids"] += tokenized_col + [eos_id]
				else:
					tokenized_text["input_ids"] += tokenized_col + [comma_id]
				if self.is_eval:
					tokenized_text["feature_value_start"].append(start_idx + len(tokenized_feature_names) -1 )
					tokenized_text["feature_value_end"].append(start_idx + len(tokenized_col) )
				start_idx += len(tokenized_col) + 1
		else:
			raise ValueError("processed_data is not found. Please run prepare function first.")	
		tokenized_text["attention_mask"] += [1] * len(tokenized_text["input_ids"])
		if self.is_eval == False:
			tokenized_text["label"] = self.labels[key]
		return tokenized_text
	
	def get_item_test(self, key):
		row = self._data.fast_slice(key, 1)
		shuffle_idx = list(range(row.num_columns))
		random.shuffle(shuffle_idx)
		
		shuffled_text = ",".join(
			[
				" %s is %s "
				% (row.column_names[i], str(row.columns[i].to_pylist()[0]).strip() )
				for i in shuffle_idx
			]
		)
		tokenized_text = self.tokenizer(shuffled_text, padding=True)

		return shuffled_text, tokenized_text 
	
	def __getitems__(self, keys: tp.Union[int, slice, str, list]):
		if isinstance(keys, list):
			return [self._getitem(key) for key in keys]
		else:
			return self._getitem(keys)

	#def add_gaussian_noise(self, value):
#		return value + np.random.normal(0, 0.1)

@dataclass
class AnoLLMDataCollator(DataCollatorWithPadding):
	def __init__(self, tokenizer, is_eval=False, **kwargs):
		# Call parent initializer first
		super().__init__(tokenizer=tokenizer, **kwargs)
		# Then set own parameters
		self.is_eval = is_eval
	def __call__(self, features: tp.List[tp.Dict[str, tp.Any]]):
		# Eval: do not split into normal/anomaly batches; do not add LM training labels.
		# Just pad and return a single batch.
		if self.is_eval:
			# Eval features already do not contain labels — just pad and return.
			return self.tokenizer.pad(
				features,
				padding=self.padding,
				max_length=self.max_length,
				pad_to_multiple_of=self.pad_to_multiple_of,
				return_tensors=self.return_tensors,
			)

		print("DEBUG: labels in this batch =", [f["label"] for f in features])
		print("DEBUG: first feature keys =", list(features[0].keys()))
		print("DEBUG: all labels == 1:", [f["label"] for f in features if f.get("label") == 1])

		normal_feats = [f for f in features if f["label"] == 0]
		anomaly_feats = [f for f in features if f["label"] == 1]
		
		normal_batch = self.tokenizer.pad(
            normal_feats,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        ) if normal_feats else None
		
		anomaly_batch = self.tokenizer.pad(
            anomaly_feats,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        ) if anomaly_feats else None
		
		if normal_batch is not None:
			normal_batch["labels"] = normal_batch["input_ids"].clone()
		if anomaly_batch is not None:
			anomaly_batch["labels"] = anomaly_batch["input_ids"].clone()
		if normal_batch is None:
			print("[N] normal_batch is None")
		if anomaly_batch is None:
			print("[N] anomaly_batch is None")
		
		return normal_batch, anomaly_batch


class AnoLLMDataLoader(DataLoader):
	'''
	Add set_epoch function so that huggingface trainer can call it 
	'''
	def set_epoch(self, epoch):
		if hasattr(self.sampler, "set_epoch"):
			self.sampler.set_epoch(epoch)
			print("Set epoch", epoch)



