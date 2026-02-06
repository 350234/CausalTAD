# CausalTAD

This repository contains the code for the CausalTAD paper.

## Environment setup


- Python 3.9+

### Core scientific stack

- numpy
- pandas
- scipy
- scikit-learn

### Causal discovery

- causallearn

### LLM and ML training

- openai
- torch
- transformers
- datasets
- peft
- wandb
- tqdm

### Other utilities and baselines

- pyod
- deepod
- modelscope
- gensim
- inflect
- feature_engine
- ucimlrepo


### One-command environment setup (requirements.txt)

Create a fresh environment and install all dependencies in one go:
```bash
conda create -n causalTAD python=3.10 -y && conda activate causalTAD && pip install -r requirements.txt
```

## Usage overview

1. Run `python CausalTAD/graph_gen/main.py` to generate the graph.
2. Based on the algorithm type used in the previous step, select the corresponding code under `CausalTAD/use_graph` to perform graph-based ranking and weight generation.
3. In `CausalTAD/AnoLLM`, use the results from the previous step to inject causal knowledge into the language model for training and inference.



