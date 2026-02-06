import os
from pathlib import Path
import argparse

from sklearn import metrics
import numpy as np
import pandas as pd
from data_utils import load_data, DATA_MAP


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type = str, default='wine', choices = [d.lower() for d in DATA_MAP.keys()],
                    help="Name of datasets in the ODDS benchmark")
    parser.add_argument("--exp_dir", type = str, default=None)
    parser.add_argument("--setting", type = str, default='semi_supervised', choices = ['semi_supervised', 'unsupervised'])
    
    #dataset hyperparameters
    parser.add_argument("--data_dir", type = str, default='data')
    parser.add_argument("--n_splits", type = int, default=5)
    parser.add_argument("--split_idx", type = int, default=None) # 0 to n_split-1

    args = parser.parse_args()
    
    return args

def tabular_metrics(y_true, y_score):
    """
    Calculates evaluation metrics for tabular anomaly detection and returns predicted anomaly indices.
    
    Args:
        y_true (np.array): Data label, 0=normal, 1=anomaly
        y_score (np.array): Predicted anomaly scores (higher=more anomalous)
        
    Returns:
        tuple: (auc_roc, auc_pr, f1, p, r, anomaly_indices)
    """
    # Preserve original indices
    original_indices = np.arange(len(y_true))
    
    # Shuffle to avoid ordering bias
    new_index = np.random.permutation(len(y_true))
    y_true = y_true[new_index]
    y_score = y_score[new_index]
    original_indices = original_indices[new_index]
    
    # Determine number of anomalies
    top_k = len(np.where(y_true == 1)[0])
    
    # Select top_k highest scores as predicted anomalies
    indices = np.argpartition(y_score, -top_k)[-top_k:]
    y_pred = np.zeros_like(y_true)
    y_pred[indices] = 1
    
    # Map back to original dataset indices
    anomaly_indices = original_indices[indices]
    
    # Compute metrics
    y_true = y_true.astype(int)
    p, r, f1, _ = metrics.precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    return (
        metrics.roc_auc_score(y_true, y_score),
        metrics.average_precision_score(y_true, y_score),
        f1, p, r,
        anomaly_indices  # Return original indices of predicted anomalies
    )

def get_metrics(args, only_raw=False, only_normalized=False, only_ordinal=False):
    X_train, X_test, y_train, y_test = load_data(args)
    
    # Ensure X_test is a DataFrame
    if isinstance(X_test, np.ndarray):
        # If column names are missing, create default names
        if hasattr(args, 'feature_names') and args.feature_names:
            columns = args.feature_names
        else:
            columns = [f'feature_{i}' for i in range(X_test.shape[1])]
        X_test = pd.DataFrame(X_test, columns=columns)
    
    if isinstance(y_test, pd.Series):
        y_test = np.array(y_test)
    
    if args.exp_dir is None:
        args.exp_dir = Path('exp') / args.dataset / args.setting / "split{}".format(args.n_splits) / "split{}".format(args.split_idx)
    
    score_dir = args.exp_dir / 'scores'
    if not os.path.exists(score_dir):
        raise ValueError("Score directory {} does not exist".format(score_dir))

    # Create a directory to store predicted anomaly samples
    anomaly_data_dir = args.exp_dir / 'predicted_anomalies'
    os.makedirs(anomaly_data_dir, exist_ok=True)

    method_dict = {}
    for score_npy in os.listdir(score_dir):
        if '.npy' in score_npy:
            if score_npy.startswith('raw'):
                continue
            if is_baseline(score_npy) and only_normalized:
                if 'normalized' not in score_npy:
                    continue
            elif is_baseline(score_npy) and only_ordinal:
                if 'ordinal' not in score_npy:
                    continue

            method = '.'.join(score_npy.split('.')[:-1])
            if method == 'rdp':
                continue
            
            score_path = score_dir / score_npy
            scores = np.load(score_path)
            
            if np.isnan(scores).any():
                print("NaNs in scores for {}".format(method))
                method_dict[method] = [0, 0, 0, 0, 0] 
            elif np.isinf(scores).any():
                print("Infs in scores for {}".format(method))
                method_dict[method] = [0, 0, 0, 0, 0] 
            else:
                # Get metrics and predicted anomaly indices
                auc_roc, auc_pr, f1, p, r, anomaly_indices = tabular_metrics(y_test, scores)
                method_dict[method] = [auc_roc, auc_pr, f1, p, r]
                
                # Save predicted anomaly samples
                anomaly_data = X_test.iloc[anomaly_indices].copy()
                
                # Add true labels
                anomaly_data['true_label'] = y_test[anomaly_indices]
                
                # Add anomaly scores
                anomaly_data['anomaly_score'] = scores[anomaly_indices]
                
                # Save to CSV
                anomaly_file = anomaly_data_dir / f"{method}_anomalies.csv"
                anomaly_data.to_csv(anomaly_file, index=False)
                print(f"Saved predicted anomalies for {method} to {anomaly_file}")

    # Build ranking info
    rankings = []
    if method_dict:  # Ensure the dict is not empty
        method = list(method_dict.keys())[0]
        for i in range(len(method_dict[method])):
            scores = [-method_dict[k][i] for k in method_dict.keys()]
            ranking = np.argsort(scores).argsort() + 1
            rankings.append(ranking)
    else:
        rankings = [[] for _ in range(5)]  # If there are no methods, create empty rankings

    print("-"*100)
    for idx, (k, v) in enumerate(method_dict.items()):
        print("{:30s}: AUC-ROC: {:.4f} ({:2d}), AUC-PR: {:.4f} ({:2d}), F1: {:.4f} ({:2d}), P: {:.4f} ({:2d}), R: {:.4f} ({:2d})".format(k, 
            v[0], rankings[0][idx] if rankings[0] else 0,
            v[1], rankings[1][idx] if rankings[1] else 0,
            v[2], rankings[2][idx] if rankings[2] else 0,
            v[3], rankings[3][idx] if rankings[3] else 0,
            v[4], rankings[4][idx] if rankings[4] else 0,
        ))

    return method_dict

def is_baseline(s):
    if 'anollm' in s:
        return False
    return True

def filter_results(d:dict):
    d2 = {}
    for k in d.keys():
        new_key = k
        if is_baseline(k):
            d2[k] = d[k]
        else:
            if '_lora' in k:
                temp = k.replace('_lora', '')
                if temp in d:
                    continue
                else:
                    new_key = new_key.replace('_lora', '')
                    d2[new_key] = d[k]
            else:
                d2[new_key] = d[k]
    return d2

def aggregate_results(m_dicts):
    if not m_dicts:
        print("No results to aggregate")
        return {}, {}

    # Initialize aggregation dict
    all_keys = set()
    for d in m_dicts:
        all_keys.update(d.keys())
    
    aggregate_results = {k: {'AUC-ROC':[], 'AUC-PR': [], 'F1': [], 'P': [], 'R':[]} for k in all_keys}
    
    for d in m_dicts:
        for k in all_keys:
            if k in d:
                aggregate_results[k]['AUC-ROC'].append(d[k][0])
                aggregate_results[k]['AUC-PR'].append(d[k][1])
                aggregate_results[k]['F1'].append(d[k][2])
                aggregate_results[k]['P'].append(d[k][3])
                aggregate_results[k]['R'].append(d[k][4])

    print("-"*100)
    
    # Build ranking info
    rankings = {}
    key = next(iter(aggregate_results.keys()), None)
    if key:
        for metric_name in aggregate_results[key].keys():
            scores = [-np.mean(aggregate_results[k][metric_name]) for k in aggregate_results.keys()] 
            ranking = np.argsort(scores).argsort() + 1
            rankings[metric_name] = ranking

    for idx, k in enumerate(aggregate_results.keys()):
        print("{:30s}: AUC-ROC: {:.4f} +- {:.4f} ({:2d}), AUC-PR: {:.4f} +- {:.4f} ({:2d}), F1: {:.4f} +- {:.4f} ({:2d})  P: {:.4f} +- {:.4f} ({:2d})  R: {:.4f} +- {:.4f} ({:2d})".format(k, 
            np.mean(aggregate_results[k]['AUC-ROC']), np.std(aggregate_results[k]['AUC-ROC']), rankings['AUC-ROC'][idx] if rankings else 0,
            np.mean(aggregate_results[k]['AUC-PR']), np.std(aggregate_results[k]['AUC-PR']), rankings['AUC-PR'][idx] if rankings else 0,
            np.mean(aggregate_results[k]['F1']), np.std(aggregate_results[k]['F1']), rankings['F1'][idx] if rankings else 0,
            np.mean(aggregate_results[k]['P']), np.std(aggregate_results[k]['P']), rankings['P'][idx] if rankings else 0,
            np.mean(aggregate_results[k]['R']), np.std(aggregate_results[k]['R']), rankings['R'][idx] if rankings else 0,
        ))
    return aggregate_results, rankings 

def main():
    args = get_args()
    if args.split_idx is None:
        L = []
        for i in range(args.n_splits):
            args.split_idx = i
            args.exp_dir = None
            results = get_metrics(args)
            L.append(results)
        aggregate_results(L)
    else:
        print(args) 
        get_metrics(args)

if __name__ == '__main__':
    main()

    