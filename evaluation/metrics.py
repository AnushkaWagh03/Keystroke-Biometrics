import numpy as np

def calculate_rank_n_accuracy(rankings, true_labels, n_values=[1, 3, 5, 10, 20]):
    if not rankings or not true_labels or len(rankings) != len(true_labels):
        return {}
        
    accuracies = {}
    total_samples = len(true_labels)
    
    for n in n_values:
        correct = 0
        for i in range(total_samples):
            # Check if true_label is in the top-N predicted rankings
            top_n_preds = rankings[i][:n]
            if true_labels[i] in top_n_preds:
                correct += 1
        accuracies[f"Rank-{n} Accuracy"] = float(correct / total_samples)
        
    return accuracies

def calculate_roc_auc(scores, labels, n_thresholds=None):
    if not scores or not labels or len(scores) == 0 or len(labels) == 0:
        return [], [], 0.0
        
    scores = np.array(scores)
    labels = np.array(labels)
    
    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]
    
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return [], [], 0.0
        
    unique_thresholds = np.sort(np.unique(scores))
    if n_thresholds and len(unique_thresholds) > n_thresholds:
        # Subsample thresholds to speed up computation
        indices = np.round(np.linspace(0, len(unique_thresholds) - 1, n_thresholds)).astype(int)
        thresholds = unique_thresholds[indices]
    else:
        thresholds = unique_thresholds
    
    fpr = []
    tpr = []
    
    for th in thresholds:
        _fpr = np.sum(impostor_scores >= th) / len(impostor_scores)
        _tpr = np.sum(genuine_scores >= th) / len(genuine_scores)
        fpr.append(float(_fpr))
        tpr.append(float(_tpr))
        
    fpr_array = np.array(fpr)
    tpr_array = np.array(tpr)
    
    # Use trapezoidal rule (handle NumPy 2.0 removal of trapz)
    if hasattr(np, 'trapezoid'):
        auc = np.trapezoid(tpr_array[::-1], fpr_array[::-1])
    elif hasattr(np, 'trapz'):
        auc = np.trapz(tpr_array[::-1], fpr_array[::-1])
    else:
        # Manual implementation of trapezoidal rule
        x = fpr_array[::-1]
        y = tpr_array[::-1]
        auc = np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2.0)
    
    return fpr, tpr, float(auc)

def calculate_eer(scores, labels, n_thresholds=None):
    if not scores or not labels or len(scores) == 0 or len(labels) == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    scores = np.array(scores)
    labels = np.array(labels)
    
    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]
    
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    unique_thresholds = np.sort(np.unique(scores))
    if n_thresholds and len(unique_thresholds) > n_thresholds:
        # Subsample thresholds to speed up computation
        indices = np.round(np.linspace(0, len(unique_thresholds) - 1, n_thresholds)).astype(int)
        thresholds = unique_thresholds[indices]
    else:
        thresholds = unique_thresholds
    
    far = []
    frr = []
    
    for th in thresholds:
        _far = np.sum(impostor_scores >= th) / len(impostor_scores)
        _frr = np.sum(genuine_scores < th) / len(genuine_scores)
        far.append(_far)
        frr.append(_frr)
        
    far = np.array(far)
    frr = np.array(frr)
    
    diff = np.abs(far - frr)
    min_idx = np.argmin(diff)
    
    eer = (far[min_idx] + frr[min_idx]) / 2.0
    optimal_th = thresholds[min_idx]
    
    return far.tolist(), frr.tolist(), float(eer), float(optimal_th)

def compute_metrics(results, config=None):
    print("Computing metrics...")
    
    # Get evaluation config with defaults
    eval_cfg = config.get('evaluation', {}) if config else {}
    
    # Verification metrics (EER, ROC, FAR, FRR, ANGA, ANIA)
    verif_cfg = eval_cfg.get('verification', {'enabled': True})
    if verif_cfg.get('enabled', False):
        requested_metrics = verif_cfg.get('metrics', ["EER", "ROC_AUC", "FAR", "FRR", "ANGA", "ANIA"])
        n_thresholds = verif_cfg.get('n_thresholds')
        
        if 'scores' in results and 'labels' in results:
            # We always need curves for verification metrics
            far_list, frr_list, eer, opt_th = calculate_eer(results['scores'], results['labels'], n_thresholds)
            fpr_list, tpr_list, roc_auc = calculate_roc_auc(results['scores'], results['labels'], n_thresholds)
            
            if "EER" in requested_metrics:
                results['EER'] = float(eer)
                results['Optimal_Threshold'] = float(opt_th)
            if "ROC_AUC" in requested_metrics:
                results['ROC_AUC'] = float(roc_auc)
            if "FAR" in requested_metrics:
                results['FAR'] = far_list
            if "FRR" in requested_metrics:
                results['FRR'] = frr_list
                
            # These are usually needed for plotting anyway if visualization is enabled
            results['FPR'] = fpr_list
            results['TPR'] = tpr_list
        
        # Filter other verification metrics like ANGA/ANIA if they exist
        for m in ["ANGA", "ANIA"]:
            if m in results and m not in requested_metrics:
                del results[m]
    else:
        # If verification is disabled, remove related metrics
        for m in ["EER", "Optimal_Threshold", "ROC_AUC", "FAR", "FRR", "FPR", "TPR", "ANGA", "ANIA"]:
            if m in results: del results[m]
    
    # Identification metrics (Rank-N)
    ident_cfg = eval_cfg.get('identification', {'enabled': True})
    if ident_cfg.get('enabled', False) and 'ident_rankings' in results and 'ident_true_labels' in results:
        n_values = ident_cfg.get('rank_n_values', [1, 3, 5, 10, 20])
        rank_n_results = calculate_rank_n_accuracy(
            results['ident_rankings'], 
            results['ident_true_labels'], 
            n_values
        )
        results.update(rank_n_results)
        
        # Also store Rank-1 predictions for confusion matrix
        results['ident_rank1_preds'] = [r[0] for r in results['ident_rankings']]
    
    # Performance metrics (Authentication Time)
    perf_cfg = eval_cfg.get('performance', {'enabled': True})
    if perf_cfg.get('enabled', False) and 'auth_times' in results:
        times = np.array(results['auth_times'])
        results['Avg Auth Time (ms)'] = float(np.mean(times) * 1000)
        results['Min Auth Time (ms)'] = float(np.min(times) * 1000)
        results['Max Auth Time (ms)'] = float(np.max(times) * 1000)
        results['Std Auth Time (ms)'] = float(np.std(times) * 1000)
    
    # Print results summary
    for key, value in results.items():
        if isinstance(value, list) and len(value) > 20:
            # Don't print large lists
            continue
        if key not in ['scores', 'labels', 'ident_rankings', 'ident_true_labels', 'auth_times', 'ident_rank1_preds']:
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
    
    return results