import numpy as np

def calculate_roc_auc(scores, labels):
    if not scores or not labels or len(scores) == 0 or len(labels) == 0:
        return [], [], 0.0
        
    scores = np.array(scores)
    labels = np.array(labels)
    
    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]
    
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return [], [], 0.0
        
    thresholds = np.sort(np.unique(scores))
    
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

def calculate_eer(scores, labels):
    if not scores or not labels or len(scores) == 0 or len(labels) == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    scores = np.array(scores)
    labels = np.array(labels)
    
    genuine_scores = scores[labels == 1]
    impostor_scores = scores[labels == 0]
    
    if len(genuine_scores) == 0 or len(impostor_scores) == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    thresholds = np.sort(np.unique(scores))
    
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

def compute_metrics(results):
    print("Computing metrics...")
    
    if 'scores' in results and 'labels' in results:
        far, frr, eer, opt_th = calculate_eer(results['scores'], results['labels'])
        results['FAR'] = far
        results['FRR'] = frr
        results['EER'] = eer
        results['Optimal_Threshold'] = opt_th
        
        fpr, tpr, roc_auc = calculate_roc_auc(results['scores'], results['labels'])
        results['FPR'] = fpr
        results['TPR'] = tpr
        results['ROC_AUC'] = roc_auc
    
    if 'auth_times' in results:
        times = np.array(results['auth_times'])
        results['Avg Auth Time (ms)'] = float(np.mean(times) * 1000)
        results['Min Auth Time (ms)'] = float(np.min(times) * 1000)
        results['Max Auth Time (ms)'] = float(np.max(times) * 1000)
        results['Std Auth Time (ms)'] = float(np.std(times) * 1000)
    
    # Your existing loop - KEEP THIS
    for key, value in results.items():
        # Avoid printing large arrays directly to console if possible, but keep the structure
        if isinstance(value, list) and len(value) > 20:
            print(f"{key}: [List of {len(value)} items]")
        else:
            print(f"{key}: {value}")
    
    # ADD this return statement
    return results