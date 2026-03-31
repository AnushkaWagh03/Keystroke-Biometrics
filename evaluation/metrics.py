import numpy as np

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
    
    # Your existing loop - KEEP THIS
    for key, value in results.items():
        # Avoid printing large arrays directly to console if possible, but keep the structure
        if isinstance(value, list) and len(value) > 20:
            print(f"{key}: [List of {len(value)} items]")
        else:
            print(f"{key}: {value}")
    
    # ADD this return statement
    return results