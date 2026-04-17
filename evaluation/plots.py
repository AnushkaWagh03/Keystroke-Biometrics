import matplotlib.pyplot as plt

def generate_plots(results, save_path):
    # Separate metrics by type
    accuracy_metrics = {}
    time_metrics = {}
    
    for k, v in results.items():
        if not isinstance(v, (list, tuple)) and k not in ['scores', 'labels']:
            if "(ms)" in k:
                time_metrics[k] = v
            else:
                accuracy_metrics[k] = v
    
    n_subplots = 1 # ROC is always there if FPR/TPR are
    if accuracy_metrics: n_subplots += 1
    if time_metrics: n_subplots += 1
    
    fig, axes = plt.subplots(1, n_subplots, figsize=(6 * n_subplots, 5))
    if n_subplots == 1: axes = [axes] # Handle single subplot case
    
    curr_ax = 0
    
    if accuracy_metrics:
        metrics = list(accuracy_metrics.keys())
        values = list(accuracy_metrics.values())
        axes[curr_ax].bar(metrics, values, color='skyblue')
        axes[curr_ax].set_title("Accuracy Metrics")
        axes[curr_ax].set_xlabel("Metrics")
        axes[curr_ax].set_ylabel("Value")
        axes[curr_ax].set_ylim([0, 1.1])
        axes[curr_ax].tick_params(axis='x', rotation=45)
        curr_ax += 1
        
    if time_metrics:
        metrics = [m.replace(" Auth Time (ms)", "") for m in time_metrics.keys()]
        values = list(time_metrics.values())
        axes[curr_ax].bar(metrics, values, color='salmon')
        axes[curr_ax].set_title("Auth Time Metrics (ms)")
        axes[curr_ax].set_xlabel("Metrics")
        axes[curr_ax].set_ylabel("Time (ms)")
        axes[curr_ax].tick_params(axis='x', rotation=45)
        curr_ax += 1
    
    if 'FPR' in results and 'TPR' in results:
        fpr = results['FPR']
        tpr = results['TPR']
        auc = results.get('ROC_AUC', 0.0)
        axes[curr_ax].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        axes[curr_ax].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[curr_ax].set_xlim([0.0, 1.0])
        axes[curr_ax].set_ylim([0.0, 1.05])
        axes[curr_ax].set_xlabel('False Positive Rate')
        axes[curr_ax].set_ylabel('True Positive Rate')
        axes[curr_ax].set_title('Receiver Operating Characteristic')
        axes[curr_ax].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print("Plots generated and saved.")
