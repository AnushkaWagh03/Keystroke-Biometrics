import matplotlib.pyplot as plt

def generate_plots(results, save_path):
    scalar_metrics = {k: v for k, v in results.items() if not isinstance(v, (list, tuple)) and k not in ['scores', 'labels']}
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    if scalar_metrics:
        metrics = list(scalar_metrics.keys())
        values = list(scalar_metrics.values())
        axes[0].bar(metrics, values)
        axes[0].set_title("Evaluation Metrics")
        axes[0].set_xlabel("Metrics")
        axes[0].set_ylabel("Values")
        axes[0].tick_params(axis='x', rotation=45)
    
    if 'FPR' in results and 'TPR' in results:
        fpr = results['FPR']
        tpr = results['TPR']
        auc = results.get('ROC_AUC', 0.0)
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('Receiver Operating Characteristic')
        axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print("Plots generated and saved.")
