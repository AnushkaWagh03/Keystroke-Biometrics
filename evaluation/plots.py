import matplotlib.pyplot as plt
import numpy as np

def generate_plots(results, save_path, config=None):
    # Get visualization config with defaults
    viz_cfg = {}
    if config and 'evaluation' in config and 'visualization' in config['evaluation']:
        viz_cfg = config['evaluation']['visualization']
    else:
        # Defaults if not specified
        viz_cfg = {
            'plot_verification_bars': True,
            'plot_auth_times': True,
            'plot_rank_n': True,
            'plot_roc': True
        }

    # Separate metrics by type
    accuracy_metrics = {}
    time_metrics = {}
    rank_n_metrics = {}
    
    for k, v in results.items():
        if not isinstance(v, (list, tuple)) and k not in ['scores', 'labels', 'ident_rankings', 'ident_true_labels', 'auth_times']:
            if "(ms)" in k:
                time_metrics[k] = v
            elif "Rank-" in k:
                rank_n_metrics[k] = v
            elif k not in ['EER', 'Optimal_Threshold', 'ROC_AUC']:
                # Other scalar metrics
                accuracy_metrics[k] = v
    
    # Add EER/ROC_AUC to accuracy metrics if present
    if 'EER' in results: accuracy_metrics['EER'] = results['EER']
    if 'ROC_AUC' in results: accuracy_metrics['ROC_AUC'] = results['ROC_AUC']

    n_subplots = 0
    show_acc = accuracy_metrics and viz_cfg.get('plot_verification_bars', True)
    show_time = time_metrics and viz_cfg.get('plot_auth_times', True)
    show_rank = rank_n_metrics and viz_cfg.get('plot_rank_n', True)
    show_roc = 'FPR' in results and 'TPR' in results and viz_cfg.get('plot_roc', True)

    if show_acc: n_subplots += 1
    if show_time: n_subplots += 1
    if show_rank: n_subplots += 1
    if show_roc: n_subplots += 1
    
    if n_subplots == 0:
        print("No metrics to plot (or all disabled in config).")
        return

    # Check if we should save individual plots
    save_individual = viz_cfg.get('save_individual_plots', True)
    import os
    report_dir = os.path.dirname(save_path)

    fig, axes = plt.subplots(1, n_subplots, figsize=(6 * n_subplots, 5))
    if n_subplots == 1: axes = [axes] # Handle single subplot case
    
    curr_ax = 0
    
    if show_acc:
        metrics = list(accuracy_metrics.keys())
        values = list(accuracy_metrics.values())
        axes[curr_ax].bar(metrics, values, color='skyblue')
        axes[curr_ax].set_title("Verification Metrics")
        axes[curr_ax].set_xlabel("Metrics")
        axes[curr_ax].set_ylabel("Value")
        axes[curr_ax].set_ylim([0, 1.1])
        axes[curr_ax].tick_params(axis='x', rotation=45)
        
        if save_individual:
            individual_fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(metrics, values, color='skyblue')
            ax.set_title("Verification Metrics")
            ax.set_xlabel("Metrics")
            ax.set_ylabel("Value")
            ax.set_ylim([0, 1.1])
            plt.setp(ax.get_xticklabels(), rotation=45)
            individual_fig.tight_layout()
            individual_fig.savefig(os.path.join(report_dir, "verification_bars.png"))
            plt.close(individual_fig)
            
        curr_ax += 1
        
    if show_rank:
        # Sort Rank-N metrics by N
        sorted_keys = sorted(rank_n_metrics.keys(), key=lambda x: int(x.split('-')[1].split(' ')[0]))
        metrics = [k.replace(" Accuracy", "") for k in sorted_keys]
        values = [rank_n_metrics[k] for k in sorted_keys]
        axes[curr_ax].bar(metrics, values, color='lightgreen')
        axes[curr_ax].set_title("Identification Accuracy (Rank-N)")
        axes[curr_ax].set_xlabel("Rank (N)")
        axes[curr_ax].set_ylabel("Accuracy")
        axes[curr_ax].set_ylim([0, 1.1])
        axes[curr_ax].tick_params(axis='x', rotation=0)
        
        if save_individual:
            individual_fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(metrics, values, color='lightgreen')
            ax.set_title("Identification Accuracy (Rank-N)")
            ax.set_xlabel("Rank (N)")
            ax.set_ylabel("Accuracy")
            ax.set_ylim([0, 1.1])
            individual_fig.tight_layout()
            individual_fig.savefig(os.path.join(report_dir, "rank_n_accuracy.png"))
            plt.close(individual_fig)
            
        curr_ax += 1

    if show_time:
        metrics = [m.replace(" Auth Time (ms)", "") for m in time_metrics.keys()]
        values = list(time_metrics.values())
        axes[curr_ax].bar(metrics, values, color='salmon')
        axes[curr_ax].set_title("Auth Time Metrics (ms)")
        axes[curr_ax].set_xlabel("Metrics")
        axes[curr_ax].set_ylabel("Time (ms)")
        axes[curr_ax].tick_params(axis='x', rotation=45)
        
        if save_individual:
            individual_fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(metrics, values, color='salmon')
            ax.set_title("Authentication Time Performance")
            ax.set_xlabel("Metric")
            ax.set_ylabel("Time (ms)")
            plt.setp(ax.get_xticklabels(), rotation=45)
            individual_fig.tight_layout()
            individual_fig.savefig(os.path.join(report_dir, "auth_times.png"))
            plt.close(individual_fig)
            
        curr_ax += 1
    
    if show_roc:
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
        
        if save_individual:
            individual_fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('Receiver Operating Characteristic (ROC)')
            ax.legend(loc="lower right")
            individual_fig.tight_layout()
            individual_fig.savefig(os.path.join(report_dir, "roc_curve.png"))
            plt.close(individual_fig)

    # Confusion Matrix for Identification (Rank-1)
    if 'ident_rank1_preds' in results and 'ident_true_labels' in results and viz_cfg.get('plot_rank_n', True):
        y_true = np.array(results['ident_true_labels'])
        y_pred = np.array(results['ident_rank1_preds'])
        
        classes = np.unique(np.concatenate([y_true, y_pred]))
        cm = np.zeros((len(classes), len(classes)))
        for t, p in zip(y_true, y_pred):
            cm[np.where(classes == t)[0], np.where(classes == p)[0]] += 1
            
        fig_cm, ax_cm = plt.subplots(figsize=(10, 8))
        im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax_cm.set_title("Identification Confusion Matrix (Rank-1)")
        plt.colorbar(im, ax=ax_cm)
        ax_cm.set_xlabel("Predicted Label")
        ax_cm.set_ylabel("True Label")
        fig_cm.tight_layout()
        fig_cm.savefig(os.path.join(report_dir, "confusion_matrix.png"))
        plt.close(fig_cm)
        print(f"Confusion matrix saved to: {report_dir}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Summary plot saved to: {save_path}")
    if save_individual:
        print(f"Individual plots saved to: {report_dir}")
