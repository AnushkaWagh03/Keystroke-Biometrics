import argparse
import os
import json
from utils.config_loader import load_config
from training.trainer import Trainer
from evaluation.metrics import compute_metrics
from evaluation.plots import generate_plots

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--save_scores", action='store_true', help="Save raw scores to file")
    args = parser.parse_args()

    # Robust path resolution
    config_path = args.config
    if not os.path.exists(config_path):
        # Try relative to the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
            
    config = load_config(config_path)
    print("Experiment Loaded:", config["experiment_name"], "\n")

    trainer = Trainer(config)
    trainer.train()
    results = trainer.evaluate()
    
    metrics = compute_metrics(results, config)

    # Resolve report directory relative to project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.join(script_dir, "reports", config["experiment_name"])
    os.makedirs(report_dir, exist_ok=True)

    plot_path = os.path.join(report_dir, "metrics_plot.png")
    generate_plots(metrics, plot_path, config)
    
    # Save metrics to JSON file
    metrics_path = os.path.join(report_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save scores if requested
    if args.save_scores and 'scores' in results:
        scores_path = os.path.join(report_dir, "scores.json")
        scores_to_save = {
            'scores': [float(s) for s in results['scores']],
            'labels': [int(l) for l in results['labels']]
        }
        if 'auth_times' in results:
            scores_to_save['auth_times'] = [float(t) for t in results['auth_times']]
            
        with open(scores_path, 'w') as f:
            json.dump(scores_to_save, f, indent=4)

    print("\nExperiment is completed successfully.")
    print(f"Results saved to: {report_dir}")

if __name__ == "__main__":
    main()