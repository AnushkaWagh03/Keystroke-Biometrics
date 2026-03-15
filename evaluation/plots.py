import matplotlib.pyplot as plt

def generate_plots(results, save_path):
    metrics = list(results.keys())
    values = list(results.values())

    plt.figure()
    plt.bar(metrics, values)
    plt.title("Evaluation Metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Values")
    plt.savefig(save_path)
    plt.close()

    print("Plots generated and saved.")
