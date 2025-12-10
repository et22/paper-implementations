import matplotlib.pyplot as plt
import os

def plot_figure2(loss_results):
    plt.figure(figsize=(7, 4))
    for key in loss_results.keys():
        plt.scatter(loss_results[key]['seq_len'], loss_results[key]['loss'], label=key)
    plt.legend()
    plt.ylabel("loss")
    plt.xlabel("number of unique keys / sequence length")
    plt.tight_layout()
    plt.savefig("figures/figure2.png")

def plot_figure3(step_results):
    plt.figure(figsize=(10, 6))
    for key in step_results.keys():
        plt.scatter(step_results[key]['step'], step_results[key]['loss'], label=key)
    
    plt.legend()
    plt.yscale("log")

    plt.ylabel("loss")
    plt.xlabel("step")
    plt.tight_layout()
    plt.savefig("figures/figure3.png")

def aggregate_loss_results(log_directory):
    loss_results = {}

    for fname in os.listdir(log_directory):
        if not fname.endswith(".csv"):
            continue
        
        model_name = fname.replace(".csv", "")
        seq_lens = []
        losses = []

        csv_path = os.path.join(log_directory, fname)

        with open(csv_path, "r") as f:
            for line in f:
                if line.strip() == "":
                    continue

                parts = line.strip().split(",")
                if len(parts) != 2:
                    continue

                seq = float(parts[0])
                loss = float(parts[1])

                seq_lens.append(seq)
                losses.append(loss)

        loss_results[model_name] = {
            "seq_len": seq_lens,
            "loss": losses
        }

    return loss_results

if __name__ == "__main__":
    log_directory = "./logs/"
    loss_results = aggregate_loss_results(log_directory)
    plot_figure2(loss_results)


