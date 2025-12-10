# training pipeline, based on
# https://github.com/ischlag/fast-weight-transformers/blob/main/synthetic/main.py
import argparse
import torch
import numpy as np
import os

from attention_models import SoftmaxAttention, LinearAttention
from synthetic_task import sample_batch

def train(n_keys, seq_len, d_key, batch_size, attn_name, attn_arg, update_rule, model_name):
    max_steps = 100000
    log_every = 25
    test_every = 100
    test_sequences = 20
    stop_criterion = 0.001
    log_folder = "logs"
    learning_rate = 1e-3
    best_loss = np.inf

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if attn_name == "softmax":
        model = SoftmaxAttention(embed_dim=2*n_keys, d_key=d_key, n_keys_values=n_keys)
    else:
        model = LinearAttention(embed_dim=2*n_keys, d_key=d_key, n_keys_values=n_keys, attention_type=attn_name, update_rule=update_rule, arg=attn_arg)
    
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    losses = []
    for step in range(1, max_steps+1):
        model.train()
        key_ids, query_ids, val_onehot, targets = sample_batch(batch_size, seq_len)
        preds = model(key_ids, query_ids, val_onehot)
        loss = 0.5 * ((targets - preds) ** 2)
        loss = loss.sum(dim=-1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.cpu().detach().numpy())

        if step % test_every == 0 and len(losses) > 100:
            loss_mean = np.mean(losses[-100:])

            if loss_mean < best_loss:
                best_loss = loss_mean
                stop_cnt = 0
            else:
                stop_cnt = stop_cnt + 1

            if loss_mean <= stop_criterion or stop_cnt > 10:
                out_path = os.path.join(log_folder, model_name)

                with open(out_path, "a") as f:
                    f.write(f"{int(seq_len)}, {float(loss_mean):.8f}\n")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to train one attention model.")
    parser.add_argument("--attn_name", default="softmax", help="which attention type is used (softmax, linear, favor, dpfp")
    parser.add_argument("--attn_arg", type=int, help="int argument of the respective attention type", default=0)
    parser.add_argument("--update_rule", help="name of the update rule (sum, fwm, ours)", default="sum")
    args = parser.parse_args()

    d_key = 64
    batch_size = 32

    for n_keys in range(20, 501, 20):
        train(n_keys = n_keys, 
                seq_len = n_keys, 
                d_key = d_key, 
                batch_size=batch_size, 
                attn_name=args.attn_name, 
                attn_arg=args.attn_arg, 
                update_rule=args.update_rule,
                model_name=f"{args.attn_name}_{args.update_rule}_{args.attn_arg}.csv")

