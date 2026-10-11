import argparse
import copy
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import cohen_kappa_score

from data import leave_one_subject_out, within_subject
from empl import EMPL


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def batches(x, y, batch_size, shuffle):
    idx = torch.randperm(len(y)) if shuffle else torch.arange(len(y))
    for i in range(0, len(y), batch_size):
        j = idx[i:i + batch_size]
        yield x[j], y[j]


@torch.no_grad()
def evaluate(model, x, y, device, batch_size=256):
    model.eval()
    preds = []
    for xb, _ in batches(x, y, batch_size, shuffle=False):
        preds.append(model(xb.to(device)).argmax(dim=1).cpu())
    preds = torch.cat(preds).numpy()
    return float((preds == y.numpy()).mean()), float(cohen_kappa_score(y.numpy(), preds))


def train(model, x_tr, y_tr, epochs, device, gamma, lr, batch_size, x_va=None, y_va=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_acc, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        for xb, yb in batches(x_tr, y_tr, batch_size, shuffle=True):
            xb, yb = xb.to(device), yb.to(device)
            loss = F.cross_entropy(model(xb), yb) + gamma * model.group_diversity_loss()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if x_va is not None:
            acc, _ = evaluate(model, x_va, y_va, device)
            if acc > best_acc:
                best_acc, best_state = acc, copy.deepcopy(model.state_dict())
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol", choices=["within", "loso"], default="within")
    p.add_argument("--subjects", type=int, nargs="+", default=list(range(1, 10)))
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=9e-4)
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default="results")
    args = p.parse_args()
    epochs = args.epochs or (1000 if args.protocol == "within" else 125)
    os.makedirs(args.out, exist_ok=True)

    results = {}
    for subject in args.subjects:
        if args.protocol == "within":
            x_tr, y_tr, x_te, y_te = within_subject(subject)
            x_va = y_va = None
        else:
            x_tr, y_tr, x_va, y_va, x_te, y_te = leave_one_subject_out(subject)
            x_va, y_va = torch.tensor(x_va), torch.tensor(y_va)
        x_tr, y_tr, x_te, y_te = map(torch.tensor, (x_tr, y_tr, x_te, y_te))
        accs, kappas = [], []
        for run in range(args.runs):
            set_seed(run)
            model = EMPL(in_channels=x_tr.shape[1], seq_len=x_tr.shape[2], n_classes=int(y_tr.max()) + 1).to(args.device)
            model = train(model, x_tr, y_tr, epochs, args.device, args.gamma, args.lr, args.batch_size, x_va, y_va)
            acc, kappa = evaluate(model, x_te, y_te, args.device)
            accs.append(acc); kappas.append(kappa)
            print(f"{args.protocol} subject {subject} run {run}: acc {acc:.4f} kappa {kappa:.4f}", flush=True)
        results[subject] = {"acc": accs, "kappa": kappas,
                            "acc_mean": float(np.mean(accs)), "kappa_mean": float(np.mean(kappas))}
        with open(os.path.join(args.out, f"{args.protocol}.json"), "w") as f:
            json.dump(results, f, indent=2)
    print({s: round(r["acc_mean"], 4) for s, r in results.items()})


if __name__ == "__main__":
    main()
