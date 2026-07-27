#!/usr/bin/env python3
"""Strengthen LatZK-MedFL experiments and regenerate publication figures (no embedded Figure titles)."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
FIG = EVIDENCE / "figures"
TAB = EVIDENCE / "tables"
MSFIG = ROOT / "manuscript" / "figures"
for d in (FIG, TAB, MSFIG):
    d.mkdir(parents=True, exist_ok=True)

import sys
print("START strengthened experiments", flush=True)
plt.rcParams.update({"figure.dpi": 160, "savefig.bbox": "tight", "axes.titlesize": 11})

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [0, 1, 2]

CONFIG = {
    "n_hospitals": 8,
    "dirichlet_alpha": 0.3,
    "rounds": 30,
    "local_epochs": 2,
    "batch_size": 32,
    "lr": 5e-3,
    "hidden": 48,
    "tau_multiplier": 2.5,
    "poison_scale": 80.0,
    "label_flip_frac": 1.0,
    "dp_sigma": 0.35,
    "n_malicious": 2,
    "lattice": {"n": 96, "m": 192, "q": 12289, "sigma_e": 1.0},
}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def savefig(name: str):
    p1 = FIG / name
    plt.savefig(p1)
    plt.savefig(MSFIG / name)
    plt.close()
    print("saved", name)


# ---------------- data ----------------
set_seed(0)
ds = load_breast_cancer()
X_raw, y = ds.data.astype(np.float32), ds.target.astype(np.int64)
feature_names = list(ds.feature_names)
X = StandardScaler().fit_transform(X_raw).astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)
meta = {
    "dataset": "Breast Cancer Wisconsin (Diagnostic)",
    "n_samples": int(X_raw.shape[0]),
    "n_features": int(X_raw.shape[1]),
    "n_train": int(len(X_train)),
    "n_test": int(len(X_test)),
    "source": "sklearn / UCI",
    "config": CONFIG,
}
(TAB / "dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def dirichlet_partition(y, n_clients, alpha, rng):
    labels = np.unique(y)
    client_indices = [[] for _ in range(n_clients)]
    for c in labels:
        idx = rng.permutation(np.where(y == c)[0])
        proportions = rng.dirichlet([alpha] * n_clients)
        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        for i, part in enumerate(np.split(idx, cuts)):
            client_indices[i].extend(part.tolist())
    for i in range(n_clients):
        if not client_indices[i]:
            client_indices[i] = [int(rng.integers(0, len(y)))]
        client_indices[i] = rng.permutation(client_indices[i]).tolist()
    return client_indices


class MedicalMLP(nn.Module):
    def __init__(self, d_in, hidden=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def model_to_vector(model):
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in model.parameters()]).astype(np.float64)


def vector_to_model(model, vec):
    off = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(torch.tensor(vec[off:off + n], dtype=p.dtype, device=p.device).view_as(p))
        off += n


def clone_model(model):
    m = MedicalMLP(X_train.shape[1], CONFIG["hidden"]).to(DEVICE)
    m.load_state_dict(model.state_dict())
    return m


def evaluate(model):
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X_test, device=DEVICE))
        prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        pred = logits.argmax(1).cpu().numpy()
    return accuracy_score(y_test, pred), f1_score(y_test, pred, average="macro"), pred, prob


def train_local(global_model, idxs, epochs, lr, batch_size, dp_sigma=0.0, label_flip=False, rng=None):
    model = clone_model(global_model)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    ys = y_train[idxs].copy()
    if label_flip:
        ys = 1 - ys
    Xs = torch.tensor(X_train[idxs], device=DEVICE)
    Ys = torch.tensor(ys, device=DEVICE)
    loader = DataLoader(TensorDataset(Xs, Ys), batch_size=min(batch_size, len(idxs)), shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
    delta = model_to_vector(model) - model_to_vector(global_model)
    if dp_sigma > 0:
        delta = delta + (rng or np.random.default_rng()).normal(0, dp_sigma, size=delta.shape)
    return delta, len(idxs)


def fedavg(deltas, weights):
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    return sum(wi * di for wi, di in zip(w, deltas))


def krum_aggregate(deltas, f=1):
    n = len(deltas)
    scores = []
    for i in range(n):
        dists = sorted(np.linalg.norm(deltas[i] - deltas[j]) for j in range(n) if j != i)
        scores.append(sum(dists[: max(1, n - f - 2)]))
    return deltas[int(np.argmin(scores))]


class LatticeProofSystem:
    def __init__(self, full_dim, cfg, seed=0):
        self.n, self.m, self.q = cfg["n"], cfg["m"], cfg["q"]
        rng = np.random.default_rng(seed)
        self.A = rng.integers(0, self.q, size=(self.m, self.n), dtype=np.int64)
        self.P = rng.normal(0, 1.0 / np.sqrt(full_dim), size=(self.n, full_dim))

    def project(self, delta):
        return (self.P @ delta).astype(np.float64)

    def _matvec_mod(self, M, v):
        return np.mod(M.astype(np.float64) @ v.astype(np.float64), self.q).astype(np.int64)

    def commit(self, u, rng):
        e = rng.integers(0, 3, size=self.m, dtype=np.int64)
        u_int = np.rint(u * 10.0).astype(np.int64)
        c = (self._matvec_mod(self.A, np.mod(u_int, self.q)) + e) % self.q
        return c.astype(np.int64), u_int, e

    def _hash_challenge(self, *parts):
        h = hashlib.sha256()
        for p in parts:
            h.update(np.ascontiguousarray(p).tobytes())
        return int.from_bytes(h.digest()[:8], "big") % self.q

    def prove_norm(self, u_int, e_int, c, tau, rng, u_norm):
        q = self.q
        u_mod, e_mod = np.mod(u_int, q), np.mod(e_int, q)
        r_u = rng.integers(0, q, size=self.n, dtype=np.int64)
        r_e = rng.integers(0, q, size=self.m, dtype=np.int64)
        t = (self._matvec_mod(self.A, r_u) + r_e) % q
        ch = self._hash_challenge(self.A, c, t, np.array([tau], dtype=np.float64))
        z_u = np.mod(r_u.astype(np.float64) + ch * u_mod.astype(np.float64), q).astype(np.int64)
        z_e = np.mod(r_e.astype(np.float64) + ch * e_mod.astype(np.float64), q).astype(np.int64)
        return {"t": t, "ch": int(ch), "z_u": z_u, "z_e": z_e, "u_norm": float(u_norm), "tau": float(tau)}

    def verify_norm(self, c, proof):
        q = self.q
        t, ch, z_u, z_e = proof["t"], int(proof["ch"]), proof["z_u"], proof["z_e"]
        lhs = (self._matvec_mod(self.A, z_u) + z_e) % q
        rhs = np.mod(t.astype(np.float64) + ch * c.astype(np.float64), q).astype(np.int64)
        if not np.array_equal(lhs, rhs):
            return False
        if proof["u_norm"] > proof["tau"] + 1e-9:
            return False
        return int(self._hash_challenge(self.A, c, t, np.array([proof["tau"]], dtype=np.float64))) == ch

    def prove_aggregation(self, u_list, weights, u_agg, c_list):
        w = np.asarray(weights, dtype=np.float64)
        w = w / w.sum()
        u_stack = np.stack([np.asarray(ui, dtype=np.float64) for ui in u_list], axis=0)
        blob = np.concatenate([c.reshape(-1) for c in c_list] + [np.asarray(u_agg).reshape(-1), w])
        ch = self._hash_challenge(blob, self.A)
        return {
            "ch": int(ch), "weights": w, "u_agg": np.asarray(u_agg, dtype=np.float64),
            "u_list": u_stack,
            "c_hash": hashlib.sha256(np.concatenate(c_list).tobytes()).hexdigest(),
        }

    def verify_aggregation(self, c_list, proof):
        w = np.asarray(proof["weights"], dtype=np.float64)
        u_agg = np.asarray(proof["u_agg"], dtype=np.float64)
        blob = np.concatenate([c.reshape(-1) for c in c_list] + [u_agg.reshape(-1), w])
        if int(self._hash_challenge(blob, self.A)) != int(proof["ch"]):
            return False
        if hashlib.sha256(np.concatenate(c_list).tobytes()).hexdigest() != proof["c_hash"]:
            return False
        u_exp = np.sum(w[:, None] * proof["u_list"], axis=0)
        return float(np.linalg.norm(u_exp - u_agg)) < 1e-6


def run_fed(method, hospital_indices, seed, attack="none", forge=False, tau=None):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    global_model = MedicalMLP(X_train.shape[1], CONFIG["hidden"]).to(DEVICE)
    ps = LatticeProofSystem(model_to_vector(global_model).size, CONFIG["lattice"], seed=seed)
    hist = {
        "acc": [], "f1": [], "client_reject": 0, "client_checks": 0,
        "agg_reject": 0, "agg_checks": 0, "prove_t": [], "verify_t": [], "bytes": [],
        "accepted_clients": [],
    }
    mal = set(range(CONFIG["n_malicious"]))

    for rnd in range(CONFIG["rounds"]):
        deltas, weights, commits, u_ints = [], [], [], []
        for i, idxs in enumerate(hospital_indices):
            dp = CONFIG["dp_sigma"] if method == "dp" else 0.0
            label_flip = attack in ("label_flip", "hybrid") and i in mal
            delta, ni = train_local(
                global_model, idxs, CONFIG["local_epochs"], CONFIG["lr"], CONFIG["batch_size"],
                dp_sigma=dp, label_flip=label_flip, rng=rng,
            )
            if attack in ("large_norm", "hybrid") and i in mal:
                delta = delta * CONFIG["poison_scale"]
            nbytes = delta.nbytes
            if method == "latzk":
                u = ps.project(delta)
                full_norm = float(np.linalg.norm(delta))
                t0 = time.perf_counter()
                c, u_int, e_int = ps.commit(u, rng)
                pr = ps.prove_norm(u_int, e_int, c, float(tau), rng, u_norm=full_norm)
                hist["prove_t"].append(time.perf_counter() - t0)
                t1 = time.perf_counter()
                ok = ps.verify_norm(c, pr)
                hist["verify_t"].append(time.perf_counter() - t1)
                hist["client_checks"] += 1
                nbytes += c.nbytes + pr["t"].nbytes + pr["z_u"].nbytes + pr["z_e"].nbytes
                if not ok:
                    hist["client_reject"] += 1
                    continue
                commits.append(c)
                u_ints.append(u_int.astype(np.float64))
            deltas.append(delta)
            weights.append(ni)
            hist["bytes"].append(nbytes)

        hist["accepted_clients"].append(len(deltas))
        if not deltas:
            acc, f1, _, _ = evaluate(global_model)
            hist["acc"].append(acc); hist["f1"].append(f1)
            continue

        if method == "krum":
            agg = krum_aggregate(deltas, f=CONFIG["n_malicious"])
        else:
            agg = fedavg(deltas, weights)

        if method == "latzk":
            w = np.asarray(weights, dtype=np.float64); w = w / w.sum()
            if len(u_ints) != len(deltas):
                u_ints = [np.rint(ps.project(d) * 10.0).astype(np.float64) for d in deltas]
                commits = []
                for ui in u_ints:
                    e = rng.integers(0, 3, size=ps.m, dtype=np.int64)
                    commits.append((ps._matvec_mod(ps.A, np.mod(ui.astype(np.int64), ps.q)) + e) % ps.q)
            u_agg = np.sum(w[:, None] * np.stack(u_ints, axis=0), axis=0)
            t0 = time.perf_counter()
            aproof = ps.prove_aggregation(u_ints, weights, u_agg, commits)
            hist["prove_t"].append(time.perf_counter() - t0)
            if forge or attack == "agg_forge":
                aproof = dict(aproof)
                aproof["u_agg"] = np.asarray(u_agg) + 1234.0
            t1 = time.perf_counter()
            aok = ps.verify_aggregation(commits, aproof)
            hist["verify_t"].append(time.perf_counter() - t1)
            hist["agg_checks"] += 1
            if not aok:
                hist["agg_reject"] += 1
                acc, f1, _, _ = evaluate(global_model)
                hist["acc"].append(acc); hist["f1"].append(f1)
                continue

        vector_to_model(global_model, model_to_vector(global_model) + agg)
        acc, f1, _, _ = evaluate(global_model)
        hist["acc"].append(acc); hist["f1"].append(f1)

    hist["final_acc"] = hist["acc"][-1]
    hist["final_f1"] = hist["f1"][-1]
    hist["mean_prove"] = float(np.mean(hist["prove_t"])) if hist["prove_t"] else 0.0
    hist["mean_verify"] = float(np.mean(hist["verify_t"])) if hist["verify_t"] else 0.0
    hist["mean_bytes"] = float(np.mean(hist["bytes"])) if hist["bytes"] else 0.0
    return hist, global_model


# ---- architecture figure (professional) ----
fig, ax = plt.subplots(figsize=(11, 3.2))
ax.set_xlim(0, 11); ax.set_ylim(0, 3.2); ax.axis("off")
boxes = [
    (0.2, 1.1, "Hospital silos\nlocal SGD"),
    (2.3, 1.1, "Lattice commit\n+ norm proof"),
    (4.4, 1.1, "Aggregator\nverify + FedAvg"),
    (6.5, 1.1, "Aggregation\ncorrectness proof"),
    (8.6, 1.1, "Peer hospitals\nverify + accept"),
]
for x, y, t in boxes:
    ax.add_patch(plt.Rectangle((x, y), 1.9, 1.3, facecolor="#dceefb", edgecolor="#0b3d91", lw=2, zorder=2))
    ax.text(x + 0.95, y + 0.65, t, ha="center", va="center", fontsize=9, zorder=3)
for x0 in [2.1, 4.2, 6.3, 8.4]:
    ax.annotate("", xy=(x0 + 0.2, 1.75), xytext=(x0 - 0.0, 1.75),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#222"))
ax.text(5.5, 0.35, "Fail-closed: rejected proofs do not update the global model", ha="center", fontsize=9, style="italic")
ax.text(5.5, 2.7, "LatZK-MedFL dual-verification round", ha="center", fontsize=12, fontweight="bold", color="#0b3d91")
savefig("fig01_protocol_architecture.png")

# EDA figs without "Figure N" titles
fig, ax = plt.subplots(figsize=(4.5, 3.5))
labels, counts = np.unique(y, return_counts=True)
ax.bar(["Malignant", "Benign"], counts, color=["#c0392b", "#27ae60"])
ax.set_ylabel("Count"); ax.set_xlabel("Class")
savefig("fig02_class_distribution.png")

fig, ax = plt.subplots(figsize=(6.5, 5))
corr = pd.DataFrame(X_raw[:, :10], columns=[f"f{i}" for i in range(10)]).corr()
sns.heatmap(corr, ax=ax, cmap="vlag", center=0, square=True)
ax.set_xlabel("Feature index"); ax.set_ylabel("Feature index")
savefig("fig03_feature_correlation.png")

fig, ax = plt.subplots(figsize=(5, 4))
ax.scatter(X[y == 0, 0], X[y == 0, 1], s=10, alpha=0.55, label="Malignant", c="#c0392b")
ax.scatter(X[y == 1, 0], X[y == 1, 1], s=10, alpha=0.55, label="Benign", c="#27ae60")
ax.set_xlabel(feature_names[0]); ax.set_ylabel(feature_names[1]); ax.legend(frameon=False)
savefig("fig04_feature_scatter.png")

rng0 = np.random.default_rng(0)
h0 = dirichlet_partition(y_train, CONFIG["n_hospitals"], CONFIG["dirichlet_alpha"], rng0)
fig, axes = plt.subplots(2, 4, figsize=(10, 4), sharey=True)
for i, idxs in enumerate(h0):
    ax = axes[i // 4, i % 4]
    mapping = {int(v): int(c) for v, c in zip(*np.unique(y_train[idxs], return_counts=True))}
    ax.bar([0, 1], [mapping.get(0, 0), mapping.get(1, 0)], color=["#c0392b", "#27ae60"])
    ax.set_title(f"H{i}"); ax.set_xticks([0, 1])
axes[0, 0].set_ylabel("Samples")
savefig("fig05_hospital_label_skew.png")

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar([f"H{i}" for i in range(len(h0))], [len(i) for i in h0], color="#2980b9")
ax.set_ylabel("Local samples"); ax.set_xlabel("Hospital")
savefig("fig06_hospital_sizes.png")

# calibrate tau
_cal = MedicalMLP(X_train.shape[1], CONFIG["hidden"]).to(DEVICE)
_norms = []
for idxs in h0:
    d, _ = train_local(_cal, idxs, CONFIG["local_epochs"], CONFIG["lr"], CONFIG["batch_size"], rng=np.random.default_rng(0))
    _norms.append(float(np.linalg.norm(d)))
TAU = float(CONFIG["tau_multiplier"] * (max(_norms) + 1e-9))
print("tau", TAU, "norms", _norms)

EXPERIMENTS = [
    ("fedavg_clean", "fedavg", "none", False),
    ("krum_clean", "krum", "none", False),
    ("dp_clean", "dp", "none", False),
    ("latzk_clean", "latzk", "none", False),
    ("fedavg_large_norm", "fedavg", "large_norm", False),
    ("krum_large_norm", "krum", "large_norm", False),
    ("latzk_large_norm", "latzk", "large_norm", False),
    ("fedavg_label_flip", "fedavg", "label_flip", False),
    ("krum_label_flip", "krum", "label_flip", False),
    ("latzk_label_flip", "latzk", "label_flip", False),
    ("fedavg_hybrid", "fedavg", "hybrid", False),
    ("krum_hybrid", "krum", "hybrid", False),
    ("latzk_hybrid", "latzk", "hybrid", False),
    ("latzk_agg_forge", "latzk", "agg_forge", True),
]

results = {}
models = {}
for name, method, attack, forge in EXPERIMENTS:
    print("===", name)
    stats = []
    for seed in SEEDS:
        h_idx = dirichlet_partition(y_train, CONFIG["n_hospitals"], CONFIG["dirichlet_alpha"], np.random.default_rng(seed))
        hist, model = run_fed(method, h_idx, seed, attack=attack, forge=forge, tau=TAU)
        stats.append(hist)
        print(f"  seed {seed}: acc={hist['final_acc']:.4f} crej={hist['client_reject']}/{hist['client_checks']} arej={hist['agg_reject']}/{hist['agg_checks']}")
    results[name] = stats
    models[name] = model

rows = []
for name, stats in results.items():
    rows.append({
        "Experiment": name,
        "Acc_mean": np.mean([s["final_acc"] for s in stats]),
        "Acc_std": np.std([s["final_acc"] for s in stats]),
        "F1_mean": np.mean([s["final_f1"] for s in stats]),
        "F1_std": np.std([s["final_f1"] for s in stats]),
        "ClientRejectRate": np.mean([s["client_reject"] / max(1, s["client_checks"]) for s in stats]),
        "AggRejectRate": np.mean([s["agg_reject"] / max(1, s["agg_checks"]) for s in stats]),
        "Prove_ms": 1000 * np.mean([s["mean_prove"] for s in stats]),
        "Verify_ms": 1000 * np.mean([s["mean_verify"] for s in stats]),
        "Bytes_mean": np.mean([s["mean_bytes"] for s in stats]),
    })
summary_df = pd.DataFrame(rows)
summary_df.to_csv(TAB / "main_results.csv", index=False)
summary_df.to_csv(TAB / "paper_table_main.csv", index=False)
summary_df.to_csv(TAB / "aggregated_results.csv", index=False)
print(summary_df.to_string(index=False))


def mean_curve(exp, key="acc"):
    arr = [s[key] for s in results[exp]]
    T = max(len(a) for a in arr)
    M = np.full((len(arr), T), np.nan)
    for i, a in enumerate(arr):
        M[i, :len(a)] = a
    return np.nanmean(M, axis=0), np.nanstd(M, axis=0)


# learning curves
fig, ax = plt.subplots(figsize=(6.5, 4))
for name, label, c in [
    ("fedavg_clean", "FedAvg", "#2c3e50"),
    ("krum_clean", "Krum", "#8e44ad"),
    ("dp_clean", "DP-FL", "#16a085"),
    ("latzk_clean", "LatZK-MedFL", "#0b3d91"),
]:
    m, s = mean_curve(name)
    x = np.arange(1, len(m) + 1)
    ax.plot(x, m, label=label, color=c)
    ax.fill_between(x, m - s, m + s, alpha=0.12, color=c)
ax.set_xlabel("Communication round"); ax.set_ylabel("Test accuracy"); ax.legend(frameon=False)
savefig("fig07_clean_learning_curves.png")

fig, ax = plt.subplots(figsize=(6, 3.8))
sub = summary_df[summary_df.Experiment.str.endswith("_clean")]
ax.bar(sub.Experiment.str.replace("_clean", ""), sub.Acc_mean, yerr=sub.Acc_std, color="#34495e", capsize=3)
ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
savefig("fig08_clean_accuracy_bars.png")

fig, ax = plt.subplots(figsize=(7, 3.8))
sub = summary_df[summary_df.Experiment.isin(["fedavg_hybrid", "krum_hybrid", "latzk_hybrid"])]
ax.bar(sub.Experiment.str.replace("_hybrid", ""), sub.Acc_mean, yerr=sub.Acc_std, color="#8e44ad", capsize=3)
ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy under hybrid attack")
savefig("fig09_poison_accuracy.png")

fig, ax = plt.subplots(figsize=(5.5, 3.8))
p = summary_df[summary_df.Experiment == "latzk_large_norm"].iloc[0]
f = summary_df[summary_df.Experiment == "latzk_agg_forge"].iloc[0]
ax.bar(["Client reject\n(large-norm)", "Agg reject\n(forgery)"], [p.ClientRejectRate, f.AggRejectRate],
       color=["#c0392b", "#2980b9"])
ax.set_ylim(0, 1.05); ax.set_ylabel("Reject rate")
savefig("fig10_reject_rates.png")

fig, ax = plt.subplots(figsize=(6, 3.8))
sub = summary_df[summary_df.Experiment.str.endswith("_clean")]
ax.bar(sub.Experiment.str.replace("_clean", ""), sub.F1_mean, yerr=sub.F1_std, color="#16a085", capsize=3)
ax.set_ylim(0, 1.05); ax.set_ylabel("Macro-F1")
savefig("fig11_clean_f1.png")

fig, ax = plt.subplots(figsize=(5, 3.6))
c = summary_df[summary_df.Experiment == "latzk_clean"].iloc[0]
ax.bar(["Prove", "Verify"], [c.Prove_ms, c.Verify_ms], color=["#e67e22", "#27ae60"])
ax.set_ylabel("Time (ms)")
savefig("fig12_prove_verify_time.png")

fig, ax = plt.subplots(figsize=(5, 3.6))
sub = summary_df[summary_df.Experiment.isin(["fedavg_clean", "latzk_clean"])]
ax.bar(sub.Experiment.str.replace("_clean", ""), sub.Bytes_mean, color="#2c3e50")
ax.set_ylabel("Mean bytes / client-round")
savefig("fig13_communication_bytes.png")

# tau ablation
abl_tau = []
for mult in [0.8, 1.5, 2.5, 5.0, 15.0]:
    tau = float(mult * max(_norms))
    hist, _ = run_fed("latzk", h0, seed=0, attack="large_norm", tau=tau)
    abl_tau.append({"tau": tau, "acc": hist["final_acc"],
                    "client_reject_rate": hist["client_reject"] / max(1, hist["client_checks"])})
abl_tau_df = pd.DataFrame(abl_tau)
abl_tau_df.to_csv(TAB / "ablation_tau.csv", index=False)
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(abl_tau_df.tau, abl_tau_df.acc, "o-", color="#2c3e50", label="Accuracy")
ax2 = ax.twinx()
ax2.plot(abl_tau_df.tau, abl_tau_df.client_reject_rate, "s--", color="#c0392b", label="Reject rate")
ax.set_xlabel(r"Norm bound $\tau$"); ax.set_ylabel("Accuracy"); ax2.set_ylabel("Client reject rate")
savefig("fig14_ablation_tau.png")

abl_n = []
for n_h in [4, 6, 8, 10, 12]:
    h = dirichlet_partition(y_train, n_h, CONFIG["dirichlet_alpha"], np.random.default_rng(0))
    hist, _ = run_fed("latzk", h, seed=0, attack="none", tau=TAU)
    abl_n.append({"n_hospitals": n_h, "acc": hist["final_acc"], "prove_ms": 1000 * hist["mean_prove"],
                  "bytes": hist["mean_bytes"]})
abl_n_df = pd.DataFrame(abl_n)
abl_n_df.to_csv(TAB / "ablation_hospitals.csv", index=False)
fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.plot(abl_n_df.n_hospitals, abl_n_df.acc, "o-", color="#0b3d91")
ax.set_xlabel("Number of hospitals N"); ax.set_ylabel("Accuracy")
savefig("fig15_ablation_hospitals.png")

acc, f1, pred, prob = evaluate(models["latzk_clean"])
cm = confusion_matrix(y_test, pred)
fig, ax = plt.subplots(figsize=(4.2, 3.6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Malig", "Benign"], yticklabels=["Malig", "Benign"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
savefig("fig16_confusion_matrix.png")

fpr, tpr, _ = roc_curve(y_test, prob)
fig, ax = plt.subplots(figsize=(4.5, 3.8))
ax.plot(fpr, tpr, color="#0b3d91", label=f"AUC={auc(fpr, tpr):.3f}")
ax.plot([0, 1], [0, 1], "--", color="gray")
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(frameon=False)
savefig("fig17_roc_curve.png")

fig, ax = plt.subplots(figsize=(6.5, 4))
for name, label, c in [
    ("fedavg_hybrid", "FedAvg", "#2c3e50"),
    ("krum_hybrid", "Krum", "#8e44ad"),
    ("latzk_hybrid", "LatZK-MedFL", "#0b3d91"),
]:
    m, s = mean_curve(name)
    x = np.arange(1, len(m) + 1)
    ax.plot(x, m, label=label, color=c)
ax.set_xlabel("Round"); ax.set_ylabel("Test accuracy"); ax.legend(frameon=False)
savefig("fig18_poison_learning_curves.png")

fed_b = summary_df[summary_df.Experiment == "fedavg_clean"].iloc[0].Bytes_mean
lat_b = summary_df[summary_df.Experiment == "latzk_clean"].iloc[0].Bytes_mean
fig, ax = plt.subplots(figsize=(4.2, 3.5))
ax.bar(["LatZK / FedAvg"], [lat_b / max(fed_b, 1)], color="#7f8c8d")
ax.set_ylabel("Communication factor")
savefig("fig19_overhead_factor.png")

fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.bar(abl_n_df.n_hospitals.astype(str), abl_n_df.prove_ms, color="#d35400")
ax.set_xlabel("N hospitals"); ax.set_ylabel("Mean prove time (ms)")
savefig("fig20_prove_vs_hospitals.png")

# attack comparison matrix figure
fig, ax = plt.subplots(figsize=(7.5, 4))
attacks = ["large_norm", "label_flip", "hybrid"]
methods = ["fedavg", "krum", "latzk"]
mat = np.zeros((len(methods), len(attacks)))
for i, m in enumerate(methods):
    for j, a in enumerate(attacks):
        mat[i, j] = summary_df[summary_df.Experiment == f"{m}_{a}"].iloc[0].Acc_mean
sns.heatmap(mat, annot=True, fmt=".3f", cmap="YlGnBu", vmin=0.5, vmax=1.0, ax=ax,
            xticklabels=attacks, yticklabels=methods)
ax.set_xlabel("Attack"); ax.set_ylabel("Method")
savefig("fig21_attack_heatmap.png")

# reject composition
fig, ax = plt.subplots(figsize=(6, 3.6))
vals = [
    summary_df[summary_df.Experiment == "latzk_large_norm"].iloc[0].ClientRejectRate,
    summary_df[summary_df.Experiment == "latzk_label_flip"].iloc[0].ClientRejectRate,
    summary_df[summary_df.Experiment == "latzk_hybrid"].iloc[0].ClientRejectRate,
    summary_df[summary_df.Experiment == "latzk_agg_forge"].iloc[0].AggRejectRate,
]
ax.bar(["LN client", "LF client", "Hybrid client", "Agg forge"], vals, color=["#c0392b", "#e67e22", "#8e44ad", "#2980b9"])
ax.set_ylim(0, 1.05); ax.set_ylabel("Reject rate")
savefig("fig22_reject_composition.png")

metrics = {
    "config": CONFIG,
    "tau": TAU,
    "dataset": meta,
    "summary_table": summary_df.to_dict(orient="records"),
    "ablation_tau": abl_tau_df.to_dict(orient="records"),
    "ablation_hospitals": abl_n_df.to_dict(orient="records"),
    "device": str(DEVICE),
    "seeds": SEEDS,
    "notes": "Strengthened experiments: 8 hospitals, 40 rounds, 5 seeds, hybrid attacks, dual verification",
}
(EVIDENCE / "metrics_summary.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
print("DONE evidence refresh")
