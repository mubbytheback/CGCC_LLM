"""
Self-contained GNN solution for CGCC city-graph classification.
Runs on CPU, uses only allowed libraries + standard library.
"""
import os
import pickle
import random

import numpy as np
import pandas as pd
import networkx as nx

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import f1_score, accuracy_score

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
SEED = 42
DATA_DIR = "data"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
TRAIN_LABELS_CSV = os.path.join(DATA_DIR, "train_labels.csv")
SUBMISSION_CSV = os.path.join(DATA_DIR, "submission.csv")

DEVICE = torch.device("cpu")
HIDDEN = 64
DROPOUT = 0.35
LR = 0.005
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 400
PATIENCE = 25
N_FOLDS = 5

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

def reset_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(SEED)

# ------------------------------------------------------------------
# Mandatory graph loading function (exact copy)
# ------------------------------------------------------------------
def nx_to_pyg(path):
    with open(path, 'rb') as f:
        G = pickle.load(f)
    nodes = list(G.nodes())
    id_map = {n: i for i, n in enumerate(nodes)}
    edges = [(id_map[u], id_map[v]) for u, v, *_ in G.edges()
             if u in id_map and v in id_map]
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    x_list = [[G.nodes[n].get('x', 0.0),
               G.nodes[n].get('y', 0.0),
               float(G.degree(n))] for n in nodes]
    x = torch.tensor(x_list, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, num_nodes=len(nodes))

# ------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------
def build_node_features(G):
    """Baseline-style node features: centered/scaled x,y + normalized degree."""
    nodes = list(G.nodes())
    if len(nodes) == 0:
        return None
    xs = np.array([G.nodes[n].get("x", 0.0) for n in nodes], dtype=np.float32)
    ys = np.array([G.nodes[n].get("y", 0.0) for n in nodes], dtype=np.float32)
    xs = xs - xs.mean()
    ys = ys - ys.mean()
    scale = float(np.sqrt(xs.var() + ys.var()) + 1e-6)
    xs = xs / scale
    ys = ys / scale
    deg = np.array([G.degree(n) for n in nodes], dtype=np.float32)
    deg = (deg - deg.mean()) / (deg.std() + 1e-6)
    return np.stack([xs, ys, deg], axis=1)


def extract_graph_features(G):
    """Graph-level structural features."""
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    if n_nodes == 0:
        return np.zeros(12, dtype=np.float32)

    G_undir = G.to_undirected()
    degs = [G.degree(n) for n in G.nodes()]
    avg_deg = np.mean(degs)
    std_deg = np.std(degs)

    lengths = [data.get('length', 0.0) for _, _, data in G.edges(data=True)]
    if lengths:
        avg_len = np.mean(lengths)
        std_len = np.std(lengths)
    else:
        avg_len = std_len = 0.0

    density = nx.density(G_undir)
    try:
        clustering = nx.average_clustering(G_undir)
    except Exception:
        clustering = 0.0

    # Edge orientation histogram (4 bins: 0-45, 45-90, 90-135, 135-180)
    angles = []
    for u, v, data in G.edges(data=True):
        xu, yu = G.nodes[u].get('x', 0.0), G.nodes[u].get('y', 0.0)
        xv, yv = G.nodes[v].get('x', 0.0), G.nodes[v].get('y', 0.0)
        dx, dy = xv - xu, yv - yu
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue
        angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
        angles.append(angle)

    if angles:
        hist, _ = np.histogram(angles, bins=[0, 22.5, 67.5, 112.5, 180])
        hist = hist / (hist.sum() + 1e-6)
    else:
        hist = np.zeros(4, dtype=np.float32)

    # Degree distribution entropy
    deg_counts = np.bincount(degs, minlength=1)
    deg_probs = deg_counts / deg_counts.sum()
    deg_entropy = -np.sum(deg_probs * np.log(deg_probs + 1e-10))

    feats = np.array([
        n_nodes, n_edges, avg_deg, std_deg,
        avg_len, std_len, density, clustering,
        deg_entropy, hist[0], hist[1], hist[2]
    ], dtype=np.float32)

    return feats


# ------------------------------------------------------------------
# Dataset builders
# ------------------------------------------------------------------
def build_dataset(graph_dir, label_map=None):
    files = sorted([f for f in os.listdir(graph_dir) if f.endswith('.pkl')])
    data_list = []
    labels = []
    for fn in files:
        if label_map is not None and fn not in label_map:
            continue
        path = os.path.join(graph_dir, fn)

        with open(path, 'rb') as f:
            G = pickle.load(f)

        if G.number_of_nodes() == 0:
            continue

        nodes = list(G.nodes())
        id_map = {n: i for i, n in enumerate(nodes)}
        edges = [(id_map[u], id_map[v]) for u, v, *_ in G.edges()
                 if u in id_map and v in id_map]
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        node_feats = build_node_features(G)
        if node_feats is None:
            continue
        x = torch.tensor(node_feats, dtype=torch.float)

        graph_feats = extract_graph_features(G)
        g = torch.tensor(graph_feats, dtype=torch.float)

        data = Data(x=x, edge_index=edge_index, num_nodes=len(nodes), g=g)
        data.filename = fn
        data_list.append(data)
        if label_map is not None:
            labels.append(int(label_map[fn]))
    return data_list, labels


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
class CityGNN(nn.Module):
    def __init__(self, node_in=3, hidden=64, num_classes=3, dropout=0.35, graph_feat_dim=12):
        super().__init__()
        self.conv1 = GCNConv(node_in, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.dropout = dropout
        # Classifier takes pooled graph embedding + graph-level features
        self.classifier = nn.Linear(hidden * 2 + graph_feat_dim, num_classes)

    def forward(self, data):
        x, edge_index, g = data.x, data.edge_index, data.g

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        g_mean = x.mean(dim=0)
        g_max = x.max(dim=0).values
        graph_emb = torch.cat([g_mean, g_max, g], dim=0)

        return self.classifier(graph_emb)


# ------------------------------------------------------------------
# Training helpers (process graphs individually like baseline)
# ------------------------------------------------------------------
def train_epoch(model, train_data, optimizer, criterion):
    model.train()
    total_loss = 0.0
    perm = np.random.permutation(len(train_data))
    for idx in perm:
        data = train_data[idx]
        optimizer.zero_grad()
        logits = model(data).unsqueeze(0)
        loss = criterion(logits, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / len(train_data)


@torch.no_grad()
def evaluate(model, data_list, criterion):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    for data in data_list:
        logits = model(data).unsqueeze(0)
        loss = criterion(logits, data.y)
        total_loss += float(loss.item())
        pred = int(torch.argmax(logits, dim=1).item())
        y_true.append(int(data.y.item()))
        y_pred.append(pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return total_loss / len(data_list), acc, f1


# ------------------------------------------------------------------
# Single fold training
# ------------------------------------------------------------------
def train_one_fold(train_data, val_data, fold_idx):
    reset_seeds(SEED + fold_idx)

    labels = [d.y.item() for d in train_data]
    counts = np.bincount(np.array(labels), minlength=3)
    weights = counts.sum() / (counts + 1e-6)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32)

    model = CityGNN(node_in=3, hidden=HIDDEN, num_classes=3, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_f1 = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(model, train_data, optimizer, criterion)
        val_loss, val_acc, val_f1 = evaluate(model, val_data, criterion)

        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_f1


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("Loading data...")
    labels_df = pd.read_csv(TRAIN_LABELS_CSV)
    label_map = dict(zip(labels_df["filename"], labels_df["target"]))

    all_data, all_labels = build_dataset(TRAIN_DIR, label_map)
    test_data, _ = build_dataset(TEST_DIR)

    print(f"Train graphs: {len(all_data)}")
    print(f"Test graphs:  {len(test_data)}")

    for d, lbl in zip(all_data, all_labels):
        d.y = torch.tensor([lbl], dtype=torch.long)

    # K-Fold Cross Validation
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_models = []
    fold_f1s = []

    print(f"\nTraining {N_FOLDS}-fold CV...")
    for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(all_data, all_labels)):
        tr_data = [all_data[i] for i in tr_idx]
        val_data = [all_data[i] for i in val_idx]
        model, best_f1 = train_one_fold(tr_data, val_data, fold_idx)
        fold_models.append(model)
        fold_f1s.append(best_f1)
        print(f"  Fold {fold_idx+1}: best val macro_f1={best_f1:.4f}")

    print(f"\nMean val macro_f1 across folds: {np.mean(fold_f1s):.4f} (+/- {np.std(fold_f1s):.4f})")

    # Retrain on full training data
    print("\nRetraining on full training set...")
    reset_seeds(SEED)
    labels = [d.y.item() for d in all_data]
    counts = np.bincount(np.array(labels), minlength=3)
    weights = counts.sum() / (counts + 1e-6)
    weights = weights / weights.mean()
    class_weights = torch.tensor(weights, dtype=torch.float32)

    final_model = CityGNN(node_in=3, hidden=HIDDEN, num_classes=3, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(final_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Internal validation split for early stopping
    tr_idx, val_idx = train_test_split(
        list(range(len(all_data))), test_size=0.15, stratify=all_labels, random_state=SEED
    )
    tr_data = [all_data[i] for i in tr_idx]
    val_data = [all_data[i] for i in val_idx]

    best_f1 = -1.0
    best_state = None
    bad_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = train_epoch(final_model, tr_data, optimizer, criterion)
        val_loss, val_acc, val_f1 = evaluate(final_model, val_data, criterion)

        if val_f1 > best_f1 + 1e-4:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in final_model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    final_model.load_state_dict(best_state)
    print(f"Final validation -> acc={val_acc:.4f}, macro_f1={best_f1:.4f}")

    # Ensemble prediction: average logits
    print("\nPredicting on test set with ensemble...")
    all_logits = []
    with torch.no_grad():
        final_model.eval()
        all_logits.append(torch.stack([final_model(d) for d in test_data]))

        for model in fold_models:
            model.eval()
            all_logits.append(torch.stack([model(d) for d in test_data]))

    avg_logits = torch.stack(all_logits).mean(dim=0)
    test_preds = torch.argmax(avg_logits, dim=1).cpu().numpy()

    pred_rows = []
    for i, data in enumerate(test_data):
        pred_rows.append({"filename": data.filename, "prediction": int(test_preds[i])})

    submission = pd.DataFrame(pred_rows).sort_values("filename")
    submission.to_csv(SUBMISSION_CSV, index=False)
    print(f"Wrote submission to {SUBMISSION_CSV}")
    print(submission.head())

    print("\nPrediction distribution:")
    print(submission['prediction'].value_counts().sort_index())


if __name__ == "__main__":
    main()
