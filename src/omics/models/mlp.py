"""
MLP classifier for multi-omics — extracted verbatim from
`Code/MLP_benchmark.ipynb` (the agreed source of truth for the MLP network).

Public API:
  set_seed, OmicsClassifierMLP, FocalLoss, build_optimizer,
  step5_train_mlp, evaluate_mlp, train_and_evaluate
"""
from __future__ import annotations

import copy
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.utils.class_weight import compute_class_weight

try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except ImportError:
    _HAS_SMOTE = False


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class OmicsClassifierMLP(nn.Module):
    """MLP for multi-omics with in-network BatchNorm, input noise + dropout."""

    def __init__(self, input_dim, num_classes, hidden_dims=(128, 64),
                 dropout=0.30, input_dropout=0.10, input_noise=0.05):
        super().__init__()
        self.input_norm = nn.BatchNorm1d(input_dim)
        self.input_drop = nn.Dropout(input_dropout)
        self.input_noise = input_noise

        layers, prev = [], input_dim
        for i, h in enumerate(hidden_dims):
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout * (0.7 ** i)),
            ]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, num_classes)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_norm(x)
        if self.training and self.input_noise > 0:
            x = x + torch.randn_like(x) * self.input_noise
        x = self.input_drop(x)
        x = self.backbone(x)
        return self.head(x)


class FocalLoss(nn.Module):
    """Focal loss + class weights + label smoothing."""

    def __init__(self, gamma=1.5, weight=None, label_smoothing=0.05, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce = F.cross_entropy(
            inputs, targets, weight=self.weight,
            label_smoothing=self.label_smoothing, reduction="none",
        )
        pt = torch.exp(-ce).clamp(1e-6, 1.0)
        loss = ((1 - pt) ** self.gamma) * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_optimizer(model, lr, weight_decay):
    """Exclude bias/norm params from weight decay (AdamW best practice)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or "bias" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr,
    )


def step5_train_mlp(X_train_opt, y_train, num_classes,
                    epochs=250, batch_size=64, lr=1e-3, patience=30,
                    hidden_dims=(128, 64), dropout=0.30,
                    gamma=1.5, weight_decay=1e-4,
                    use_smote=False, use_class_weights=True,
                    val_size=0.15, seed=42, verbose=True):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_opt = np.asarray(X_train_opt, dtype=np.float32)
    y_train = np.asarray(y_train)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_opt, y_train, test_size=val_size, stratify=y_train, random_state=seed
    )

    class_weight = None
    if use_smote and _HAS_SMOTE:
        k = int(max(1, min(5, np.bincount(y_tr).min() - 1)))
        try:
            X_tr, y_tr = SMOTE(random_state=seed, k_neighbors=k).fit_resample(X_tr, y_tr)
            use_class_weights = False
        except Exception as e:
            if verbose:
                print(f"[SMOTE skipped] {e}")
    if use_class_weights:
        classes = np.unique(y_tr)
        cw = compute_class_weight("balanced", classes=classes, y=y_tr)
        full = np.ones(num_classes, dtype=np.float32)
        full[classes] = cw
        class_weight = torch.tensor(full, device=device)

    train_ds = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long())
    drop_last = len(train_ds) > batch_size
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=drop_last)

    X_val_t = torch.from_numpy(np.asarray(X_val, np.float32)).to(device)

    model = OmicsClassifierMLP(
        X_tr.shape[1], num_classes, hidden_dims=hidden_dims, dropout=dropout
    ).to(device)
    criterion = FocalLoss(gamma=gamma, weight=class_weight)
    optimizer = build_optimizer(model, lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs,
        steps_per_epoch=max(1, len(train_loader)), pct_start=0.1,
    )

    best_f1, wait = -1.0, 0
    best_state = copy.deepcopy(model.state_dict())
    min_delta = 1e-4

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).argmax(1).cpu().numpy()
        val_f1 = f1_score(y_val, val_pred, average="macro")

        if val_f1 > best_f1 + min_delta:
            best_f1, wait = val_f1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"Early stop @ epoch {epoch} | best macro-F1 = {best_f1:.4f}")
                break

    model.load_state_dict(best_state)
    return model


def evaluate_mlp(model, X_test, y_test) -> dict:
    device = next(model.parameters()).device
    X_test = torch.from_numpy(np.asarray(X_test, np.float32)).to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(X_test), dim=1).cpu().numpy()
    y_pred = probs.argmax(1)
    y_true = np.asarray(y_test)
    n_classes = probs.shape[1]

    try:
        if n_classes == 2:
            auc = roc_auc_score(y_true, probs[:, 1])
        else:
            auc = roc_auc_score(
                y_true, probs, multi_class="ovr", average="macro",
                labels=np.arange(n_classes),
            )
    except Exception:
        auc = np.nan

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "auc_macro": auc,
    }


def train_and_evaluate(X_train, y_train, X_test, y_test, num_classes, cfg: dict) -> dict:
    """Uniform model entry point used by the pipeline."""
    kwargs = dict(cfg or {})
    seed = int(kwargs.pop("seed", 42))
    if "hidden_dims" in kwargs and isinstance(kwargs["hidden_dims"], list):
        kwargs["hidden_dims"] = tuple(kwargs["hidden_dims"])
    model = step5_train_mlp(
        X_train_opt=X_train, y_train=y_train, num_classes=num_classes, seed=seed, **kwargs
    )
    return evaluate_mlp(model, X_test, y_test)
