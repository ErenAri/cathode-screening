import json
import argparse
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import MultiStepLR
from pymatgen.core import Composition

# ---------------------------------------------------------------------------
# Model-agnostic forward helpers
# ---------------------------------------------------------------------------

def _model_forward(model, batch_data, model_type, debug=False):
    """Unified forward call for CGCNN and MACE models."""
    if model_type == "mace":
        return model(batch_data, debug=debug)
    # CGCNN path
    x, src, dst, e, b = batch_data
    return model(x, src, dst, e, b, debug=debug)


def _unpack_batch(batch, model_type, device):
    """Unpack a DataLoader batch -> (batch_data, y) on *device*.

    For CGCNN: batch is (x, src, dst, e, batch_index, y)
    For MACE:  batch is (data_dict, y)
    """
    if model_type == "mace":
        data_dict, y = batch
        data_dict = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in data_dict.items()
        }
        return data_dict, y.to(device)
    # CGCNN
    x, src, dst, e, b, y = batch
    return (
        (x.to(device), src.to(device), dst.to(device), e.to(device), b.to(device)),
        y.to(device),
    )

# Stability thresholds (eV/atom)
THRESH_STABLE = 0.05      # Ehull < 0.05 = stable (KEEP)
THRESH_METASTABLE = 0.10  # Ehull < 0.10 = metastable (MAYBE)


class Normalizer:
    """Normalize targets to zero mean and unit variance (like original CGCNN)."""
    def __init__(self, tensor):
        self.mean = torch.mean(tensor).item()
        self.std = torch.std(tensor).item()
        if self.std < 1e-8:
            self.std = 1.0

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


def load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    try:
        from cathode_screening.common.config import validate_training_config
        issues = validate_training_config(cfg)
        for issue in issues:
            print(f"[config warning] {issue}")
    except ImportError:
        pass
    return cfg


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def composition_fingerprint(formula: str):
    vocab = ["Li", "O", "Fe", "Mn", "Co", "Ni", "Ti", "V", "Cr"]
    c = Composition(formula)
    total = c.num_atoms if c.num_atoms else 1.0
    v = np.zeros(len(vocab), dtype=np.float32)
    for i, el in enumerate(vocab):
        v[i] = float(c.get_el_amt_dict().get(el, 0.0)) / float(total)
    return v


def make_cluster_splits(df: pd.DataFrame, seed: int, train_frac: float, val_frac: float, n_clusters: int):
    X = np.stack([composition_fingerprint(f) for f in df["formula_pretty"]], axis=0)
    X = normalize(X)
    unique = np.unique(X, axis=0).shape[0]
    k = min(n_clusters, unique)
    km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
    c = km.fit_predict(X)
    df = df.copy()
    df["cluster_id"] = c

    clusters = df["cluster_id"].unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(clusters)

    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ids, val_ids, test_ids = [], [], []
    counts = {"train": 0, "val": 0, "test": 0}

    for cid in clusters:
        mids = df.loc[df["cluster_id"] == cid, "material_id"].tolist()
        if counts["train"] < n_train:
            train_ids += mids
            counts["train"] += len(mids)
        elif counts["val"] < n_val:
            val_ids += mids
            counts["val"] += len(mids)
        else:
            test_ids += mids
            counts["test"] += len(mids)

    return {"train": train_ids, "val": val_ids, "test": test_ids}


class GraphNPZDataset(Dataset):
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df.reset_index(drop=True)
        self.target_col = target_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        r = self.df.iloc[idx]
        z = np.load(r["graph_npz"])
        x = torch.from_numpy(z["x"])
        src = torch.from_numpy(z["edge_src"]).long()
        dst = torch.from_numpy(z["edge_dst"]).long()
        e = torch.from_numpy(z["e_attr"])
        y = torch.tensor(float(r[self.target_col]), dtype=torch.float32)
        return x, src, dst, e, y


def collate(batch):
    xs, srcs, dsts, es, ys = zip(*batch)
    x = torch.cat(xs, dim=0)
    y = torch.stack(ys, dim=0)

    node_offsets = []
    off = 0
    for xi in xs:
        node_offsets.append(off)
        off += xi.shape[0]

    src = []
    dst = []
    for i, (s, d) in enumerate(zip(srcs, dsts)):
        o = node_offsets[i]
        src.append(s + o)
        dst.append(d + o)
    src = torch.cat(src, dim=0)
    dst = torch.cat(dst, dim=0)

    e = torch.cat(es, dim=0)

    batch_index = []
    for i, xi in enumerate(xs):
        batch_index.append(torch.full((xi.shape[0],), i, dtype=torch.long))
    batch_index = torch.cat(batch_index, dim=0)

    return x, src, dst, e, batch_index, y


class CGCNNBlock(nn.Module):
    """CGCNN convolution block with skip connection for better gradient flow."""
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()
        self.lin_msg = nn.Linear(node_dim * 2 + edge_dim, node_dim)
        self.lin_upd = nn.Linear(node_dim * 2, node_dim)
        self.bn = nn.BatchNorm1d(node_dim)  # Batch norm for stability

    def forward(self, x, src, dst, e_attr):
        identity = x  # Skip connection
        m_in = torch.cat([x[src], x[dst], e_attr], dim=-1)
        m = torch.tanh(self.lin_msg(m_in))
        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, m)
        u_in = torch.cat([x, agg], dim=-1)
        x = self.bn(self.lin_upd(u_in))
        x = F.relu(x + identity)  # Skip connection with ReLU after
        return x


class CGCNN(nn.Module):
    """Crystal Graph CNN with multi-head output for regression + classification."""
    def __init__(self, node_in: int, node_dim: int, edge_dim: int, layers: int, dropout: float, pooling: str):
        super().__init__()
        self.embed = nn.Linear(node_in, node_dim)
        self.blocks = nn.ModuleList([CGCNNBlock(node_dim, edge_dim) for _ in range(layers)])
        self.dropout = dropout
        self.pooling = pooling
        self.pool_attn = nn.Linear(node_dim, 1) if pooling == "attn" else None
        
        # Shared hidden layer before heads
        self.fc_hidden = nn.Linear(node_dim, node_dim)
        
        # ORDERED Quantile regression: predict q50 + deltas to guarantee q10 <= q50 <= q90
        # This avoids quantile crossing which breaks decision logic
        self.head_q50 = nn.Linear(node_dim, 1)    # Median (point estimate)
        self.head_d_lo = nn.Linear(node_dim, 1)   # Delta below median (will use softplus)
        self.head_d_hi = nn.Linear(node_dim, 1)   # Delta above median (will use softplus)
        
        # Initialize delta biases to reasonable starting values
        # softplus(0) = 0.693, so bias=-1 gives softplus(-1)=0.31 ≈ reasonable starting interval
        nn.init.constant_(self.head_d_lo.bias, -1.0)
        nn.init.constant_(self.head_d_hi.bias, -1.0)
        
        # Classification heads (stability probabilities) - auxiliary
        self.head_p_stable = nn.Linear(node_dim, 1)      # P(Ehull < 0.05)
        self.head_p_metastable = nn.Linear(node_dim, 1)  # P(Ehull < 0.10)

    def pool(self, x, batch_index):
        batch_index = batch_index.long()
        n_graphs = int(batch_index.max().item()) + 1
        
        if self.pooling == "attn":
            # Attention scores
            w = self.pool_attn(x).squeeze(-1)  # [num_nodes]
            
            # Per-graph softmax with -inf initialization (robust!)
            # This guarantees at least one node per graph has exp(0)=1
            w_max = torch.full((n_graphs,), -torch.inf, device=x.device, dtype=w.dtype)
            w_max.scatter_reduce_(0, batch_index, w, reduce="amax", include_self=True)
            
            w_exp = torch.exp(w - w_max[batch_index])
            denom = torch.zeros(n_graphs, device=x.device, dtype=w.dtype)
            denom.index_add_(0, batch_index, w_exp)
            
            alpha = w_exp / denom[batch_index].clamp_min(1e-12)
            
            out = torch.zeros((n_graphs, x.size(-1)), device=x.device, dtype=x.dtype)
            out.index_add_(0, batch_index, x * alpha.unsqueeze(-1))
            return out
        else:
            # Mean pooling per graph
            out = torch.zeros((n_graphs, x.size(-1)), device=x.device, dtype=x.dtype)
            out.index_add_(0, batch_index, x)
            counts = torch.zeros(n_graphs, device=x.device, dtype=x.dtype)
            counts.index_add_(0, batch_index, torch.ones(x.size(0), device=x.device, dtype=x.dtype))
            return out / counts.unsqueeze(-1).clamp_min(1e-8)

    def forward(self, x, src, dst, e_attr, batch_index, debug=False):
        batch_index = batch_index.long()
        x = self.embed(x)
        
        if debug:
            n_graphs = int(batch_index.max().item()) + 1
            unique_graphs = batch_index.unique().numel()
            print(f"  [DEBUG] n_graphs={n_graphs}, unique_graphs={unique_graphs}, batch_index range=[{batch_index.min()},{batch_index.max()}]")
            print(f"  [DEBUG] After embed: x.shape={x.shape}, x.std()={x.std().item():.6f}")
        
        for i, blk in enumerate(self.blocks):
            x = blk(x, src, dst, e_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if debug:
                print(f"  [DEBUG] After block {i}: x.std()={x.std().item():.6f}")
        
        g = self.pool(x, batch_index)
        
        if debug:
            print(f"  [DEBUG] After pool: g.shape={g.shape}")
            print(f"  [DEBUG] g.std()={g.std().item():.6f}, g.var(dim=0).mean()={g.var(dim=0).mean().item():.6f}")
            print(f"  [DEBUG] g[:5, :3]=\n{g[:5, :3]}")  # First 5 graphs, first 3 features
        
        # Shared hidden
        h = F.relu(self.fc_hidden(g))
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        if debug:
            print(f"  [DEBUG] After fc_hidden: h.std()={h.std().item():.6f}")
        
        # ORDERED Quantile outputs: q50 + softplus deltas
        # This GUARANTEES q10 <= q50 <= q90 (no crossing!)
        q50 = self.head_q50(h).squeeze(-1)
        d_lo = F.softplus(self.head_d_lo(h).squeeze(-1))  # Always >= 0
        d_hi = F.softplus(self.head_d_hi(h).squeeze(-1))  # Always >= 0
        
        q10 = q50 - d_lo  # 10th percentile = median - positive delta
        q90 = q50 + d_hi  # 90th percentile = median + positive delta
        
        if debug:
            print(f"  [DEBUG] q50: shape={q50.shape}, min={q50.min().item():.4f}, max={q50.max().item():.4f}")
            print(f"  [DEBUG] d_lo: min={d_lo.min().item():.4f}, max={d_lo.max().item():.4f}, mean={d_lo.mean().item():.4f}")
            print(f"  [DEBUG] d_hi: min={d_hi.min().item():.4f}, max={d_hi.max().item():.4f}, mean={d_hi.mean().item():.4f}")
            print(f"  [DEBUG] First 5 (q10, q50, q90): {list(zip(q10[:5].tolist(), q50[:5].tolist(), q90[:5].tolist()))}")
        
        # Classification outputs (logits) - auxiliary
        logit_stable = self.head_p_stable(h).squeeze(-1)
        logit_metastable = self.head_p_metastable(h).squeeze(-1)
        
        return q10, q50, q90, logit_stable, logit_metastable


def gaussian_nll(mu, log_sigma, y, min_log_sigma=-4.0, max_log_sigma=0.0):
    """
    Gaussian negative log-likelihood (heteroscedastic).
    
    Note: max_log_sigma clamped to 0.0 (σ_max = 1.0 in normalized space)
    to prevent the model from "buying" loss reduction by inflating σ globally.
    In denormalized space with std~0.37, this means σ_max ≈ 0.37 eV.
    """
    log_sigma = torch.clamp(log_sigma, min=min_log_sigma, max=max_log_sigma)
    sigma2 = torch.exp(2.0 * log_sigma)
    return torch.mean((y - mu) ** 2 / (2.0 * sigma2) + log_sigma)


def huber_loss(mu, y, delta=0.1):
    """Huber loss - robust to outliers."""
    return F.huber_loss(mu, y, reduction='mean', delta=delta)


def pinball_loss(pred, target, tau):
    """
    Pinball (quantile) loss for quantile regression.
    
    tau=0.1 -> 10th percentile (optimistic)
    tau=0.5 -> median
    tau=0.9 -> 90th percentile (pessimistic)
    
    Loss = (tau - 1{y < pred}) * (y - pred)
    """
    diff = target - pred
    return torch.mean(torch.max(tau * diff, (tau - 1) * diff))


def quantile_loss(q10, q50, q90, y):
    """
    Combined quantile loss for (q10, q50, q90) predictions.
    
    No crossing penalty needed - architecture guarantees q10 <= q50 <= q90!
    
    Includes interval width regularization to encourage tighter predictions,
    which is critical for getting KEEPs (need q90 < 0.05).
    """
    loss_q10 = pinball_loss(q10, y, tau=0.10)
    loss_q50 = pinball_loss(q50, y, tau=0.50)
    loss_q90 = pinball_loss(q90, y, tau=0.90)
    
    # STRONGER interval width regularization (increased from 0.01 to 0.05)
    # Target width ~0.08 in normalized space (≈0.03 eV) for confident decisions
    interval_width = q90 - q10
    width_reg = 0.05 * torch.mean(F.relu(interval_width - 0.5))  # Penalize if width > 0.5 normalized
    
    return loss_q10 + loss_q50 + loss_q90 + width_reg


def residual_sigma_loss(log_sigma, mu, y, normalizer, min_log_sigma=-4.0, max_log_sigma=2.0):
    """
    Train sigma to predict absolute residual |y - mu| in NORMALIZED space.
    
    This avoids Gaussian NLL's tendency to inflate sigma globally.
    Instead, sigma learns to be small where mu is accurate and large where it's not.
    
    IMPORTANT: Works in normalized space so sigma head learns meaningful values,
    not just dataset_std.
    """
    log_sigma = torch.clamp(log_sigma, min=min_log_sigma, max=max_log_sigma)
    sigma = torch.exp(log_sigma)  # In normalized space
    
    # Target: absolute residual in NORMALIZED space
    # mu is already normalized, y needs to be normalized
    y_norm = normalizer.norm(y) if normalizer is not None else y
    residual = torch.abs(y_norm - mu.detach())  # Detach mu so gradients only flow to sigma head
    
    # Clamp residual to encode physical resolution limits (in normalized space):
    # - min=0.05: ~0.02 eV / 0.366 std ≈ 0.05 normalized (DFT error floor)
    # - max=1.5: ~0.55 eV / 0.366 std ≈ 1.5 normalized (reasonable uncertainty cap)
    residual = torch.clamp(residual, min=0.05, max=1.5)
    
    # L2 loss between predicted sigma and actual residual
    loss = F.mse_loss(sigma, residual)
    
    return loss


def focal_bce_loss(logits, labels, gamma=2.0, alpha=0.25):
    """Focal loss - down-weights easy examples, focuses on hard ones."""
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    pt = torch.exp(-bce)  # pt = p if y=1, 1-p if y=0
    focal_weight = (alpha * labels + (1 - alpha) * (1 - labels)) * (1 - pt) ** gamma
    return (focal_weight * bce).mean()


def compute_multi_task_loss(
    q10, q50, q90, logit_stable, logit_metastable, y, y_raw,
    cls_weight=0.1
):
    """
    Multi-task loss with QUANTILE REGRESSION (no more σ collapse!):
    - Quantile loss: pinball loss for q10, q50, q90 with crossing penalty
    - Classification: Focal BCE for auxiliary stability heads
    
    Why quantile regression wins:
    - No Gaussian assumption (heavy-tailed data is fine)
    - Direct interval outputs for decisions (q90 < 0.05 -> KEEP)
    - No σ collapse / inflation issues
    - Median (q50) serves as point estimate for MAE
    """
    # Quantile loss (primary) - works in normalized space
    loss_quantile = quantile_loss(q10, q50, q90, y)
    
    # Classification labels from raw targets
    label_stable = (y_raw < THRESH_STABLE).float()
    label_metastable = (y_raw < THRESH_METASTABLE).float()
    
    # Focal loss for classification heads (auxiliary)
    loss_cls_stable = focal_bce_loss(logit_stable, label_stable, gamma=2.0, alpha=0.75)
    loss_cls_meta = focal_bce_loss(logit_metastable, label_metastable, gamma=2.0, alpha=0.6)
    
    # Combined loss
    loss = loss_quantile + cls_weight * (loss_cls_stable + loss_cls_meta)
    
    return loss, {
        'loss_quantile': loss_quantile.item(),
        'loss_cls_stable': loss_cls_stable.item(),
        'loss_cls_meta': loss_cls_meta.item()
    }


def compute_ece(probs, labels, n_bins=10):
    """Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() > 0:
            avg_conf = probs[mask].mean()
            avg_acc = labels[mask].mean()
            ece += mask.sum() * np.abs(avg_conf - avg_acc)
    return ece / len(probs) if len(probs) > 0 else 0.0


class TemperatureScaling(nn.Module):
    """Post-hoc temperature scaling for calibration."""
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits):
        return logits / self.temperature
    
    def calibrate(self, logits, labels, lr=0.01, max_iter=50):
        """Learn optimal temperature on validation set."""
        self.temperature.data.fill_(1.0)
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def closure():
            optimizer.zero_grad()
            scaled = self.forward(logits)
            loss = F.binary_cross_entropy_with_logits(scaled, labels)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        return self.temperature.item()


def compute_decision_quantile(q10, q50, q90, p_stable=None):
    """
    Decision policy using ORDERED QUANTILE INTERVALS + classification confidence.
    
    Two-tier KEEP decisions:
    - KEEP_STRICT: q90 < 0.05 AND p_stable > 0.8 (100% confident stable)
    - KEEP_RELAXED: q90 < 0.10 AND p_stable > 0.7 (high confidence stable)
    
    KILL decisions:
    - KILL: q10 > 0.10 (even optimistic bound is unstable)
    
    The p_stable conjunction reduces false KEEP/KILL when regression is uncertain.
    
    Args:
        q10: 10th percentile predictions (optimistic, in eV)
        q50: median predictions (point estimate, in eV)  
        q90: 90th percentile predictions (pessimistic, in eV)
        p_stable: probability of stable (y < 0.05) from classification head
    """
    decisions = []
    for i in range(len(q50)):
        # Get p_stable if available, else assume neutral 0.5
        ps = p_stable[i] if p_stable is not None else 0.5
        
        # KEEP_STRICT: pessimistic bound very stable AND classifier very confident
        if q90[i] < THRESH_STABLE and ps > 0.8:  # 0.05 eV
            decisions.append('KEEP')
        # KEEP_RELAXED: pessimistic bound below metastable AND classifier confident
        elif q90[i] < THRESH_METASTABLE and ps > 0.7:  # 0.10 eV
            decisions.append('KEEP')  # Still KEEP, but looser criteria
        # KILL: optimistic bound unstable (no p_stable needed - regression is confident)
        elif q10[i] > THRESH_METASTABLE:  # 0.10 eV
            decisions.append('KILL')
        else:
            decisions.append('MAYBE')
    
    return decisions


@torch.no_grad()
def evaluate(model, loader, device, normalizer=None, model_type="cgcnn"):
    """Comprehensive evaluation with quantile regression outputs."""
    model.eval()
    ys = []
    q10s, q50s, q90s = [], [], []
    p_stables = []
    p_metas = []

    for batch in loader:
        batch_data, y = _unpack_batch(batch, model_type, device)
        q10, q50, q90, logit_stable, logit_meta = _model_forward(
            model, batch_data, model_type
        )
        
        # Denormalize quantile predictions
        if normalizer is not None:
            q10 = normalizer.denorm(q10)
            q50 = normalizer.denorm(q50)
            q90 = normalizer.denorm(q90)
        
        ys.append(y.detach().cpu().numpy())
        q10s.append(q10.detach().cpu().numpy())
        q50s.append(q50.detach().cpu().numpy())
        q90s.append(q90.detach().cpu().numpy())
        p_stables.append(torch.sigmoid(logit_stable).detach().cpu().numpy())
        p_metas.append(torch.sigmoid(logit_meta).detach().cpu().numpy())
    
    y = np.concatenate(ys)
    q10 = np.concatenate(q10s)
    q50 = np.concatenate(q50s)  # This is our point estimate (median)
    q90 = np.concatenate(q90s)
    p_stable = np.concatenate(p_stables)
    p_meta = np.concatenate(p_metas)
    
    # Regression metrics (using q50 as point estimate)
    mu = q50  # Median is our point estimate
    err = np.abs(y - mu)
    mae = float(np.mean(err))
    rmse = float(np.sqrt(np.mean((y - mu) ** 2)))
    p95 = float(np.percentile(err, 95))
    p99 = float(np.percentile(err, 99))
    maxe = float(np.max(err))
    
    # Interval width (uncertainty proxy)
    interval_width = q90 - q10
    
    # Quantile calibration: what fraction of y falls within [q10, q90]?
    # Should be ~80% for well-calibrated (10th to 90th percentile)
    coverage_80 = float(np.mean((y >= q10) & (y <= q90)))
    
    # Classification metrics
    label_stable = (y < THRESH_STABLE).astype(float)
    label_meta = (y < THRESH_METASTABLE).astype(float)
    
    # Accuracy for classifier heads
    pred_stable = (p_stable >= 0.5).astype(float)
    pred_meta = (p_meta >= 0.5).astype(float)
    acc_stable = float(np.mean(pred_stable == label_stable))
    acc_meta = float(np.mean(pred_meta == label_meta))
    
    # Precision@K and Recall@K for stable (product metrics!)
    # RANK BY q90 (risk score): lower q90 = more confidently stable
    # This is better than q50 because it considers the pessimistic bound
    n_true_stable = int(label_stable.sum())
    prec_at_k_dict = {}
    recall_at_k_dict = {}
    for k in [25, 100, 250]:
        if k <= len(q90):
            top_k_idx = np.argsort(q90)[:k]  # lowest q90 = most confidently stable
            prec_at_k = label_stable[top_k_idx].mean()
            recall_at_k = label_stable[top_k_idx].sum() / n_true_stable if n_true_stable > 0 else 0
            prec_at_k_dict[k] = float(prec_at_k)
            recall_at_k_dict[k] = float(recall_at_k)
        else:
            prec_at_k_dict[k] = 0.0
            recall_at_k_dict[k] = 0.0
    prec_at_25 = prec_at_k_dict.get(25, 0.0)
    recall_at_25 = recall_at_k_dict.get(25, 0.0)
    
    # ECE (calibration for classifier heads)
    ece_stable = compute_ece(p_stable, label_stable)
    ece_meta = compute_ece(p_meta, label_meta)
    
    # False-stable rate
    pred_stable_mask = pred_stable == 1
    if pred_stable_mask.sum() > 0:
        false_stable_rate = float(np.mean(y[pred_stable_mask] > THRESH_METASTABLE))
    else:
        false_stable_rate = 0.0
    
    # Spearman correlation (ranking quality)
    spearman_r, _ = stats.spearmanr(mu, y)
    
    # QUANTILE-BASED DECISIONS with p_stable conjunction
    decisions = compute_decision_quantile(q10, q50, q90, p_stable=p_stable)
    n_keep = decisions.count('KEEP')
    n_maybe = decisions.count('MAYBE')
    n_kill = decisions.count('KILL')
    
    # Sanity check
    assert n_keep + n_maybe + n_kill == len(y), f"Partition error: {n_keep}+{n_maybe}+{n_kill} != {len(y)}"
    
    # Quality of KEEP/KILL decisions
    keep_mask = np.array([d == 'KEEP' for d in decisions])
    kill_mask = np.array([d == 'KILL' for d in decisions])
    
    keep_precision = float(np.mean(y[keep_mask] < THRESH_STABLE)) if keep_mask.sum() > 0 else 0.0
    kill_precision = float(np.mean(y[kill_mask] >= THRESH_METASTABLE)) if kill_mask.sum() > 0 else 0.0
    
    # Ground truth counts
    n_true_stable = int(label_stable.sum())
    n_true_meta = int(label_meta.sum())
    
    return {
        # Regression (using q50 as point estimate)
        "mae": mae, 
        "rmse": rmse, 
        "p95_abs_err": p95, 
        "p99_abs_err": p99, 
        "max_abs_err": maxe,
        "spearman": float(spearman_r),
        # Quantile-specific
        "coverage_80": coverage_80,  # Should be ~0.80 if calibrated
        "interval_width_mean": float(np.mean(interval_width)),
        "interval_width_p50": float(np.percentile(interval_width, 50)),
        "q10_p50": float(np.percentile(q10, 50)),
        "q50_p50": float(np.percentile(q50, 50)),
        "q90_p50": float(np.percentile(q90, 50)),
        # Classification (auxiliary)
        "acc_stable": acc_stable,
        "acc_metastable": acc_meta,
        "ece_stable": ece_stable,
        "ece_metastable": ece_meta,
        "false_stable_rate": false_stable_rate,
        # Product metrics (prec@K, recall@K at K=25,100,250)
        "prec_at_25": prec_at_25,
        "recall_at_25": recall_at_25,
        "prec_at_100": prec_at_k_dict.get(100, 0.0),
        "recall_at_100": recall_at_k_dict.get(100, 0.0),
        "prec_at_250": prec_at_k_dict.get(250, 0.0),
        "recall_at_250": recall_at_k_dict.get(250, 0.0),
        # Decisions (quantile-based!)
        "n_keep": n_keep,
        "n_maybe": n_maybe,
        "n_kill": n_kill,
        "keep_pct": n_keep / len(y) * 100,
        "kill_pct": n_kill / len(y) * 100,
        "keep_precision": keep_precision,
        "kill_precision": kill_precision,
        # Ground truth
        "n_true_stable": n_true_stable,
        "n_true_meta": n_true_meta,
    }


def create_balanced_sampler(df: pd.DataFrame, target_col: str, n_bins: int = 10):
    """
    Create a weighted sampler that balances samples across Ehull bins.
    This prevents the model from being dominated by near-zero samples.
    """
    targets = df[target_col].values
    
    # Create bins - MUST be strictly increasing, no duplicates!
    # Use unique edges covering the full range
    bin_edges = np.array([
        0.0,    # Start
        0.02,   # Very stable
        0.05,   # Stable threshold (KEEP)
        0.10,   # Metastable threshold (MAYBE)
        0.20,   # Moderately unstable
        0.50,   # Unstable
        np.inf  # Very unstable (KILL)
    ])
    
    bin_indices = np.digitize(targets, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, len(bin_edges) - 2)  # Ensure valid range
    bin_counts = np.bincount(bin_indices, minlength=len(bin_edges)-1)
    
    # Weight = 1 / sqrt(bin_count) - softer than 1/count to reduce oscillation
    bin_weights = 1.0 / np.sqrt(bin_counts + 1)  # sqrt for gentler rebalancing
    sample_weights = bin_weights[bin_indices]
    
    # Normalize weights
    sample_weights = sample_weights / sample_weights.sum()
    
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(df),
        replacement=True
    )
    
    print(f"Balanced sampling: {len(bin_edges)-1} bins, counts={bin_counts.tolist()}")
    return sampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    seed = int(cfg["train"]["seed"])
    set_seed(seed)

    processed_dir = Path(cfg["data"]["processed_dir"])
    ds_name = cfg["data"]["dataset_name"]
    processed_path = processed_dir / f"processed_{ds_name}.parquet"
    df = pd.read_parquet(processed_path)

    splits_manifest = Path(cfg["data"]["splits_manifest"])
    if splits_manifest.exists():
        splits = json.loads(splits_manifest.read_text(encoding="utf-8"))
    else:
        train_frac = float(cfg.get("splits", {}).get("train_frac", 0.70))
        val_frac = float(cfg.get("splits", {}).get("val_frac", 0.15))
        n_clusters = int(cfg.get("splits", {}).get("n_clusters", 200))
        splits = make_cluster_splits(df, seed=seed, train_frac=train_frac, val_frac=val_frac, n_clusters=n_clusters)

    df_train = df[df["material_id"].isin(set(splits["train"]))].reset_index(drop=True)
    df_val = df[df["material_id"].isin(set(splits["val"]))].reset_index(drop=True)
    df_test = df[df["material_id"].isin(set(splits["test"]))].reset_index(drop=True)

    target = cfg["data"]["target"]["name"]
    model_type = str(cfg["model"].get("type", "cgcnn"))

    # --- Device + model creation (before datasets, so MACE z_table is available) ---
    device = torch.device("cuda" if torch.cuda.is_available() and cfg["train"]["device"] in ["auto", "cuda"] else "cpu")
    print(f"Using device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    if model_type == "mace":
        from cathode_screening.models.mace_finetune.model import MACEFineTuner
        model = MACEFineTuner.from_pretrained(
            model_name=str(cfg["model"].get("backbone", "medium")),
            head_dim=int(cfg["model"].get("head_dim", 128)),
            dropout=float(cfg["model"].get("dropout", 0.1)),
            freeze_backbone=bool(cfg["model"].get("freeze_backbone", True)),
            unfreeze_last_n=int(cfg["model"].get("unfreeze_last_n", 1)),
            device=str(device),
        ).to(device)
    else:
        node_in = int(cfg["model"].get("node_in_dim", 6))
        node_dim = int(cfg["model"]["node_embed_dim"])
        edge_dim = int(cfg["model"]["edge_rbf_bins"])
        layers = int(cfg["model"]["message_passing_layers"])
        dropout = float(cfg["model"]["dropout"])
        pooling = str(cfg["model"]["pooling"])
        model = CGCNN(node_in=node_in, node_dim=node_dim, edge_dim=edge_dim, layers=layers, dropout=dropout, pooling=pooling).to(device)

    # --- Dataset + collate depend on model type ---
    if model_type == "mace":
        from cathode_screening.models.mace_finetune.data import (
            StructureDataset,
            mace_collate,
        )
        structure_dir = cfg["data"].get(
            "structure_dir",
            str(Path(cfg["data"]["processed_dir"]) / "structures"),
        )
        z_table = getattr(model, "z_table", None)
        r_max = float(cfg["model"].get("r_max", 5.0))
        train_ds = StructureDataset(df_train, target, structure_dir, r_max=r_max, z_table=z_table)
        val_ds = StructureDataset(df_val, target, structure_dir, r_max=r_max, z_table=z_table)
        test_ds = StructureDataset(df_test, target, structure_dir, r_max=r_max, z_table=z_table)
        collate_fn = mace_collate
    else:
        train_ds = GraphNPZDataset(df_train, target_col=target)
        val_ds = GraphNPZDataset(df_val, target_col=target)
        test_ds = GraphNPZDataset(df_test, target_col=target)
        collate_fn = collate

    bs = int(cfg["train"]["batch_size"])
    nw = int(cfg["train"]["num_workers"])

    # Balanced sampling config - DELAYED start to stabilize early training
    use_balanced = cfg["train"].get("balanced_sampling", False if model_type == "mace" else True)
    balanced_start_epoch = int(cfg["train"].get("balanced_sampling_start_epoch", 6))

    # Create BOTH loaders - we'll switch between them
    train_sampler = create_balanced_sampler(df_train, target)
    train_loader_balanced = DataLoader(train_ds, batch_size=bs, sampler=train_sampler,
                                       num_workers=nw, collate_fn=collate_fn, pin_memory=True)
    train_loader_unbalanced = DataLoader(train_ds, batch_size=bs, shuffle=True,
                                         num_workers=nw, collate_fn=collate_fn, pin_memory=True)

    # Start with unbalanced (vanilla shuffle) for stability
    train_loader = train_loader_unbalanced if use_balanced else train_loader_unbalanced

    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate_fn, pin_memory=True)

    # Create target normalizer from training data (like original CGCNN)
    train_targets = torch.tensor(df_train[target].values, dtype=torch.float32)
    normalizer = Normalizer(train_targets)
    print(f"Target normalizer: mean={normalizer.mean:.4f}, std={normalizer.std:.4f}")

    lr = float(cfg["train"]["optimizer"]["lr"])
    wd = float(cfg["train"]["optimizer"]["weight_decay"])
    if model_type == "mace" and hasattr(model, "get_param_groups"):
        lr_bb_factor = float(cfg["train"]["optimizer"].get("lr_backbone_factor", 0.01))
        param_groups = model.get_param_groups(lr_backbone=lr * lr_bb_factor, lr_heads=lr)
        opt = torch.optim.AdamW(param_groups, weight_decay=wd)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # MultiStepLR scheduler (like original CGCNN)
    lr_milestones = cfg["train"].get("scheduler", {}).get("milestones", [60, 80])
    lr_gamma = cfg["train"].get("scheduler", {}).get("gamma", 0.1)
    scheduler = MultiStepLR(opt, milestones=lr_milestones, gamma=lr_gamma)
    base_lrs = [float(pg["lr"]) for pg in opt.param_groups]

    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"]["early_stopping"]["patience"])
    heads_cfg = cfg["model"].get("heads", {})
    min_log_sigma = float(heads_cfg.get("min_log_sigma", -4.0))
    max_log_sigma = float(heads_cfg.get("max_log_sigma", 2.0))

    run_id = cfg["project"].get("run_id") or f"run_seed{seed}"
    art_dir = Path(cfg["project"]["output_dir"]) / cfg["project"].get("artifacts_dir", "data/artifacts") / run_id
    rep_dir = Path(cfg["project"]["output_dir"]) / cfg["project"].get("reports_dir", "data/reports") / run_id
    art_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    # Early stopping: track screening_score for model selection, best_mae for reporting
    best_score = float("-inf")  # For model selection (higher is better)
    best_mae = float("inf")      # For reporting (lower is better)
    bad = 0
    best_path = art_dir / "best.pt"
    
    # Loss config (simplified for quantile regression)
    cls_weight = float(cfg.get("loss", {}).get("cls_weight", 0.1))
    
    # Warmup config
    warmup_epochs = int(cfg["train"].get("warmup_epochs", 5))
    grad_accum_steps = max(1, int(cfg["train"].get("grad_accum_steps", 1)))
    if grad_accum_steps > 1:
        print(
            f"Gradient accumulation enabled: steps={grad_accum_steps} "
            f"(effective batch size={bs * grad_accum_steps})"
        )

    for ep in range(1, epochs + 1):
        # LINEAR WARMUP: preserve per-parameter-group LR ratios
        if ep <= warmup_epochs:
            warmup_scale = ep / warmup_epochs
            for i, param_group in enumerate(opt.param_groups):
                param_group["lr"] = base_lrs[i] * warmup_scale
        
        # DELAYED BALANCED SAMPLING: switch at balanced_start_epoch
        if use_balanced and ep == balanced_start_epoch:
            train_loader = train_loader_balanced
            print(f"[Epoch {ep}] Switching to balanced sampling")
        
        model.train()
        epoch_losses = {'loss_quantile': 0, 'loss_cls_stable': 0, 'loss_cls_meta': 0}
        n_batches = 0
        opt.zero_grad(set_to_none=True)
        
        for batch_idx, raw_batch in enumerate(train_loader):
            batch_data, y = _unpack_batch(raw_batch, model_type, device)

            # Debug first batch of first epoch
            debug_this_batch = (ep == 1 and batch_idx == 0)
            if debug_this_batch:
                if model_type != "mace":
                    x_dbg = batch_data[0]
                    b_dbg = batch_data[4]
                    print(f"\n[DEBUG BATCH] x.shape={x_dbg.shape}, batch_index unique={b_dbg.unique().shape[0]} graphs")
                print(f"[DEBUG BATCH] y range: [{y.min().item():.4f}, {y.max().item():.4f}], std={y.std().item():.4f}")

            # Keep raw targets for classification, normalize for regression
            y_raw = y.clone()
            y_norm = normalizer.norm(y)

            try:
                q10, q50, q90, logit_stable, logit_meta = _model_forward(
                    model, batch_data, model_type, debug=debug_this_batch
                )
            except RuntimeError as exc:
                err = str(exc).lower()
                if (
                    model_type == "mace"
                    and device.type == "cuda"
                    and (
                        "cublas_status_execution_failed" in err
                        or "cuda out of memory" in err
                        or "cuda error" in err
                    )
                ):
                    n_graphs = int(y.shape[0])
                    n_nodes = int(batch_data["positions"].shape[0])
                    n_edges = int(batch_data["edge_index"].shape[1])
                    raise RuntimeError(
                        "MACE CUDA forward failed (likely GPU memory pressure from a large "
                        f"variable-size batch): batch_graphs={n_graphs}, "
                        f"batch_nodes={n_nodes}, batch_edges={n_edges}, train.batch_size={bs}. "
                        "Try lowering train.batch_size (for example 16) and setting "
                        "train.grad_accum_steps > 1 to preserve effective batch size."
                    ) from exc
                raise
            
            loss, loss_dict = compute_multi_task_loss(
                q10, q50, q90, logit_stable, logit_meta, y_norm, y_raw,
                cls_weight=cls_weight
            )
            
            loss = loss / grad_accum_steps
            loss.backward()
            should_step = ((batch_idx + 1) % grad_accum_steps == 0) or (
                (batch_idx + 1) == len(train_loader)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(cfg["train"]["grad_clip_norm"])
                )
                opt.step()
                opt.zero_grad(set_to_none=True)
            
            for k, v in loss_dict.items():
                epoch_losses[k] += v
            n_batches += 1

        # Step scheduler only after warmup completes
        if ep > warmup_epochs:
            scheduler.step()
        val_metrics = evaluate(model, val_loader, device, normalizer=normalizer, model_type=model_type)
        val_mae = val_metrics["mae"]
        if len(opt.param_groups) > 1:
            lr_msg = (
                f"lr_head={opt.param_groups[-1]['lr']:.6f} "
                f"lr_backbone={opt.param_groups[0]['lr']:.6f}"
            )
        else:
            lr_msg = f"lr={opt.param_groups[0]['lr']:.6f}"
        
        # Average epoch losses
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)
        
        # Comprehensive logging (quantile-based)
        print(f"epoch={ep} {lr_msg} mae={val_mae:.4f} rmse={val_metrics['rmse']:.4f} "
              f"coverage_80={val_metrics['coverage_80']:.2f}")
        print(f"  quantiles: q10_p50={val_metrics['q10_p50']:.3f} q50_p50={val_metrics['q50_p50']:.3f} "
              f"q90_p50={val_metrics['q90_p50']:.3f} interval_width={val_metrics['interval_width_p50']:.3f}")
        print(f"  prec@K: 25={val_metrics['prec_at_25']:.2f} 100={val_metrics['prec_at_100']:.2f} 250={val_metrics['prec_at_250']:.2f}")
        print(f"  recall@K: 25={val_metrics['recall_at_25']:.2f} 100={val_metrics['recall_at_100']:.2f} 250={val_metrics['recall_at_250']:.2f}")
        print(f"  KEEP={val_metrics['n_keep']}(prec={val_metrics['keep_precision']:.2f}) "
              f"MAYBE={val_metrics['n_maybe']} "
              f"KILL={val_metrics['n_kill']}(prec={val_metrics['kill_precision']:.2f}) "
              f"[true: {val_metrics['n_true_stable']} stable, {val_metrics['n_true_meta']} meta]")

        # EARLY STOPPING: Use prec_at_25 as primary metric (product KPI!)
        # Hard constraint: val_mae < 0.06 (regression must be reasonable)
        # Score: prec_at_25 - 0.5 * false_stable_rate
        prec_at_25 = val_metrics['prec_at_25']
        recall_at_25 = val_metrics['recall_at_25']
        false_stable = val_metrics['false_stable_rate']
        
        # Composite score using actual product metric
        if val_mae < 0.06:
            # Full screening mode: prec@25 - penalty for false stables
            screening_score = prec_at_25 - 0.5 * false_stable
        else:
            # Fallback: use negative MAE until MAE is acceptable
            screening_score = -val_mae
        
        # Always track best MAE separately for reporting
        if val_mae < best_mae:
            best_mae = val_mae
        
        if screening_score > best_score:
            best_score = screening_score
            bad = 0
            torch.save({"state_dict": model.state_dict(), "cfg": cfg, "val": val_metrics, "normalizer": normalizer.state_dict()}, best_path)
            if val_mae < 0.06:
                print(f"  ✓ New best! score={screening_score:.3f} (prec@25={prec_at_25:.2f}, recall@25={recall_at_25:.2f}, false_stable={false_stable:.2f})")
            else:
                print(f"  ✓ New best (MAE phase)! mae={val_mae:.4f}")
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stopping at epoch {ep}")
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    if "normalizer" in ckpt:
        normalizer.load_state_dict(ckpt["normalizer"])

    # Temperature scaling for calibration (post-hoc)
    print("Calibrating with temperature scaling...")
    temp_scaler_stable = TemperatureScaling().to(device)
    temp_scaler_meta = TemperatureScaling().to(device)
    
    # Collect validation logits for calibration
    model.eval()
    val_logits_stable, val_logits_meta, val_labels = [], [], []
    with torch.no_grad():
        for raw_batch in val_loader:
            batch_data, y = _unpack_batch(raw_batch, model_type, device)
            _, _, _, logit_stable, logit_meta = _model_forward(model, batch_data, model_type)
            val_logits_stable.append(logit_stable)
            val_logits_meta.append(logit_meta)
            val_labels.append(y)
    
    val_logits_stable = torch.cat(val_logits_stable)
    val_logits_meta = torch.cat(val_logits_meta)
    val_labels = torch.cat(val_labels)
    
    # Calibrate temperatures
    label_stable = (val_labels < THRESH_STABLE).float()
    label_meta = (val_labels < THRESH_METASTABLE).float()
    
    temp_stable = temp_scaler_stable.calibrate(val_logits_stable, label_stable)
    temp_meta = temp_scaler_meta.calibrate(val_logits_meta, label_meta)
    print(f"Calibrated temperatures: stable={temp_stable:.3f}, meta={temp_meta:.3f}")

    val_metrics = evaluate(model, val_loader, device, normalizer=normalizer, model_type=model_type)
    test_metrics = evaluate(model, test_loader, device, normalizer=normalizer, model_type=model_type)

    report = {
        "run_id": run_id,
        "dataset": ds_name,
        "target": target,
        "counts": {"train": int(len(df_train)), "val": int(len(df_val)), "test": int(len(df_test))},
        "val": val_metrics,
        "test": test_metrics,
        "best_val_mae": float(best_mae),  # Fixed: now actually reports MAE, not score
        "best_screening_score": float(best_score),  # The actual selection metric
        "checkpoint": str(best_path),
        "calibration": {"temp_stable": temp_stable, "temp_meta": temp_meta},
    }
    (rep_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
