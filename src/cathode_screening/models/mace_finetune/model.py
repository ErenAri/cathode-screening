"""MACE-MP-0 backbone with quantile regression heads for fine-tuning.

Wraps a pre-trained MACE universal potential and replaces the energy readout
with ordered quantile heads (q10, q50, q90) and auxiliary stability classifiers.
Output interface matches CGCNN so the training loop and inference pipeline
are model-agnostic.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MACEFineTuner(nn.Module):
    """MACE-MP-0 backbone + quantile / classifier heads.

    Freezing strategy (configurable):
      * ``freeze_backbone=True, unfreeze_last_n=0`` – pure feature extraction
      * ``freeze_backbone=True, unfreeze_last_n=1`` – train last interaction
      * ``freeze_backbone=False`` – full fine-tuning (needs small LR)
    """

    def __init__(
        self,
        backbone: nn.Module,
        backbone_dim: int,
        head_dim: int = 128,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        unfreeze_last_n: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.backbone_dim = backbone_dim
        self.dropout_rate = dropout
        self._dim_verified = False

        self._configure_freezing(freeze_backbone, unfreeze_last_n)

        # ---------- task-specific heads (identical to CGCNN) ----------
        self.fc_hidden = nn.Linear(backbone_dim, head_dim)

        self.head_q50 = nn.Linear(head_dim, 1)
        self.head_d_lo = nn.Linear(head_dim, 1)
        self.head_d_hi = nn.Linear(head_dim, 1)
        nn.init.constant_(self.head_d_lo.bias, -1.0)
        nn.init.constant_(self.head_d_hi.bias, -1.0)

        self.head_p_stable = nn.Linear(head_dim, 1)
        self.head_p_metastable = nn.Linear(head_dim, 1)

    # ------------------------------------------------------------------
    # Freezing helpers
    # ------------------------------------------------------------------
    def _configure_freezing(self, freeze: bool, unfreeze_last_n: int) -> None:
        if not freeze:
            return
        for p in self.backbone.parameters():
            p.requires_grad = False
        if unfreeze_last_n > 0 and hasattr(self.backbone, "interactions"):
            n_blocks = len(self.backbone.interactions)
            for i in range(max(0, n_blocks - unfreeze_last_n), n_blocks):
                for p in self.backbone.interactions[i].parameters():
                    p.requires_grad = True
                if hasattr(self.backbone, "products") and i < len(
                    self.backbone.products
                ):
                    for p in self.backbone.products[i].parameters():
                        p.requires_grad = True
        n_frozen = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
        n_train = sum(1 for p in self.backbone.parameters() if p.requires_grad)
        logger.info("Backbone: %d frozen, %d trainable params", n_frozen, n_train)

    # ------------------------------------------------------------------
    # Pooling
    # ------------------------------------------------------------------
    @staticmethod
    def _mean_pool(feats: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.long()
        n_graphs = int(batch.max().item()) + 1
        out = torch.zeros(
            (n_graphs, feats.size(-1)), device=feats.device, dtype=feats.dtype
        )
        out.index_add_(0, batch, feats)
        cnt = torch.zeros(n_graphs, device=feats.device, dtype=feats.dtype)
        cnt.index_add_(
            0, batch, torch.ones(feats.size(0), device=feats.device, dtype=feats.dtype)
        )
        return out / cnt.unsqueeze(-1).clamp_min(1e-8)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        data: Dict[str, torch.Tensor],
        debug: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run MACE backbone → pool → quantile heads.

        Returns the same 5-tuple as CGCNN:
            (q10, q50, q90, logit_stable, logit_metastable)
        """
        output = self.backbone(data, training=self.training, compute_force=False)
        node_feats = output["node_feats"]  # [N, feat_dim]
        batch = data["batch"]

        # Extract scalar (l=0) features — always the leading channels
        if node_feats.shape[-1] > self.backbone_dim:
            scalar_feats = node_feats[:, : self.backbone_dim]
        else:
            scalar_feats = node_feats
            # Auto-correct on first forward if mismatch
            if not self._dim_verified and scalar_feats.shape[-1] != self.backbone_dim:
                actual = scalar_feats.shape[-1]
                logger.warning(
                    "Adjusting backbone_dim %d → %d", self.backbone_dim, actual
                )
                self.backbone_dim = actual
                self.fc_hidden = nn.Linear(actual, self.fc_hidden.out_features).to(
                    scalar_feats.device
                )
        self._dim_verified = True

        # Pool to graph-level, cast to float32 for heads
        g = self._mean_pool(scalar_feats.float(), batch)

        h = F.relu(self.fc_hidden(g))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)

        q50 = self.head_q50(h).squeeze(-1)
        d_lo = F.softplus(self.head_d_lo(h).squeeze(-1))
        d_hi = F.softplus(self.head_d_hi(h).squeeze(-1))
        q10 = q50 - d_lo
        q90 = q50 + d_hi

        logit_stable = self.head_p_stable(h).squeeze(-1)
        logit_meta = self.head_p_metastable(h).squeeze(-1)

        return q10, q50, q90, logit_stable, logit_meta

    def get_embedding(self, data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Graph-level embedding for OOD detection (matches CGCNN interface)."""
        with torch.no_grad():
            output = self.backbone(data, training=False, compute_force=False)
            nf = output["node_feats"]
            if nf.shape[-1] > self.backbone_dim:
                nf = nf[:, : self.backbone_dim]
            return self._mean_pool(nf.float(), data["batch"])

    # ------------------------------------------------------------------
    # Differential LR helpers
    # ------------------------------------------------------------------
    def get_param_groups(
        self, lr_backbone: float = 1e-5, lr_heads: float = 1e-3
    ) -> List[Dict]:
        """Parameter groups with separate learning rates."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params: list = []
        for m in [
            self.fc_hidden,
            self.head_q50,
            self.head_d_lo,
            self.head_d_hi,
            self.head_p_stable,
            self.head_p_metastable,
        ]:
            head_params.extend(m.parameters())
        groups: List[Dict] = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": lr_backbone})
        groups.append({"params": head_params, "lr": lr_heads})
        return groups

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "medium",
        head_dim: int = 128,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        unfreeze_last_n: int = 1,
        device: str = "cpu",
    ) -> "MACEFineTuner":
        """Load pre-trained MACE-MP-0 and attach quantile heads."""
        try:
            from mace.calculators import mace_mp
        except ImportError:
            raise ImportError(
                "mace-torch is required for MACE fine-tuning. "
                "Install with: pip install mace-torch"
            )

        logger.info("Loading MACE-MP-0 (%s) …", model_name)
        calc = mace_mp(model=model_name, device=device, default_dtype="float32")
        backbone = calc.models[0]

        backbone_dim = cls._detect_backbone_dim(backbone)
        logger.info("Detected backbone scalar dim: %d", backbone_dim)

        # Extract z_table for dataset construction
        z_table: Optional[List[int]] = None
        if hasattr(calc, "z_table"):
            z_table = [int(z) for z in calc.z_table.zs]

        model = cls(
            backbone=backbone,
            backbone_dim=backbone_dim,
            head_dim=head_dim,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n=unfreeze_last_n,
        )
        model.z_table = z_table  # type: ignore[attr-defined]
        return model

    @staticmethod
    def _detect_backbone_dim(backbone: nn.Module) -> int:
        """Infer scalar (l=0) feature dimension from MACE architecture."""
        # Method 1: parse hidden_irreps string
        if hasattr(backbone, "hidden_irreps"):
            try:
                scalar_dim = 0
                for part in str(backbone.hidden_irreps).split("+"):
                    part = part.strip()
                    if "x0e" in part:
                        scalar_dim += int(part.split("x")[0])
                if scalar_dim > 0:
                    return scalar_dim
            except (ValueError, IndexError):
                pass

        # Method 2: inspect readout layer input
        if hasattr(backbone, "readouts") and backbone.readouts:
            for m in backbone.readouts[-1].modules():
                if hasattr(m, "irreps_in"):
                    try:
                        scalar_dim = 0
                        for part in str(m.irreps_in).split("+"):
                            part = part.strip()
                            if "x0e" in part:
                                scalar_dim += int(part.split("x")[0])
                        if scalar_dim > 0:
                            return scalar_dim
                    except (ValueError, IndexError):
                        pass

        logger.warning("Could not auto-detect backbone dim, defaulting to 128")
        return 128
