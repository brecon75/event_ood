"""
monitor.py — VmemMonitor for SpikingJelly MultiStepParametricLIFNode.

Hooks into PLIF layers to collect membrane potential 'v' and compute
phi vectors [mean, variance, excess_kurtosis] with Global Average Pooling.

Shape contract
--------------
spike_model.py reshapes input (B, 20, H, W) → (B, 2, 10, H, W) then passes
it to features_01 / features_23 (SeqToANNContainer + MultiStepParametricLIFNode).
SpikingJelly's MultiStepParametricLIFNode in clock_driven mode stores its
membrane potential as module.v with shape (T, C, H, W) — the batch dim is
folded into the time sequence by the backbone's own reshape.

To keep all downstream code simple and correct, _make_hook canonicalises
every captured tensor to (T, 1, C, H, W) immediately after capture.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode


class VmemMonitor:
    def __init__(self, model: nn.Module, selected: Optional[List[int]] = None):
        """
        model    : the backbone or full model to monitor
        selected : list of PLIF indices to hook (0-indexed). None hooks all.
        """
        self._v: Dict[int, List[torch.Tensor]] = {}
        self._hooks = []
        self._selected = selected

        idx = 0
        for name, module in model.named_modules():
            if isinstance(module, MultiStepParametricLIFNode):
                if selected is None or idx in selected:
                    self._v[idx] = []
                    self._hooks.append(
                        module.register_forward_hook(self._make_hook(idx))
                    )
                idx += 1

        if not self._hooks:
            print("[VmemMonitor] WARNING: No MultiStepParametricLIFNode layers found!")
        else:
            print(f"[VmemMonitor] Hooked {len(self._hooks)} PLIF layer(s).")

    # ------------------------------------------------------------------
    # Internal hook
    # ------------------------------------------------------------------
    def _make_hook(self, idx: int):
        def hook(module, input, output):
            if not (hasattr(module, 'v') and module.v is not None):
                return
            v = module.v.detach().float()  # KEEP ON GPU for fast moments calculation!

            # Canonicalise to (T, 1, C, H, W) regardless of what SpikingJelly
            # hands us.  The backbone squeezes the batch dim into T, so v is
            # always 4-D here.  We unsqueeze a B=1 dim so every downstream
            # function has a consistent 5-D tensor.
            if v.ndim == 4:          # (T, C, H, W)  — expected path
                v = v.unsqueeze(1)   # → (T, 1, C, H, W)
            elif v.ndim == 5:        # already (T, B, C, H, W)
                pass
            else:
                # Unexpected — skip to avoid silent corruption
                print(f"[VmemMonitor] layer {idx}: unexpected v.ndim={v.ndim}, skipping.")
                return

            self._v[idx].append(v)  # each entry: (T, B, C, H, W)
        return hook

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def reset(self):
        """Clear all collected membrane potentials."""
        for k in self._v:
            self._v[k] = []

    # ------------------------------------------------------------------
    # Phi extraction (B, 3*sum_C)
    # ------------------------------------------------------------------
    def collect_phi(self) -> torch.Tensor:
        """
        Compute phi = [GAP(mean), GAP(var), GAP(excess_kurtosis)] per layer
        and concatenate across layers.

        Returns
        -------
        (B, 3 * sum_layers(C_l))  — one row per sample in the batch.
        With BATCH_SIZE=1 this is (1, 2112).
        """
        parts = []
        for idx in sorted(self._v.keys()):
            v_list = self._v[idx]
            if not v_list:
                continue

            # Stack along batch axis so we get (T, total_B, C, H, W).
            # Each entry in v_list is already (T, B, C, H, W).
            V = torch.cat(v_list, dim=1)          # (T, B, C, H, W)
            T, B, C, H, W = V.shape

            # Flatten spatial → (T, B, C, D)
            D = H * W
            V = V.view(T, B, C, D)

            # --- Temporal moments per neuron: (B, C, D) ---
            mu   = V.mean(0)                       # (B, C, D)
            var  = V.var(0, unbiased=False).clamp(min=1e-8)  # (B, C, D)

            diff = V - mu.unsqueeze(0)             # (T, B, C, D)
            kurt = (diff ** 4).mean(0) / (var ** 2) - 3.0   # (B, C, D)

            # --- Global Average Pooling over D → (B, C) ---
            mu_gap   = mu.mean(-1)
            var_gap  = var.mean(-1)
            kurt_gap = kurt.mean(-1)

            # Concatenate stats for this layer: (B, 3*C)
            p = torch.cat([mu_gap, var_gap, kurt_gap], dim=-1)
            parts.append(p)

        if not parts:
            return torch.empty((0,))

        return torch.cat(parts, dim=-1)  # (B, 3 * sum(C_layers))

    # ------------------------------------------------------------------
    # Spatial GAP trajectory extraction (B, T, sum(C_layers))
    # ------------------------------------------------------------------
    def collect_temporal_gap(self) -> torch.Tensor:
        """
        Compute spatial Global Average Pooling (GAP) online on GPU
        over hooked layers to bypass the 15 TB trajectory storage bottleneck.

        Returns
        -------
        (B, T, sum(C_layers)) on CPU
        """
        parts = []
        for idx in sorted(self._v.keys()):
            v_list = self._v[idx]
            if not v_list:
                continue
            V = torch.cat(v_list, dim=1)  # (T, B, C, H, W)
            T, B, C, H, W = V.shape

            # Spatial average: (T, B, C, H, W) -> (T, B, C)
            V_gap = V.mean(dim=(3, 4))
            parts.append(V_gap.cpu())

        if not parts:
            return torch.empty((0,))

        # Concatenate channels along dim -1: (T, B, sum(C_layers))
        cat_gap = torch.cat(parts, dim=-1)
        # Permute (T, B, sum(C_layers)) -> (B, T, sum(C_layers))
        return cat_gap.permute(1, 0, 2)

    # ------------------------------------------------------------------
    # Temporal phi extraction (B, sum_layers * 7)
    # ------------------------------------------------------------------
    def collect_temporal_phi(self, theta: float = 1.0) -> torch.Tensor:
        parts = []
        for idx in sorted(self._v.keys()):
            v_list = self._v[idx]
            if not v_list:
                continue
            V = torch.cat(v_list, dim=1)  # (T, B, C, H, W)
            T, B, C, H, W = V.shape
            if T < 2:
                continue
            
            # Average over channels and space to get scalar trace per batch sample: (T, B)
            V_scalar = V.mean(dim=(2, 3, 4))
            
            margin = theta - V_scalar  # (T, B)
            m_mean = margin.mean(dim=0)  # (B,)
            m_min  = margin.min(dim=0).values  # (B,)
            m_var  = margin.var(dim=0, unbiased=False)  # (B,)
            
            dV      = V_scalar[1:] - V_scalar[:-1]  # (T-1, B)
            dV_mean = dV.abs().mean(dim=0)  # (B,)
            dV_var  = dV.var(dim=0, unbiased=False)  # (B,)
            
            std  = V_scalar.std(dim=0, unbiased=False).clamp(min=1e-8)  # (B,)
            Vc   = V_scalar - V_scalar.mean(dim=0, keepdim=True)  # (T, B)
            autocorr = (Vc[:-1] * Vc[1:]).mean(dim=0) / std ** 2  # (B,)
            
            fft_mag  = torch.fft.rfft(V_scalar, dim=0).abs() ** 2  # (freq, B)
            total_e  = fft_mag.sum(dim=0).clamp(min=1e-8)  # (B,)
            hf_e     = fft_mag[max(1, T // 4):].sum(dim=0)  # (B,)
            hf_ratio = hf_e / total_e  # (B,)
            
            # Stack features: shape (B, 7)
            layer_feat = torch.stack(
                [m_mean, m_min, m_var, dV_mean, dV_var, autocorr, hf_ratio], dim=1
            )
            parts.append(layer_feat)
            
        if not parts:
            device = "cpu"
            for k in self._v:
                if self._v[k]:
                    device = self._v[k][0].device
                    break
            return torch.empty((0, 0), device=device)
            
        return torch.cat(parts, dim=1)  # (B, sum_layers * 7)

    # ------------------------------------------------------------------
    # Trajectory extraction  {layer_idx: (T, n_samples, D)}
    # ------------------------------------------------------------------
    def trajectories(self, n_samples: int = 50) -> Dict[int, torch.Tensor]:
        """
        Return raw V(t) trajectories for the first n_samples in the batch.

        Returns
        -------
        dict  layer_idx → (T, min(B, n_samples), D)
        """
        out = {}
        for idx, v_list in self._v.items():
            if not v_list:
                continue
            V = torch.cat(v_list, dim=1).float()  # (T, B, C, H, W)
            T, B, C, H, W = V.shape
            take = min(B, n_samples)
            D = C * H * W
            # Slice samples (dim 1), then flatten C*H*W → D
            out[idx] = V[:, :take].reshape(T, take, D)
        return out

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def remove(self):
        """Remove all hooks from the model."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
        print("[VmemMonitor] All hooks removed.")
