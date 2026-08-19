"""
monitor.py — VmemMonitor for SpikingJelly MultiStepParametricLIFNode.

Hooks into PLIF layers to collect membrane potential 'v' and compute
phi vectors [mean, variance, excess_kurtosis] with Global Average Pooling.

Shape contract (verified against spikingjelly.clock_driven.neuron source)
-------------------------------------------------------------------------
spike_model.py reshapes the input (B, 20, H, W) → (B, 10, 2, H, W) and feeds
it to features_01 / features_23 (SeqToANNContainer + MultiStepParametricLIFNode).
SpikingJelly's multi-step nodes unroll dim 0 as time and store:
  module.v_seq — membrane at ALL unrolled steps, shape (dim0, *rest)
  module.v     — membrane at the LAST unrolled step only, shape (*rest)

With the mandatory BATCH_SIZE=1, dim 0 (the batch axis) has length 1, so the
node performs exactly ONE integration step whose "neuron batch" is the 10
time bins. module.v therefore has shape (10, C, H, W): the leading axis is
the 10 time bins, each processed by an independent single LIF step (reset
every frame). This is what downstream code treats as the "T" axis of phi.

NOTE: this only holds for B=1 — with B>1 module.v would be the membrane of
the LAST sample only, silently corrupting phi. extract.py enforces B=1.

To keep all downstream code simple and consistent, _make_hook canonicalises
every captured tensor to (T, 1, C, H, W) immediately after capture.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode


# ---------------------------------------------------------------------------
# Spatial-organization statistics (the compact block appended to phi_spatial)
# ---------------------------------------------------------------------------
# Rationale: spatial_var/spatial_pr are global averages over the whole frame,
# so a spatially-confined deviation gets diluted by the majority of pixels
# that didn't change — the same blind spot GAP has for the temporal mean.
# These statistics instead encode two priors that natural sensor data obeys
# everywhere and always:
#   * a local-texture floor — real membrane maps carry fine spatial structure
#     in every neighbourhood, in at least some channel; a neighbourhood that
#     is smooth in ALL channels simultaneously violates it ("too flat");
#   * motion non-stationarity — with an ego-moving event sensor, whatever
#     smooth areas do occur drift across the pixel grid; structure locked to
#     fixed pixel coordinates over a whole sequence violates it ("too
#     persistent").
# A violation of either prior is anomalous per se; nothing here is shaped to
# any particular corruption and nothing corrupted is ever seen during fit.
SPATIAL_SCALES = (0.25, 0.125)        # sliding-window sizes as fraction of map
COH_QUANTS = (0.001, 0.005, 0.02)     # low quantiles of the coherence map
COH_TAUS = (0.02, 0.05, 0.10)         # thresholds for flat-pixel fractions
FLAT_TAU = 0.05                       # indicator threshold feeding persistence
PERS_QUANTS = (0.995, 0.999)          # high quantiles of the persistence map
PERS_THRESH = (0.9, 0.99)             # thresholds for persistent-pixel fractions
SPATIAL_COMPACT_DIMS = (2 * len(SPATIAL_SCALES)              # cohflat (mu,var)
                        + 2 * len(COH_QUANTS)                # cohq (mu,var)
                        + 2 * len(COH_TAUS)                  # cohfrac (mu,var)
                        + len(PERS_QUANTS) + len(PERS_THRESH))  # 20 per layer


def _win_relstd(m: torch.Tensor, frac: float) -> torch.Tensor:
    """Within-window spatial std of a (C, H, W) map, relative to each
    channel's global spatial std. Window size scales with the map (frac of
    H, W) so it's comparable across PLIF layers at different resolutions.
    Returns (C, n_windows)."""
    C, H, W = m.shape
    kh = max(2, round(H * frac))
    kw = max(2, round(W * frac))
    sh = max(1, kh // 4)
    sw = max(1, kw // 4)
    E = F.avg_pool2d(m.unsqueeze(0), (kh, kw), (sh, sw))[0]
    E2 = F.avg_pool2d((m * m).unsqueeze(0), (kh, kw), (sh, sw))[0]
    std_w = (E2 - E * E).clamp(min=0).sqrt()
    g = m.flatten(1).std(dim=1, unbiased=False).clamp(min=1e-8)
    return (std_w / g[:, None, None]).flatten(1)


def _coh_map(m: torch.Tensor) -> torch.Tensor:
    """Per-pixel cross-channel texture: local 3x3 spatial std relative to the
    channel's global std, MAX over channels. A pixel that is smooth in every
    channel at once scores low — natural smooth spots usually keep texture in
    at least one channel. (C, H, W) -> (H-2, W-2)."""
    E = F.avg_pool2d(m.unsqueeze(0), 3, 1)[0]
    E2 = F.avg_pool2d((m * m).unsqueeze(0), 3, 1)[0]
    lstd = (E2 - E * E).clamp(min=0).sqrt()
    g = m.flatten(1).std(dim=1, unbiased=False).clamp(min=1e-8)
    return (lstd / g[:, None, None]).amax(dim=0)


class VmemMonitor:
    def __init__(self, model: nn.Module, selected: Optional[List[int]] = None,
                 valid_frac: Optional[tuple] = None):
        """
        model      : the backbone or full model to monitor
        selected   : list of PLIF indices to hook (0-indexed). None hooks all.
        valid_frac : (h_frac, w_frac) of each layer map that is real sensor
                     data. The model's input padder pads right+bottom only, so
                     the valid region is the TOP-LEFT crop; the pad band is
                     constant by construction and would contaminate every
                     spatial-organization statistic. None = full map. Set it
                     (or call set_valid_frac) before the first frame; the
                     fraction is resolution-free, so one value serves all
                     layers.
        """
        self._v: Dict[int, List[torch.Tensor]] = {}
        self._spikes: Dict[int, List[torch.Tensor]] = {}
        self._hooks = []
        self._selected = selected
        self.valid_frac = valid_frac
        # per-sequence causal persistence state (see new_sequence())
        self._pers_acc: Dict[int, torch.Tensor] = {}
        self._pers_n = 0

        idx = 0
        for name, module in model.named_modules():
            if isinstance(module, MultiStepParametricLIFNode):
                if selected is None or idx in selected:
                    self._v[idx] = []
                    self._spikes[idx] = []
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

            if isinstance(output, torch.Tensor):
                spikes = output.detach().float()
                if spikes.ndim == 4:
                    spikes = spikes.unsqueeze(1)
                elif spikes.ndim == 5:
                    # spike_seq has shape (1, 10, C, H, W): dim 0 is the single
                    # multistep step (B=1), dim 1 the 10 time bins. Canonicalise
                    # to (T=10, B=1, C, H, W) so spike rows align 1:1 with the
                    # phi rows (one per frame) instead of 10 per frame.
                    if spikes.shape[0] == 1:
                        spikes = spikes.flatten(0, 1).unsqueeze(1)
                else:
                    return
                self._spikes[idx].append(spikes)
        return hook

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------
    def reset(self):
        """Clear all collected membrane potentials and spikes."""
        for k in self._v:
            self._v[k] = []
        for k in self._spikes:
            self._spikes[k] = []

    def set_valid_frac(self, valid_frac: tuple):
        """Set (h_frac, w_frac) of the map that is real sensor data (padding
        excluded). Derived by the caller from the padder's input/output shapes
        so nothing is hardcoded to one sensor resolution."""
        self.valid_frac = valid_frac

    def new_sequence(self):
        """Reset the per-sequence persistence accumulators. MUST be called at
        the start of every sequence — the persistence statistics are causal
        running means over the sequence so far, and letting them leak across
        sequences breaks the motion-non-stationarity prior they encode."""
        self._pers_acc = {}
        self._pers_n = 0

    # ------------------------------------------------------------------
    # Spike features extraction
    # ------------------------------------------------------------------
    def collect_spikes(self) -> Dict[str, torch.Tensor]:
        """
        Compute spike_rate and spike_entropy per layer and channel.
        Returns a dict with:
          'spike_rate': (B, sum_C)
          'spike_entropy': (B, sum_C)
        """
        rate_parts = []
        entropy_parts = []
        
        for idx in sorted(self._spikes.keys()):
            sp_list = self._spikes[idx]
            if not sp_list:
                continue
            
            # Concatenate list of batch tensors along dim 1 (batch)
            S = torch.cat(sp_list, dim=1)  # (T, B, C, H, W)
            p = S.mean(dim=(0, 3, 4))      # (B, C)
            
            # Compute binary entropy: H(p) = -p log2(p) - (1-p) log2(1-p)
            p_clip = torch.clamp(p, min=1e-8, max=1-1e-8)
            entropy = -p_clip * torch.log2(p_clip) - (1 - p_clip) * torch.log2(1 - p_clip)
            
            rate_parts.append(p.cpu())
            entropy_parts.append(entropy.cpu())
            
        if not rate_parts:
            # Handle empty case with correct device (cpu)
            device = "cpu"
            for k in self._spikes:
                if self._spikes[k]:
                    device = self._spikes[k][0].device
                    break
            return {
                'spike_rate': torch.empty((0, 0), device=device),
                'spike_entropy': torch.empty((0, 0), device=device)
            }
            
        return {
            'spike_rate': torch.cat(rate_parts, dim=-1),      # (B, sum_C)
            'spike_entropy': torch.cat(entropy_parts, dim=-1) # (B, sum_C)
        }

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
    # Spatial-organization compact block (1, 20) per layer
    # ------------------------------------------------------------------
    def _compact_spatial_stats(self, mu_map: torch.Tensor, var_map: torch.Tensor,
                               idx: int) -> torch.Tensor:
        """The 20 spatial-organization statistics for one layer's (B, C, H, W)
        temporal-mean and temporal-variance maps (B=1 per the repo-wide
        contract; the persistence accumulator is per-sequence state and only
        meaningful at B=1).

        Per-layer layout (20 dims):
          [cohflat_mu_w25, cohflat_mu_w12, cohflat_var_w25, cohflat_var_w12,
           cohq_mu(3), cohq_var(3), cohfrac_mu(3), cohfrac_var(3),
           persq(2), persfrac(2)]

        cohflat : channel-mean windowed relative std, min over windows — the
                  local-texture floor at two window scales, for both maps.
        cohq    : low quantiles of the cross-channel coherence map.
        cohfrac : fraction of pixels below each coherence threshold.
        persq   : high quantiles of the causal per-pixel persistence map R
                  (running mean of the flat indicator over the sequence so
                  far) — how persistent the MOST persistent pixels are.
        persfrac: fraction of pixels with R above each persistence threshold.
        """
        dev = mu_map.device
        vh_f, vw_f = self.valid_frac if self.valid_frac is not None else (1.0, 1.0)
        m_mu = mu_map[0]
        m_var = var_map[0]
        C, H, W = m_mu.shape
        vh = max(4, round(H * vh_f))
        vw = max(4, round(W * vw_f))
        m_mu = m_mu[:, :vh, :vw]
        m_var = m_var[:, :vh, :vw]

        vals = []
        for m in (m_mu, m_var):
            for frac in SPATIAL_SCALES:
                vals.append(_win_relstd(m, frac).mean(0).amin().reshape(1))
        coh_mu = _coh_map(m_mu)
        coh_var = _coh_map(m_var)
        for cmap in (coh_mu, coh_var):
            vals.append(torch.quantile(cmap.flatten(),
                                       torch.tensor(COH_QUANTS, device=dev)))
        for cmap in (coh_mu, coh_var):
            fc = cmap.flatten()
            vals.append(torch.stack([(fc < t).float().mean() for t in COH_TAUS]))
        # causal persistence: flat means flat in BOTH maps at once
        ind = ((coh_mu < FLAT_TAU) & (coh_var < FLAT_TAU)).float()
        if idx not in self._pers_acc:
            self._pers_acc[idx] = torch.zeros_like(ind)
        self._pers_acc[idx] += ind
        R = (self._pers_acc[idx] / max(1, self._pers_n + 1)).flatten()
        vals.append(torch.quantile(R, torch.tensor(PERS_QUANTS, device=dev)))
        vals.append(torch.stack([(R > t).float().mean() for t in PERS_THRESH]))
        return torch.cat(vals).unsqueeze(0)  # (1, 20)

    # ------------------------------------------------------------------
    # Spatial-dispersion phi (B, 2*sum_C + 20*L) — survives what GAP discards
    # ------------------------------------------------------------------
    def collect_phi_spatial(self) -> torch.Tensor:
        """Spatial statistics of the membrane activity map that GAP destroys.

        collect_phi() Global-Average-Pools over space, discarding *where* each
        channel fired and keeping only the average. Spatially-structured
        deviations leave their primary signature in that layout, so they are
        nearly invisible to GAP'd phi. From the per-pixel temporal-mean map
        mu_pix (B,C,D) and temporal-variance (energy) map var_pix (B,C,D) we
        keep, per channel (near-zero extra cost, same forward pass):

          spatial_var : Var over space of mu_pix — how NON-UNIFORM mean
                        activity is across the sensor. A two-sided signal
                        that GAP averages to zero.
          spatial_pr  : participation ratio of the energy map var_pix over
                        space, (Σ)^2 / (D · Σ²) in (0,1]; 1 = energy spread
                        evenly across pixels, →0 = concentrated in a few.
                        Captures spatial sparsity / concentration.

        plus the 20-dim-per-layer spatial-ORGANIZATION block (see
        _compact_spatial_stats): both stats above are still global averages,
        so a small region that deviates from the rest gets diluted by the
        majority of unchanged pixels (the same blind spot GAP has). The
        organization block instead scores each frame against two natural-data
        priors — the local-texture floor and motion non-stationarity — that
        are violated exactly by localized / pixel-locked structure, whatever
        its cause. Requires new_sequence() at each sequence start; call
        EITHER this method OR collect_phi_and_spatial() once per frame, not
        both (each advances the persistence state).

        Returns
        -------
        (B, 2 * sum_layers(C_l) + 20 * n_layers) — per layer
        [spatial_var(C) | spatial_pr(C)] concatenated across layers, then the
        per-layer 20-dim compact blocks. With BATCH_SIZE=1 and 4 hooked
        layers this is (1, 1488).
        """
        parts = []
        compact_parts = []
        for idx in sorted(self._v.keys()):
            v_list = self._v[idx]
            if not v_list:
                continue
            V = torch.cat(v_list, dim=1)          # (T, B, C, H, W)
            T, B, C, H, W = V.shape
            D = H * W
            V = V.view(T, B, C, D)

            mu  = V.mean(0)                         # (B, C, D)
            var = V.var(0, unbiased=False).clamp(min=1e-8)  # (B, C, D)

            spatial_var = mu.var(-1, unbiased=False)        # (B, C)
            s1 = var.sum(-1)                                # (B, C)
            # Floor only guards a true 0/0; it must stay well below D*(var floor)^2
            # (= D*1e-16) or it breaks the ratio's cancellation for low-energy
            # channels, collapsing a uniform map's PR from 1.0 toward 0.
            s2 = (var ** 2).sum(-1).clamp(min=1e-20)        # (B, C)
            spatial_pr = (s1 ** 2) / (D * s2)               # (B, C) in (0, 1]

            parts.append(torch.cat([spatial_var, spatial_pr], dim=-1))  # (B, 2C)
            compact_parts.append(self._compact_spatial_stats(
                mu.view(B, C, H, W), var.view(B, C, H, W), idx))        # (1, 20)

        if not parts:
            return torch.empty((0,))

        self._pers_n += 1
        return torch.cat(parts + compact_parts, dim=-1)

    # ------------------------------------------------------------------
    # Combined phi + spatial-dispersion phi (single shared pass)
    # ------------------------------------------------------------------
    def collect_phi_and_spatial(self):
        """Compute collect_phi() and collect_phi_spatial() in one pass.

        Both methods materialise the same per-layer ``V = cat(v_list)``, its
        view, and the temporal moments ``mu``/``var``. Calling them separately
        does that work twice per frame; this fuses them so mu/var are computed
        once and reused for kurtosis (phi) and the spatial stats. Same math as
        the two standalone methods — but call only ONE of the two spatial
        collectors per frame: each advances the per-sequence persistence
        state (see new_sequence()).

        Returns
        -------
        (phi, phi_spatial) : tuple of tensors with the same shapes/semantics as
        collect_phi() and collect_phi_spatial(); empty tensors when no membrane
        was captured.
        """
        phi_parts = []
        sp_parts = []
        compact_parts = []
        for idx in sorted(self._v.keys()):
            v_list = self._v[idx]
            if not v_list:
                continue
            V = torch.cat(v_list, dim=1)                     # (T, B, C, H, W)
            T, B, C, H, W = V.shape
            D = H * W
            V = V.view(T, B, C, D)

            mu  = V.mean(0)                                  # (B, C, D)
            var = V.var(0, unbiased=False).clamp(min=1e-8)   # (B, C, D)

            # --- phi: [GAP(mu), GAP(var), GAP(excess kurtosis)] ---
            diff = V - mu.unsqueeze(0)
            kurt = (diff ** 4).mean(0) / (var ** 2) - 3.0
            phi_parts.append(
                torch.cat([mu.mean(-1), var.mean(-1), kurt.mean(-1)], dim=-1))

            # --- spatial phi: [Var_space(mu) | participation ratio(var)] + compact block ---
            spatial_var = mu.var(-1, unbiased=False)         # (B, C)
            s1 = var.sum(-1)                                 # (B, C)
            s2 = (var ** 2).sum(-1).clamp(min=1e-20)         # (B, C)
            spatial_pr = (s1 ** 2) / (D * s2)                # (B, C)
            sp_parts.append(torch.cat([spatial_var, spatial_pr], dim=-1))
            compact_parts.append(self._compact_spatial_stats(
                mu.view(B, C, H, W), var.view(B, C, H, W), idx))  # (1, 20)

        if not phi_parts:
            return torch.empty((0,)), torch.empty((0,))
        self._pers_n += 1
        return (torch.cat(phi_parts, dim=-1),
                torch.cat(sp_parts + compact_parts, dim=-1))

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
