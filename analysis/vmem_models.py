import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Fix paths for imports
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
from analysis.vmem_utils import LAYER_SPECS

class CouplingLayer(nn.Module):
    def __init__(self, dim, hidden_dim=64, flip=False):
        super().__init__()
        self.dim = dim
        self.split_dim = dim // 2
        self.flip = flip
        # The conditioner (x1) and the transformed half (x2) swap sizes when the
        # split is flipped. For odd `dim` these differ, so the s/t nets must be
        # sized for THIS layer's orientation; sizing them for one orientation
        # only (the previous code) crashed on flip with odd dim. For even dim
        # both sizes equal dim//2, so this is identical to the old shapes
        # (existing even-dim checkpoints still load).
        if flip:
            cond_dim, trans_dim = dim - self.split_dim, self.split_dim
        else:
            cond_dim, trans_dim = self.split_dim, dim - self.split_dim
        self.s_net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, trans_dim), nn.Tanh()
        )
        self.t_net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, trans_dim)
        )

    def forward(self, x, flip=None):
        flip = self.flip if flip is None else flip
        if flip:
            x1, x2 = x[:, self.split_dim:], x[:, :self.split_dim]
        else:
            x1, x2 = x[:, :self.split_dim], x[:, self.split_dim:]
        s = self.s_net(x1)
        t = self.t_net(x1)
        y2 = x2 * torch.exp(s) + t
        y = torch.cat([y2, x1] if flip else [x1, y2], dim=-1)
        return y, s.sum(dim=-1)


class RealNVP(nn.Module):
    def __init__(self, dim, hidden_dim=64, n_layers=4):
        super().__init__()
        # Each layer's split orientation is fixed at construction (alternating),
        # so its s/t nets are sized correctly even for odd dim.
        self.layers = nn.ModuleList(
            [CouplingLayer(dim, hidden_dim, flip=(i % 2 == 1)) for i in range(n_layers)])

    def forward(self, x):
        log_det_tot = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        z = x
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_tot += log_det
        return z, log_det_tot

    def log_prob(self, x):
        z, log_det = self.forward(x)
        prior = -0.5 * torch.sum(z ** 2 + 1.837877, dim=-1)  # log(2*pi) = 1.837877
        return prior + log_det


class Autoencoder(nn.Module):
    def __init__(self, dim, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))


class TemporalAutoencoder(nn.Module):
    def __init__(self, dim, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(10, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * dim, latent_dim),
            nn.ReLU()
        )
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 4 * dim),
            nn.ReLU()
        )
        self.decoder_conv = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 10, kernel_size=3, padding=1)
        )
        self.dim = dim
    def forward(self, x):
        B = x.shape[0]
        z = self.encoder(x)
        dec_fc = self.decoder_fc(z).view(B, 4, self.dim)
        recon = self.decoder_conv(dec_fc)
        return recon


def train_flow_model(clean_pca, epochs=100, lr=1e-3, batch_size=128, device="cuda"):
    fast_mode = "--fast" in sys.argv
    if fast_mode:
        epochs = min(epochs, 1)
    flow = RealNVP(dim=clean_pca.shape[1]).to(device)
    optimizer = optim.Adam(flow.parameters(), lr=lr)
    dataset = torch.from_numpy(clean_pca).float()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    flow.train()
    pbar = tqdm(range(epochs), desc="Training Flow Model", leave=False, disable=epochs <= 1)
    for _ in pbar:
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = -flow.log_prob(batch).mean()
            loss.backward()
            optimizer.step()
    flow.eval()
    return flow


def train_ae_model(clean_raw, epochs=40, lr=1e-3, batch_size=128, device="cuda"):
    fast_mode = "--fast" in sys.argv
    if fast_mode:
        epochs = min(epochs, 1)
    ae = Autoencoder(dim=clean_raw.shape[1]).to(device)
    optimizer = optim.Adam(ae.parameters(), lr=lr)
    dataset = torch.from_numpy(clean_raw).float()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    ae.train()
    pbar = tqdm(range(epochs), desc="Training Autoencoder", leave=False, disable=epochs <= 1)
    for _ in pbar:
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = ae(batch)
            loss = nn.MSELoss()(recon, batch)
            loss.backward()
            optimizer.step()
    ae.eval()
    return ae


def prepare_temporal_ae_input(trajs):
    clean_gaps = []
    for idx in sorted(trajs.keys()):
        V = trajs[idx].float()
        C = LAYER_SPECS[idx]["C"]
        V_gap = V.view(10, -1, C, V.shape[-1] // C).mean(-1)
        clean_gaps.append(V_gap)
    clean_x = torch.cat(clean_gaps, dim=-1).permute(1, 0, 2)
    return clean_x


def train_temporal_ae_model(clean_trajs, epochs=200, lr=1e-3, batch_size=64, device="cuda"):
    fast_mode = "--fast" in sys.argv
    if fast_mode:
        epochs = min(epochs, 2)
    if isinstance(clean_trajs, torch.Tensor):
        clean_x = clean_trajs
    else:
        clean_x = prepare_temporal_ae_input(clean_trajs)
    ae = TemporalAutoencoder(dim=clean_x.shape[2]).to(device)
    optimizer = optim.Adam(ae.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(clean_x),
        batch_size=batch_size,
        shuffle=True,
    )
    ae.train()
    pbar = tqdm(range(epochs), desc="Training Temporal AE", leave=False, disable=epochs <= 1)
    for _ in pbar:
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = ae(batch)
            loss = nn.MSELoss()(recon, batch)
            loss.backward()
            optimizer.step()
    ae.eval()
    return ae
