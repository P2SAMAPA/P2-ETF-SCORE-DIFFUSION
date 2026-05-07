"""
Score-Based Diffusion Model (DDPM) for ETF returns with macro conditioning.

FIXES:
  Bug 2 — Cosine noise schedule was wrong: used a cosine-shaped ramp
           (beta values only) which left alpha_bar[T] ≈ 0.36 instead of ~0.
           The model could never fully destroy the signal, so it learned a
           near-identity function at high t. Replaced with the correct
           Nichol & Dhariwal (2021) formulation derived from cumulative
           alpha_bar, with betas clipped to [1e-4, 0.999].

  Bug 4 — Dead cond_proj layer: ResidualBlock defined self.cond_proj but
           never called it. The conditioning was already injected via
           concatenation in self.net. Removed the dead layer and replaced
           with a proper FiLM-style conditioning (scale+shift) so the macro
           signal modulates each residual block's activations directly.

  Bug 7 — No LR schedule, no gradient clipping, no early stopping.
           Added cosine-decay LR scheduler, gradient clipping (max_norm=1.0),
           and patience-based early stopping on validation loss.
           Training data now split 90/10 train/val chronologically.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Learnable projection on top of sinusoidal embedding
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.proj(emb)


class ResidualBlock(nn.Module):
    """
    FIX Bug 4: replaced dead cond_proj with FiLM conditioning.
    FiLM (Feature-wise Linear Modulation) produces per-channel scale and shift
    from the conditioning vector, which is far more expressive than concatenation
    and guarantees the macro signal actively modulates every residual block.
    """
    def __init__(self, dim, cond_dim, time_dim):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, dim)
        # FiLM: predict (scale, shift) from macro condition
        self.film = nn.Linear(cond_dim, dim * 2)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x, t_emb, cond):
        # Time modulation
        h = x + self.time_proj(t_emb)
        # FiLM conditioning: scale and shift
        film_params = self.film(cond)
        scale, shift = film_params.chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        return x + self.net(h)


class ScoreNetwork(nn.Module):
    def __init__(self, data_dim, cond_dim, hidden_dim=128, num_layers=4, time_dim=64):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        # Input projection: data only (cond injected via FiLM in each block)
        self.proj = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.SiLU(),
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, cond_dim, time_dim) for _ in range(num_layers)
        ])
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, data_dim),
        )

    def forward(self, x, t, cond):
        t_emb = self.time_embed(t)
        h = self.proj(x)
        for block in self.res_blocks:
            h = block(h, t_emb, cond)
        return self.out(h)


class DiffusionPredictor:
    def __init__(self, data_dim, cond_dim, hidden_dim=128, num_layers=4,
                 num_steps=100, noise_schedule="cosine", lr=1e-3, wd=1e-5, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_steps = num_steps
        self.data_dim = data_dim
        self.model = ScoreNetwork(data_dim, cond_dim, hidden_dim, num_layers).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=wd
        )
        self._build_schedule(noise_schedule)

    def _build_schedule(self, schedule):
        """
        FIX Bug 2: correct cosine schedule from Nichol & Dhariwal (2021).
        Derives beta_t from the ratio of consecutive alpha_bar values,
        guaranteeing alpha_bar[T] ≈ 0 (full noise at the end).
        """
        if schedule == "cosine":
            s = 0.008
            steps = self.num_steps + 1
            t_arr = torch.linspace(0, self.num_steps, steps)
            f = torch.cos(((t_arr / self.num_steps) + s) / (1 + s) * math.pi / 2) ** 2
            alpha_bar = f / f[0]
            alpha_bar = torch.clamp(alpha_bar, min=1e-6, max=1.0)
            # beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
            beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
            beta = torch.clamp(beta, min=1e-4, max=0.999)
            self.alpha_bar = alpha_bar[1:].to(self.device)
        else:  # linear
            beta = torch.linspace(1e-4, 0.02, self.num_steps)
            self.alpha_bar = torch.cumprod(1.0 - beta, dim=0).to(self.device)

        self.beta  = beta.to(self.device)
        self.alpha = (1.0 - self.beta).to(self.device)

    def _extract(self, arr, t):
        return arr[t].reshape(-1, 1)

    def fit(self, X, cond, epochs=200, batch_size=128, patience=40):
        """
        FIX Bug 7: chronological 90/10 train/val split, cosine LR decay,
        gradient clipping, and early stopping on validation loss.
        """
        X    = torch.tensor(X,    dtype=torch.float32)
        cond = torch.tensor(cond, dtype=torch.float32)

        # Chronological split — no shuffle on time-series data
        n_train = int(len(X) * 0.9)
        X_tr, X_val     = X[:n_train],    X[n_train:]
        c_tr, c_val     = cond[:n_train], cond[n_train:]

        tr_dataset = torch.utils.data.TensorDataset(X_tr, c_tr)
        tr_loader  = torch.utils.data.DataLoader(
            tr_dataset, batch_size=batch_size, shuffle=True, drop_last=False
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-5
        )

        best_val_loss = float("inf")
        patience_count = 0

        for epoch in range(epochs):
            # ── Training ──────────────────────────────────────────
            self.model.train()
            tr_loss = 0.0
            for bx, bc in tr_loader:
                bx, bc = bx.to(self.device), bc.to(self.device)
                t   = torch.randint(0, self.num_steps, (len(bx),), device=self.device)
                eps = torch.randn_like(bx)
                ab  = self._extract(self.alpha_bar, t)
                x_noisy  = torch.sqrt(ab) * bx + torch.sqrt(1 - ab) * eps
                pred_eps = self.model(x_noisy, t.float() / self.num_steps, bc)
                loss = F.mse_loss(pred_eps, eps)
                self.optimizer.zero_grad()
                loss.backward()
                # FIX Bug 7: gradient clipping prevents exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                tr_loss += loss.item() * len(bx)

            scheduler.step()

            # ── Validation ────────────────────────────────────────
            if len(X_val) > 0:
                self.model.eval()
                with torch.no_grad():
                    bx = X_val.to(self.device)
                    bc = c_val.to(self.device)
                    t  = torch.randint(0, self.num_steps, (len(bx),), device=self.device)
                    eps = torch.randn_like(bx)
                    ab  = self._extract(self.alpha_bar, t)
                    x_noisy  = torch.sqrt(ab) * bx + torch.sqrt(1 - ab) * eps
                    pred_eps = self.model(x_noisy, t.float() / self.num_steps, bc)
                    val_loss = F.mse_loss(pred_eps, eps).item()

                if (epoch + 1) % 50 == 0:
                    print(f"    Epoch {epoch+1}/{epochs} - "
                          f"Train: {tr_loss/n_train:.6f}  Val: {val_loss:.6f}")

                # FIX Bug 7: early stopping
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss  = val_loss
                    patience_count = 0
                    # Save best weights
                    self._best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                else:
                    patience_count += 1
                    if patience_count >= patience:
                        print(f"    Early stopping at epoch {epoch+1}")
                        break
            else:
                if (epoch + 1) % 50 == 0:
                    print(f"    Epoch {epoch+1}/{epochs} - Loss: {tr_loss/n_train:.6f}")

        # Restore best weights
        if hasattr(self, "_best_state"):
            self.model.load_state_dict(self._best_state)

    def sample_trajectories(self, cond: torch.Tensor, num_traj: int = 64) -> torch.Tensor:
        """DDPM reverse diffusion — correct formula (Ho et al., 2020)."""
        self.model.eval()
        cond = cond.view(1, -1).expand(num_traj, -1).to(self.device)
        x = torch.randn(num_traj, self.data_dim, device=self.device)

        with torch.no_grad():
            for step in reversed(range(self.num_steps)):
                t      = torch.full((num_traj,), step, device=self.device).float() / self.num_steps
                alpha_t     = self.alpha[step]
                alpha_bar_t = self.alpha_bar[step]
                beta_t      = self.beta[step]

                eps_pred = self.model(x, t, cond)
                # Clamp predicted noise to ±3 to prevent exploding denoising steps.
                # An undertrained score network can output large epsilon values that
                # compound across 100 reverse steps → trajectories with magnitude >>1
                # in scaled space → thousands-of-percent returns after inverse_transform.
                eps_pred = torch.clamp(eps_pred, -3.0, 3.0)

                # Correct DDPM posterior mean (Ho et al. eq. 11)
                x = (1.0 / torch.sqrt(alpha_t)) * (
                    x - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred
                )
                # Clamp x at each step to keep trajectories in a sane range
                x = torch.clamp(x, -5.0, 5.0)
                if step > 0:
                    x = x + torch.sqrt(beta_t) * torch.randn_like(x)

        return x.detach()
