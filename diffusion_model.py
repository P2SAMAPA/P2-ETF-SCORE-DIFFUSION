"""
Score‑Based Diffusion Model (DDPM) for ETF returns with macro conditioning.
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

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

class ResidualBlock(nn.Module):
    def __init__(self, dim, cond_dim, time_dim):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, dim)
        self.cond_proj = nn.Linear(cond_dim, dim)
        self.net = nn.Sequential(
            nn.Linear(dim + cond_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x, t_emb, cond):
        h = x + self.time_proj(t_emb)
        h = torch.cat([h, cond], dim=-1)
        return x + self.net(h)

class ScoreNetwork(nn.Module):
    def __init__(self, data_dim, cond_dim, hidden_dim=128, num_layers=4, time_dim=64):
        super().__init__()
        self.time_embed = TimeEmbedding(time_dim)
        self.proj = nn.Linear(data_dim + cond_dim, hidden_dim)
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, cond_dim, time_dim) for _ in range(num_layers)
        ])
        self.out = nn.Linear(hidden_dim, data_dim)

    def forward(self, x, t, cond):
        # x: (batch, data_dim), t: (batch,), cond: (batch, cond_dim)
        t_emb = self.time_embed(t)
        h = torch.cat([x, cond], dim=-1)
        h = self.proj(h)
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        self._build_schedule(noise_schedule)

    def _build_schedule(self, schedule):
        if schedule == "linear":
            self.beta = torch.linspace(1e-4, 0.02, self.num_steps).to(self.device)
        else:  # cosine
            t = torch.linspace(0, self.num_steps, self.num_steps + 1)
            self.beta = (1 - torch.cos((t[1:] / self.num_steps) * math.pi)) / 2 * 0.02 + 1e-4
            self.beta = self.beta.to(self.device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def _extract_into_tensor(self, arr, t):
        return arr[t].reshape(-1, 1)

    def fit(self, X, cond, epochs=200, batch_size=128):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        cond = torch.tensor(cond, dtype=torch.float32).to(self.device)
        n = len(X)
        dataset = torch.utils.data.TensorDataset(X, cond)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_c in loader:
                batch_x, batch_c = batch_x.to(self.device), batch_c.to(self.device)
                t = torch.randint(0, self.num_steps, (len(batch_x),), device=self.device)
                eps = torch.randn_like(batch_x)
                alpha_bar_t = self._extract_into_tensor(self.alpha_bar, t)
                x_noisy = torch.sqrt(alpha_bar_t) * batch_x + torch.sqrt(1 - alpha_bar_t) * eps
                pred_eps = self.model(x_noisy, t.float() / self.num_steps, batch_c)
                loss = F.mse_loss(pred_eps, eps)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item() * len(batch_x)
            if (epoch + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/n:.6f}")

    def sample_trajectories(self, cond: torch.Tensor, num_traj: int = 64) -> torch.Tensor:
        """Generate multiple trajectories for a given macro condition."""
        self.model.eval()
        batch_size = 1  # single condition, multiple trajectories
        cond = cond.view(1, -1).expand(num_traj, -1).to(self.device)
        x = torch.randn(num_traj, self.data_dim, device=self.device)
        for step in reversed(range(self.num_steps)):
            t = torch.full((num_traj,), step, device=self.device).float() / self.num_steps
            alpha = self.alpha[step]
            alpha_bar = self.alpha_bar[step]
            beta = self.beta[step]
            eps_pred = self.model(x, t, cond)
            if step > 0:
                z = torch.randn_like(x)
            else:
                z = torch.zeros_like(x)
            x = (x - beta / torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha)
            if step > 0:
                x += torch.sqrt(beta) * z
        return x
