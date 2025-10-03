from dataclasses import dataclass, field
from typing import Tuple, Optional
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import itertools
import sys
import logging

# 配置 logger（只需在程序入口处配置一次）
logging.basicConfig(
    level=logging.INFO,  # 控制日志等级
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ------------------------------
# Utils
# ------------------------------

def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x

def set_seed(seed: int = 0, deterministic: bool = True):
    """Set all RNGs and deterministic flags for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

def beta_dist(gen_shape:tuple, alpha, beta, min_bound, max_bound, device):
    """Sample from a Beta distribution and linearly map to [min_bound, max_bound]."""
    beta_d = torch.distributions.Beta(concentration1=alpha, concentration0=beta)
    samples = to_device(beta_d.sample(gen_shape), device)
    gen_data = min_bound + (max_bound - min_bound) * samples
    return gen_data

# Angle helpers for 3D spherical coordinates
def wrap_phi_0_2pi(phi: torch.Tensor) -> torch.Tensor:
    """Wrap azimuth angle φ to [0, 2π)."""
    two_pi = 2.0 * math.pi
    return torch.remainder(phi, two_pi)

def clamp_theta_0_pi(theta: torch.Tensor) -> torch.Tensor:
    """Clamp polar angle θ to [0, π]."""
    return torch.clamp(theta, 0.0, math.pi)

# ------------------------------
# Dataset: Latent-Controlled Spheres (3D)
# ------------------------------

class LatentControlledSpheres3D(Dataset):
    """
    Synthetic 3D spherical shells controlled by latent (r, theta, phi).
    theta in [0, π], phi in [0, 2π).
    """
    def __init__(
        self,
        n: int = 50000,
        centers=(0.5, 1.0, 1.5),
        sigma_r: float = 0.05,
        sigma_x: float = 0.02,
        return_z: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        g = torch.Generator().manual_seed(seed)

        centers = torch.tensor(centers, dtype=torch.float32)
        idx = torch.randint(low=0, high=len(centers), size=(n,), generator=g)
        Rc = centers[idx]
        r = torch.normal(mean=Rc, std=sigma_r, generator=g)

        phi = torch.rand(n, generator=g) * 2.0 * math.pi
        u = torch.rand(n, generator=g) * 2.0 - 1.0   # cos(theta) ~ Uniform[-1,1]
        theta = torch.acos(u)

        sin_theta = torch.sin(theta)

        x = torch.stack([
            r * sin_theta * torch.cos(phi),
            r * sin_theta * torch.sin(phi),
            r * torch.cos(theta)
        ], dim=-1)

        x = x + torch.normal(mean=0.0, std=sigma_x, size=x.shape, generator=g)

        self.x = x
        self.z = torch.stack([r, theta, phi], dim=-1)
        self.return_z = return_z

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        # Keep returning the global index i for bank compatibility
        if self.return_z:
            return self.x[i], self.z[i], i
        return self.x[i], i


# ------------------------------
# Energy network (3D)
# ------------------------------

class EnergyMLP(nn.Module):
    """
    Energy function E(x, z). For 3D we use spherical features:
    z = (r, theta, phi) -> [r, sin(theta), cos(theta), sin(phi), cos(phi)].
    """
    def __init__(self, x_dim, z_dim, hidden=128, depth=3, lam_x: float = 0.0, lam_r: float = 1e-3):
        super().__init__()
        self.use_spherical = True
        in_dim = x_dim + (5 if self.use_spherical else z_dim)
        layers = []
        for d in range(depth):
            layers += [nn.Linear(in_dim if d==0 else hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        self.lam_x, self.lam_r = lam_x, lam_r
        self.register_buffer("alpha_const", torch.tensor(1.0))

        # Optional radial floor/walls (unused by default, kept for parity)
        self.r_floor = 0.05
        self.floor_margin = 0.05
        self.lam_wall = 4

    def forward(self, x, z):
        if self.use_spherical:
            r, theta, phi = z[:, 0], z[:, 1], z[:, 2]
            z_feat = torch.stack([r, torch.sin(theta), torch.cos(theta), torch.sin(phi), torch.cos(phi)], dim=-1)
        else:
            z_feat = z
            r = z[:, 0]
        core = self.net(torch.cat([x, z_feat], dim=-1)).squeeze(-1)

        # Mild quadratic prior on r only
        prior = 0.5 * self.lam_r * (r**2)
        return self.alpha_const * core + prior


# ------------------------------
# Decoder network (3D)
# ------------------------------

class DecoderMLP(nn.Module):
    """
    Simple MLP decoder: input z=(r, theta, phi) (spherical features -> [r, sinθ, cosθ, sinφ, cosφ]) -> output x in R^3.
    """
    def __init__(self, z_dim=3, x_dim=3, hidden=128, depth=3):
        super().__init__()
        self.use_spherical = True
        in_dim = 5 if self.use_spherical else z_dim
        layers = []
        for d in range(depth):
            layers += [nn.Linear(in_dim if d==0 else hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, x_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        if self.use_spherical:
            r, theta, phi = z[:, 0], z[:, 1], z[:, 2]
            z_feat = torch.stack([r, torch.sin(theta), torch.cos(theta), torch.sin(phi), torch.cos(phi)], dim=-1)
        else:
            z_feat = z
        return self.net(z_feat)


# ------------------------------
# Particle buffers (3D)
# ------------------------------

@dataclass
class JointBufferConfig:
    size: int = 8192
    k_steps: int = 40
    step_size_x: float = 1e-2
    step_size_z: float = 1e-2
    noise_scale: float = 1.0
    reinit_prob: float = 0.05
    x_init_std: float = 1.0
    z_init_std: float = 1.0
    x_min: float = -2.0
    x_max: float = 2.0
    r_min: float = 0.2
    r_max: float = 2.0
    dist_alpha: float = 2.0
    dist_beta: float = 2.0
    boundary_margin: float = 0.05

class JointParticleBuffer:
    """Persistent buffer for negative (x,z) particles; updated via short-run Langevin. 3D version."""
    def __init__(self, cfg: JointBufferConfig, x_dim:int, z_dim:int, device=None, seed:int=0):
        self.cfg = cfg
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device or default_device()
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

        # x ~ Beta box in R^3
        self.x = beta_dist((cfg.size, x_dim), cfg.dist_alpha, cfg.dist_beta, cfg.x_min, cfg.x_max, self.device)
        # z = (r, theta, phi)
        r = beta_dist((cfg.size, 1), cfg.dist_alpha, cfg.dist_beta, cfg.r_min, cfg.r_max, self.device)
        theta = beta_dist((cfg.size, 1), cfg.dist_alpha, cfg.dist_beta, 0.0, math.pi, self.device)
        phi = beta_dist((cfg.size, 1), cfg.dist_alpha, cfg.dist_beta, 0.0, 2.0*math.pi, self.device)
        self.z = torch.cat([r, theta, phi], dim=-1)

    @torch.no_grad()
    def refresh(self, energy: EnergyMLP):
        K = self.cfg.k_steps
        eta_x = self.cfg.step_size_x
        eta_z = self.cfg.step_size_z
        noise_x = math.sqrt(2.0 * eta_x) * self.cfg.noise_scale
        noise_z = math.sqrt(2.0 * eta_z) * self.cfg.noise_scale
        for _ in range(K):
            self.x = self.x.detach().requires_grad_(True)
            self.z = self.z.detach().requires_grad_(True)
            with torch.enable_grad():
                e = energy(self.x, self.z).sum()
                grad_x, grad_z = torch.autograd.grad(e, (self.x, self.z), create_graph=False)
            self.x = (self.x - eta_x * grad_x + noise_x * torch.randn(self.x.shape, device=self.x.device, dtype=self.x.dtype, generator=self.gen)).detach()
            self.z = (self.z - eta_z * grad_z + noise_z * torch.randn(self.z.shape, device=self.z.device, dtype=self.z.dtype, generator=self.gen)).detach()

            # Constrain z = (r, theta, phi)
            self.z[:, 0].clamp_(self.cfg.r_min, self.cfg.r_max)   # r
            self.z[:, 1] = clamp_theta_0_pi(self.z[:, 1])         # theta
            self.z[:, 2] = wrap_phi_0_2pi(self.z[:, 2])           # phi

    @torch.no_grad()
    def part_reinit(self, mask:torch.Tensor=None):
        # Random reinit
        if mask is None:
            mask = torch.rand(self.x.shape[0], device=self.device, generator=self.gen) < self.cfg.reinit_prob

        if mask.any():
            num = int(mask.sum().item())
            self.x[mask] = beta_dist((num, self.x_dim), self.cfg.dist_alpha, self.cfg.dist_beta, self.cfg.x_min, self.cfg.x_max, self.device)
            r_new = beta_dist((num,), self.cfg.dist_alpha, self.cfg.dist_beta, self.cfg.r_min, self.cfg.r_max, self.device)
            theta_new = beta_dist((num,), self.cfg.dist_alpha, self.cfg.dist_beta, 0.0, math.pi, self.device)
            phi_new = beta_dist((num,), self.cfg.dist_alpha, self.cfg.dist_beta, 0.0, 2.0*math.pi, self.device)
            self.z[mask, :] = torch.stack([r_new, theta_new, phi_new], dim=-1)

    def sample(self, n:int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(low=0, high=self.cfg.size, size=(n,), device=self.device, generator=self.gen)
        return self.x[idx], self.z[idx]

@dataclass
class LatentBankConfig:
    m_per_example: int = 4
    k_steps: int = 10
    step_size: float = 1e-2
    noise_scale: float = 1.0
    z_init_std: float = 1.0

    r_min: float = 0.2
    r_max: float = 2.0
    reinit_prob: float = 0.10
    dist_alpha: float = 2.0
    dist_beta: float = 2.0

class LatentParticleBank:
    """
    Stores M latent particles per example. 3D spherical (r, theta, phi).
    """
    def __init__(self, cfg: LatentBankConfig, n_examples:int, z_dim:int, device=None, seed:int=0):
        self.cfg = cfg
        self.n_examples = n_examples
        self.z_dim = z_dim
        self.device = device or default_device()
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

        B, M = n_examples, cfg.m_per_example
        r = beta_dist((B, M), cfg.dist_alpha, cfg.dist_beta, cfg.r_min, cfg.r_max, self.device)
        theta = beta_dist((B, M), cfg.dist_alpha, cfg.dist_beta, 0.0, math.pi, self.device)
        phi = beta_dist((B, M), cfg.dist_alpha, cfg.dist_beta, 0.0, 2.0*math.pi, self.device)
        self.z = torch.stack([r, theta, phi], dim=-1)  # [B, M, 3]

    @torch.no_grad()
    def update_for_batch(self, energy: EnergyMLP, x_batch: torch.Tensor, idx_batch: torch.Tensor):
        M, K, eta = self.cfg.m_per_example, self.cfg.k_steps, self.cfg.step_size
        noise = math.sqrt(2.0 * eta) * self.cfg.noise_scale

        z_part = self.z[idx_batch]  # [B, M, 3]
        B = x_batch.shape[0]
        x_rep = x_batch[:, None, :].expand(B, M, x_batch.shape[-1]).reshape(B * M, -1)
        z_flat = z_part.reshape(B * M, 3).contiguous()

        for _ in range(K):
            z_flat = z_flat.detach().requires_grad_(True)
            with torch.enable_grad():
                e = energy(x_rep, z_flat).sum()
                (grad_z,) = torch.autograd.grad(e, (z_flat,), create_graph=False)

            z_flat = (z_flat - eta * grad_z + noise * torch.randn(z_flat.shape, device=z_flat.device, dtype=z_flat.dtype, generator=self.gen)).detach()

            # Constrain (r, theta, phi)
            z_flat[:, 0].clamp_(self.cfg.r_min, self.cfg.r_max)       # r
            z_flat[:, 1] = clamp_theta_0_pi(z_flat[:, 1])             # theta
            z_flat[:, 2] = wrap_phi_0_2pi(z_flat[:, 2])               # phi

        self.z[idx_batch] = z_flat.view(B, M, -1)

    @torch.no_grad()
    def part_reinit(self, mask:torch.Tensor=None):
        B, M = self.z.shape[0], self.z.shape[1]
        z_flat = self.z.reshape(B*M, -1)

        # Random reinit
        if mask is None:
            mask = torch.rand(z_flat.shape[0], device=self.device, generator=self.gen) < self.cfg.reinit_prob
        else:
            mask = mask.to(self.device)

        if mask.any():
            num = int(mask.sum().item())
            r_new = beta_dist((num,), self.cfg.dist_alpha, self.cfg.dist_beta, self.cfg.r_min, self.cfg.r_max, self.device)
            theta_new = beta_dist((num,), self.cfg.dist_alpha, self.cfg.dist_beta, 0.0, math.pi, self.device)
            phi_new = beta_dist((num,), self.cfg.dist_alpha, self.cfg.dist_beta, 0.0, 2.0*math.pi, self.device)
            z_flat[mask, :] = torch.stack([r_new, theta_new, phi_new], dim=-1)

        self.z = z_flat.reshape(B, M, -1)

    def sample_for_batch(self, idx_batch: torch.Tensor) -> torch.Tensor:
        """Return latent particles [r, theta, phi] for the batch indices."""
        return self.z[idx_batch]


# ------------------------------
# Trainer (unchanged logic, 3D shapes)
# ------------------------------

@dataclass
class TrainerConfig:
    x_dim: int = 3
    z_dim: int = 3
    joint: JointBufferConfig = field(default_factory=JointBufferConfig)
    bank: LatentBankConfig = field(default_factory=LatentBankConfig)
    lr: float = 2e-3
    lr_min: float = 2e-4
    warmup_steps: int = 300
    weight_decay: float = 0.0
    steps: int = 20000
    batch_size: int = 128
    log_interval: int = 100
    device: Optional[torch.device] = None
    seed: int = 0
    lambda_nce: float = 1.0
    lambda_joint: float = 1.0
    tau_init: float = 1.0
    tau_min: float = 0.1
    use_infoNCE: bool = False
    output_dir: str = "."

class LVEBMTrainer:
    def __init__(
        self,
        energy: EnergyMLP,
        dataset: Dataset,
        cfg: TrainerConfig = TrainerConfig(),
        train_idx: Optional[torch.Tensor] = None,
        test_idx: Optional[torch.Tensor] = None,
    ):
        self.energy = energy
        self.cfg = cfg
        self.device = cfg.device or default_device()
        self.energy.to(self.device)
        set_seed(cfg.seed, deterministic=True)

        # ---------- Train/Test split (80/20) ----------
        if (train_idx is not None) and (test_idx is not None):
            self.train_idx = train_idx.clone().cpu()
            self.test_idx  = test_idx.clone().cpu()
        else:
            gsplit = torch.Generator().manual_seed(cfg.seed + 2024)
            n = len(dataset)
            perm = torch.randperm(n, generator=gsplit)
            n_test = max(1, int(0.2 * n))
            self.test_idx = perm[:n_test].cpu()
            self.train_idx = perm[n_test:].cpu()

        self.dataset = dataset

        self.train_sampler = SubsetRandomSampler(self.train_idx.tolist(), generator=torch.Generator().manual_seed(cfg.seed + 555))
        self.test_sampler  = SubsetRandomSampler(self.test_idx.tolist(),  generator=torch.Generator().manual_seed(cfg.seed + 556))

        self.train_loader = DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=True,
            sampler=self.train_sampler, num_workers=0
        )
        self.test_loader = DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
            sampler=self.test_sampler, num_workers=0
        )

        # Banks
        self.joint = JointParticleBuffer(cfg.joint, x_dim=cfg.x_dim, z_dim=cfg.z_dim, device=self.device, seed=cfg.seed+1)
        self.bank  = LatentParticleBank(cfg.bank, n_examples=len(dataset), z_dim=cfg.z_dim, device=self.device, seed=cfg.seed+2)

        # Optimizer & cosine LR with warmup
        self.opt = torch.optim.Adam(self.energy.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        total, warmup, lr0, lr_min = cfg.steps, max(0, cfg.warmup_steps), cfg.lr, cfg.lr_min
        min_factor = lr_min / lr0
        def lr_lambda(step):
            if step < warmup and warmup > 0:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total - warmup)
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
            return min_factor + (1.0 - min_factor) * cos_factor
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=lr_lambda)

        # Eval set prepared from the test split
        self._prepare_eval_set(n_eval=1024)
        with torch.no_grad():
            x_pos_e, _, _, _ = self.eval_set
            self.r_real_eval = x_pos_e.norm(dim=1)

    # --- posterior probe used for evaluation and for decoder training (do not touch bank) ---
    @torch.no_grad()
    def _posterior_probe(self, x_batch: torch.Tensor) -> torch.Tensor:
        """
        Given a batch of x, initialize (r,theta,phi) via independent Beta mappings and run K-step ULA
        in latent space. Return z_post with shape [B, M, 3]. Does NOT modify internal buffers.
        """
        device = x_batch.device
        B = x_batch.shape[0]
        M = self.cfg.bank.m_per_example
        K = self.cfg.bank.k_steps
        eta = self.cfg.bank.step_size
        noise_scale = self.cfg.bank.noise_scale
        r_min, r_max = self.cfg.bank.r_min, self.cfg.bank.r_max

        g = torch.Generator(device=device).manual_seed(self.cfg.seed + 33333)

        r = beta_dist((B, M), self.bank.cfg.dist_alpha, self.bank.cfg.dist_beta, self.bank.cfg.r_min, self.bank.cfg.r_max, device)
        theta = beta_dist((B, M), self.bank.cfg.dist_alpha, self.bank.cfg.dist_beta, 0.0, math.pi, device)
        phi = beta_dist((B, M), self.bank.cfg.dist_alpha, self.bank.cfg.dist_beta, 0.0, 2.0*math.pi, device)
        z = torch.stack([r, theta, phi], dim=-1).reshape(B * M, 3)  # [B*M, 3]

        x_rep = x_batch[:, None, :].expand(B, M, x_batch.shape[-1]).reshape(B * M, -1)
        noise = math.sqrt(2.0) * math.sqrt(eta) * noise_scale

        for _ in range(K):
            z = z.detach().requires_grad_(True)
            with torch.enable_grad():
                e = self.energy(x_rep, z).sum()
                (grad_z,) = torch.autograd.grad(e, (z,), create_graph=False)

            z = (z - eta * grad_z + noise * torch.randn(z.shape, device=device, dtype=z.dtype, generator=g)).detach()

            # Constrain z
            z[:, 0].clamp_(r_min, r_max)                 # r
            z[:, 1] = clamp_theta_0_pi(z[:, 1])          # theta
            z[:, 2] = wrap_phi_0_2pi(z[:, 2])            # phi

        return z.view(B, M, 3)

    def _prepare_eval_set(self, n_eval: int = 1024):
        """
        Build eval set from the test split (20% hold-out). Falls back to whole dataset if needed.
        """
        if hasattr(self, "test_idx") and self.test_idx.numel() > 0:
            ix_pool = self.test_idx
        else:
            ix_pool = torch.arange(len(self.dataset))

        g = torch.Generator(device='cpu').manual_seed(self.cfg.seed + 12345)
        n_pick = min(n_eval, ix_pool.numel())
        perm = torch.randperm(ix_pool.numel(), generator=g)[:n_pick]
        ix = ix_pool[perm]
        self.eval_ix = ix.to(self.device)
        x_pos_eval = to_device(self.dataset.x[ix], self.device)
        z_pos_eval = to_device(self.dataset.z[ix], self.device)

        # Mismatched negatives: radial distortion depending on phi, keep same (theta,phi)
        z_neg_eval = z_pos_eval.detach().clone()
        g_dev = torch.Generator(device=self.device).manual_seed(self.cfg.seed + 12346)
        r   = z_pos_eval[:, 0]
        theta = z_pos_eval[:, 1]
        phi = z_pos_eval[:, 2]
        alpha = 0.35; k = 2
        scale = 1.0 + alpha * torch.cos(k * phi)
        r_bad = r * scale
        x1 = r_bad * torch.sin(theta) * torch.cos(phi)
        x2 = r_bad * torch.sin(theta) * torch.sin(phi)
        x3 = r_bad * torch.cos(theta)
        x_neg_eval = torch.stack([x1, x2, x3], dim=-1)
        x_neg_eval = x_neg_eval + torch.normal(0.0, 0.2, size=x_neg_eval.shape, generator=g_dev, device=self.device)

        self.eval_set = (x_pos_eval, z_pos_eval, x_neg_eval, z_neg_eval)

    # ------------------------------
    # Core energy training loop
    # ------------------------------
    def train(self):
        self.energy.train()
        for p in self.energy.parameters():
            if not p.requires_grad:
                p.requires_grad_(True)
        data_iter = iter(self.train_loader)
        for step in range(1, self.cfg.steps + 1):

            # if step == 1:
            #     model_save_path = os.path.join(self.cfg.output_dir, f"energy_model_step_{step}.pt")
            #     torch.save(self.energy.state_dict(), model_save_path)
            #     print(f"[INFO] Model saved to {model_save_path}")

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader); batch = next(data_iter)

            if len(batch) == 3:
                x, _, idx = batch
            else:
                x, idx = batch
            x = to_device(x, self.device)
            idx = to_device(idx, self.device)

            # (1) conditional latent updates
            self.bank.update_for_batch(self.energy, x_batch=x, idx_batch=idx)
            # (2) refresh joint negatives
            self.joint.refresh(self.energy)

            # (3) objective (free energy positive phase)
            M = self.cfg.bank.m_per_example
            z_pos = self.bank.sample_for_batch(idx)                          # [B, M, 3]
            x_pos = x[:, None, :].expand(x.shape[0], M, x.shape[-1])         # [B, M, 3]

            tau0, tau_min = 1.0, 0.1
            progress = (step-1) / max(1, self.cfg.steps-1)
            tau = tau_min + (tau0 - tau_min) * 0.5 * (1 + math.cos(math.pi * progress))
            Epos = self.energy(x_pos.reshape(-1, x_pos.shape[-1]),
                               z_pos.reshape(-1, z_pos.shape[-1])).view(x.shape[0], M)
            e_pos = (-tau * torch.logsumexp(-Epos / tau, dim=1) + math.log(M)).mean()

            x_neg, z_neg = self.joint.sample(n=x.shape[0])
            e_neg = self.energy(x_neg, z_neg).mean()

            objective = e_neg - e_pos
            loss = F.softplus(e_pos - e_neg)

            if self.cfg.use_infoNCE:
                # InfoNCE: encourage matching x_i with its soft posterior average z_soft_i
                w = torch.softmax(-Epos / tau, dim=1)                 # [B,M]
                z_soft = (w.unsqueeze(-1) * z_pos).sum(dim=1)         # [B,3]
                B = x.shape[0]
                X_rep = x[:,None,:].expand(B,B,3).reshape(-1,3)
                Z_rep = z_soft[None,:,:].expand(B,B,3).reshape(-1,3)
                S = -self.energy(X_rep, Z_rep).view(B,B)              # [B,B]
                loss_nce = (torch.logsumexp(S, dim=1) - S.diag()).mean()
                loss = loss + self.cfg.lambda_nce * loss_nce

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.energy.parameters(), max_norm=5.0)
            self.opt.step()
            self.sched.step()
            
            logger.info(f"Step {step}: loss = {loss.item():.6f}")

            # Reinit out-of-bound particles for stability
            with torch.no_grad():
                low_val = self.joint.cfg.r_min * (1 + self.joint.cfg.boundary_margin)
                high_val = self.joint.cfg.r_max * (1 - self.joint.cfg.boundary_margin)

                # Bad Particles for Joint
                z_batch = self.joint.z
                r = z_batch[..., 0]
                low = torch.full_like(r, low_val).to(self.device)
                high = torch.full_like(r, high_val).to(self.device)
                bad_examples = ( (r <= low) | (r >= high) )   # [B] bool
                if bad_examples.any():
                    self.joint.part_reinit(bad_examples)

                # Bad for LatentBank
                z_bank_b = self.bank.z[idx]              # [B, M, 3]
                r_bank_b = z_bank_b[..., 0]              # [B, M]
                low_b = torch.full_like(r_bank_b, low_val).to(self.device)
                high_b = torch.full_like(r_bank_b, high_val).to(self.device)
                bad_bank_b = ((r_bank_b <= low_b) | (r_bank_b >= high_b))   # [B, M] bool

                if bad_bank_b.any():
                    N_examples = self.bank.z.shape[0]
                    M_bank = self.bank.z.shape[1]
                    mask_global = torch.zeros((N_examples, M_bank),
                                            dtype=torch.bool, device=self.device)
                    mask_global[idx] = bad_bank_b
                    mask_flat = mask_global.reshape(-1)   # [N_examples*M]
                    self.bank.part_reinit(mask_flat)

            if step % int(0.5*self.cfg.log_interval) == 0:
                self.bank.part_reinit()
                self.joint.part_reinit()

            if step % self.cfg.log_interval == 0:
                print_str = f"[step {step:6d}] obj={objective.item(): .4f}  E-={e_neg.item(): .4f}  E+={e_pos.item(): .4f}"
                with torch.no_grad():
                    x_pos_e, z_pos_e, x_neg_e, z_neg_e = self.eval_set
                    epos = self.energy(x_pos_e, z_pos_e)
                    eneg = self.energy(x_neg_e, z_neg_e)
                    obj_eval = (eneg.mean() - epos.mean()).item()
                    rank_acc = (epos[:, None] < eneg[None, :]).float().mean().item()

                visualize_distributions(self, step, self.cfg.output_dir)

                if step % 500 == 0:
                    model_save_path = os.path.join(self.cfg.output_dir, f"energy_model_step_{step}.pt")
                    torch.save(self.energy.state_dict(), model_save_path)
                    print(f"[INFO] Model saved to {model_save_path}")

                print_str += f"  | eval_gap={obj_eval: .4f}  rank_acc={rank_acc: .3f}  "
                print(print_str)
        print("Training finished.")


# ------------------------------
# Visualization & posterior extraction (3D)
# ------------------------------

@torch.no_grad()
def visualize_distributions(trainer: "LVEBMTrainer", step:int=0, output_dir: str = "."):
    """
    Visualization uses the test split only. In 3D we scatter GT vs reconstructions.
    """
    device = trainer.device
    ds = trainer.dataset
    test_idx = trainer.test_idx
    X = ds.x[test_idx].to(device)

    # Collect posterior particles on test split only
    Z_particles = collect_posterior_particles(
        trainer.energy, ds, trainer.cfg.bank, device=device,
        seed=trainer.cfg.seed + 777, indices=test_idx
    )  # [N_test, M, 3]

    # z_mean = aggregate_from_particles(X, Z_particles, trainer.energy, how="mean")
    z_map  = aggregate_from_particles(X, Z_particles, trainer.energy, how="map")

    # x_hat_mean = reconstruct_x(z_mean[:, 0], z_mean[:, 1], z_mean[:, 2])  # [N_test,3]
    x_hat_map  = reconstruct_x(z_map[:, 0],  z_map[:, 1],  z_map[:, 2])

    # plot_xyz(f"{step} x̂ from z' (mean)", x_hat_mean, output_dir=output_dir)
    plot_xyz(f"{step} x̂ from z' (MAP)",  x_hat_map,  output_dir=output_dir)

@torch.no_grad()
def reconstruct_x(r: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Given (r, theta, phi), return x=[r sinθ cosφ, r sinθ sinφ, r cosθ]."""
    theta = clamp_theta_0_pi(theta)
    phi = wrap_phi_0_2pi(phi)
    sin_t = torch.sin(theta)
    x0 = r * sin_t * torch.cos(phi)
    x1 = r * sin_t * torch.sin(phi)
    x2 = r * torch.cos(theta)
    return torch.stack([x0, x1, x2], dim=-1)

def plot_xyz(title: str, X: torch.Tensor, max_points: int = 15000, output_dir : str = "."):
    """3D scatter plot for points (tensor of shape [N,3])."""
    xs = X.detach().cpu()
    if xs.shape[0] > max_points:
        idx = torch.randperm(xs.shape[0])[:max_points]
        xs = xs[idx]
    fig = plt.figure(figsize=(5.6, 5.6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs[:, 0].numpy(), xs[:, 1].numpy(), xs[:, 2].numpy(), s=3, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel('x[0]'); ax.set_ylabel('x[1]'); ax.set_zlabel('x[2]')
    lim = float(xs.abs().max()) * 1.05 + 1e-6
    ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim]); ax.set_zlim([-lim, lim])
    ax.view_init(elev=20, azim=35)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{title}.png"))
    plt.close()

@torch.no_grad()
def aggregate_from_particles(x: torch.Tensor, z_M: torch.Tensor, energy: EnergyMLP, how: str = "map"):
    """
    Aggregate posterior candidates for each x.
    For 'map': pick argmin E(x,z).
    For 'mean': average r, and compute spherical mean of angles (θ, φ) via unit-vector averaging.
    """
    N, M, _ = z_M.shape
    if how == "map":
        x_rep = x[:, None, :].expand(N, M, x.shape[-1]).reshape(-1, x.shape[-1])
        z_flat = z_M.reshape(-1, 3)
        e = energy(x_rep, z_flat).view(N, M)
        idx_min = e.argmin(dim=1)
        z_hat = z_M[torch.arange(N, device=z_M.device), idx_min, :]
        # Ensure angles are in canonical ranges
        z_hat = torch.stack([z_hat[:, 0], clamp_theta_0_pi(z_hat[:, 1]), wrap_phi_0_2pi(z_hat[:, 2])], dim=-1)
        return z_hat
    elif how == "mean":
        r_hat = z_M[:, :, 0].mean(dim=1)
        theta_M = z_M[:, :, 1]
        phi_M = z_M[:, :, 2]
        # Unit-vector average on the sphere
        s_t = torch.sin(theta_M)
        X = (s_t * torch.cos(phi_M)).mean(dim=1)
        Y = (s_t * torch.sin(phi_M)).mean(dim=1)
        Z = (torch.cos(theta_M)).mean(dim=1)
        # Convert back to (theta, phi)
        eps = 1e-8
        norm = torch.sqrt(X*X + Y*Y + Z*Z + eps)
        Zc = torch.clamp(Z / (norm + eps), -1.0, 1.0)
        theta_hat = torch.acos(Zc)
        phi_hat = torch.atan2(Y, X)
        phi_hat = wrap_phi_0_2pi(phi_hat)
        return torch.stack([r_hat, theta_hat, phi_hat], dim=-1)
    else:
        raise ValueError("how must be 'map' or 'mean'")

@torch.no_grad()
def collect_posterior_particles(
    energy: EnergyMLP,
    dataset: Dataset,
    bank_cfg: LatentBankConfig,
    device=None,
    seed: int = 123,
    indices: Optional[torch.Tensor] = None,
):
    """
    Run conditional Langevin over a chosen subset of the dataset (default: whole dataset)
    and return transformed particles with shape [N_subset, M, 3] in the same order as indices.
    """
    device = device or default_device()
    energy.eval()

    full_N = len(dataset)
    bank = LatentParticleBank(bank_cfg, n_examples=full_N, z_dim=3, device=device, seed=seed)

    if indices is not None:
        indices = indices.clone().cpu()
        sampler = SubsetRandomSampler(indices.tolist(), generator=torch.Generator().manual_seed(seed + 2025))
        loader = DataLoader(dataset, batch_size=2048, shuffle=False, drop_last=False, num_workers=0, sampler=sampler)
    else:
        loader = DataLoader(dataset, batch_size=2048, shuffle=False, drop_last=False, num_workers=0)

    for batch in loader:
        if len(batch) == 3:
            x, _, idx = batch
        else:
            x, idx = batch
        x = x.to(device); idx = idx.to(device)
        bank.update_for_batch(energy, x_batch=x, idx_batch=idx)

    if indices is None:
        all_idx = torch.arange(full_N, device=device)
        return bank.sample_for_batch(all_idx)  # [N_all, M, 3]
    else:
        return bank.sample_for_batch(indices.to(device))  # [N_subset, M, 3]

# ------------------------------
# Decoder training helpers (3D)
# ------------------------------

@torch.no_grad()
def map_z_for_batch(energy: EnergyMLP, trainer: LVEBMTrainer, x_batch: torch.Tensor) -> torch.Tensor:
    """
    For each x in the batch, sample M candidates via trainer._posterior_probe(x),
    then pick the MAP z (argmin energy(x, z)).
    Returns: z_map [B, 3]
    """
    energy.eval()
    z_samps = trainer._posterior_probe(x_batch)  # [B, M, 3]
    B, M, _ = z_samps.shape
    x_rep = x_batch[:, None, :].expand(B, M, x_batch.shape[-1]).reshape(B * M, -1)
    z_flat = z_samps.reshape(B * M, -1)
    E = energy(x_rep, z_flat).view(B, M)  # [B, M]
    idx = torch.argmin(E, dim=1)          # [B]
    z_map = z_samps[torch.arange(B, device=x_batch.device), idx]
    # canonicalize
    z_map = torch.stack([z_map[:,0], clamp_theta_0_pi(z_map[:,1]), wrap_phi_0_2pi(z_map[:,2])], dim=-1)
    return z_map

def train_decoder(
    trainer: LVEBMTrainer,
    energy: EnergyMLP,
    decoder: DecoderMLP,
    epochs: int = 200,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    patience: int = 20,
    log_every: int = 1,
    save_plot_path: str = "recon_vs_gt.png",
    data_save_path: str = "recon_data.pt",
):
    """
    Train decoder on train split with frozen energy. Each step:
    - Probe posterior candidates z' for x via trainer._posterior_probe
    - Take MAP z' by energy minimization
    - Decode x' = f_dec(z'), minimize MSE(x', x)
    Validate on held-out test split each epoch; use early stopping.
    Finally, reconstruct test set and save GT vs reconstruction 3D scatter plot.
    """
    device = trainer.device
    energy.eval()
    for p in energy.parameters():
        p.requires_grad_(False)

    decoder.to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=max(3, patience//3), verbose=False)
    best_val = float("inf")
    best_state = None
    bad_epochs = 0

    mse = nn.MSELoss()

    for ep in range(1, epochs + 1):
        # ---- Train ----
        decoder.train()
        tr_loss_acc = 0.0
        tr_count = 0

        for batch in trainer.train_loader:
            if len(batch) == 3:
                x, _, _ = batch
            else:
                x, _ = batch
            x = to_device(x, device)

            with torch.no_grad():
                z_map = map_z_for_batch(energy, trainer, x)  # [B, 3]

            x_hat = decoder(z_map)
            loss = mse(x_hat, x)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0)
            opt.step()

            tr_loss_acc += loss.item() * x.shape[0]
            tr_count += x.shape[0]

        tr_loss = tr_loss_acc / max(1, tr_count)

        # ---- Validation on test split ----
        decoder.eval()
        val_loss_acc = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in trainer.test_loader:
                if len(batch) == 3:
                    x, _, _ = batch
                else:
                    x, _ = batch
                x = to_device(x, device)
                z_map = map_z_for_batch(energy, trainer, x)
                x_hat = decoder(z_map)
                loss = mse(x_hat, x)
                val_loss_acc += loss.item() * x.shape[0]
                val_count += x.shape[0]
        val_loss = val_loss_acc / max(1, val_count)
        sched.step(val_loss)

        if ep % log_every == 0:
            print(f"[Decoder][epoch {ep:03d}] train_mse={tr_loss:.6f}  val_mse={val_loss:.6f}")

        # ---- Early stopping ----
        if val_loss + 1e-8 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[Decoder] Early stopping at epoch {ep} (best val={best_val:.6f}).")
                break

    if best_state is not None:
        decoder.load_state_dict(best_state)

    # ---- Final reconstruction on test split and plot ----
    X_gt = []
    X_rec = []
    decoder.eval()
    with torch.no_grad():
        for batch in trainer.test_loader:
            if len(batch) == 3:
                x, _, _ = batch
            else:
                x, _ = batch
            x = to_device(x, device)
            z_map = map_z_for_batch(energy, trainer, x)
            x_hat = decoder(z_map)
            X_gt.append(x.detach().cpu())
            X_rec.append(x_hat.detach().cpu())
    X_gt = torch.cat(X_gt, dim=0).numpy()
    X_rec = torch.cat(X_rec, dim=0).numpy()

    # Save data

    os.makedirs(os.path.dirname(data_save_path) or ".", exist_ok=True)
    np.savez(
        data_save_path,   # 建议用 .npz 结尾
        X_gt=X_gt,
        X_rec=X_rec,
    )
    print(f"[Data] Saved numpy arrays to: {data_save_path}")

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_gt[:,0],  X_gt[:,1],  X_gt[:,2],  s=6, alpha=0.5, label="GT (test)")
    ax.scatter(X_rec[:,0], X_rec[:,1], X_rec[:,2], s=6, alpha=0.5, label="Reconstruction")
    ax.legend()
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    lim = float(max(np.abs(X_gt).max(), np.abs(X_rec).max())) * 1.05 + 1e-6
    ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim]); ax.set_zlim([-lim, lim])
    ax.view_init(elev=20, azim=35)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_plot_path) or ".", exist_ok=True)
    plt.savefig(save_plot_path, dpi=150)
    print(f"[Decoder] Saved reconstruction plot to: {save_plot_path}")


# ------------------------------
# Main
# ------------------------------

if __name__ == "__main__":
    device = default_device()
    print("Using device:", device)

    # 3D dataset
    ds = LatentControlledSpheres3D(n=50000, return_z=False, seed=77)

    # Explicit 80/20 split (shared by trainer and decoder stages)
    gsplit = torch.Generator().manual_seed(77)
    n_total = len(ds)
    perm = torch.randperm(n_total, generator=gsplit)
    n_test = max(1, int(0.2 * n_total))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    # r_min_list = [0.0]
    # r_max_list = [10.0]
    # k_steps_list = [20, 25, 30]
    # step_size_list = [1e-3, 2e-3, 8e-4]
    # noise_scale_list = [1.0]
    # seed_list = [77, 177, 277]
    # lr_list = [2e-3, 3e-3]

    # param_grid = list(itertools.product(r_min_list, r_max_list, k_steps_list, step_size_list, noise_scale_list, seed_list, lr_list))
    param_grid = [
        (0.0, 10, 20, 8e-4, 1.0, 77, 3e-3),
        (0.0, 10, 25, 8e-4, 1.0, 77, 2e-3),
        (0.0, 10, 25, 1e-3, 1.0, 277, 3e-3)
    ]    

    for i, (r_min, r_max, k_steps, step_size, noise_scale, base_seed, lr) in enumerate(param_grid):
        print(f"\n=== Grid {i+1}/{len(param_grid)}: r_min={r_min}, k_steps={k_steps}, step_size={step_size}, noise_scale={noise_scale}, seed={base_seed}, lr={lr} ===")

        set_seed(base_seed, deterministic=True)
        energy = EnergyMLP(x_dim=3, z_dim=3, hidden=128, depth=3, lam_x=0.0, lam_r=0.0)

        loaded_step = None
        # if len(sys.argv) > 1:
        #     loaded_step = sys.argv[1]
        #     energy.load_state_dict(torch.load(".../energy_model_step_{loaded_step}.pt"))

        cfg = TrainerConfig(
            x_dim=3, z_dim=3,
            steps=5000,
            lr=lr,
            batch_size=512,
            log_interval=100,
            seed=base_seed,
            use_infoNCE=True,
            joint=JointBufferConfig(
                size=4096, k_steps=k_steps, step_size_x=step_size, step_size_z=step_size,
                noise_scale=noise_scale, reinit_prob=0.02,
                x_min=-2.0, x_max=2.0, r_min=r_min, r_max=r_max
            ),
            bank=LatentBankConfig(
                m_per_example=16, k_steps=k_steps, step_size=step_size,
                noise_scale=noise_scale, r_min=r_min, r_max=r_max, reinit_prob=0.1
            ),
            output_dir=f"./loss_pic_3d/grid_{i+1}_rmin{r_min}_k{k_steps}_step{step_size}_noise{noise_scale}_seed{base_seed}_lr{lr}/"
        )

        trainer = LVEBMTrainer(energy=energy, dataset=ds, cfg=cfg, train_idx=train_idx, test_idx=test_idx)
        trainer.train()

        # -------- After energy training: freeze energy and train decoder --------
        energy.eval()
        for p in energy.parameters():
            p.requires_grad_(False)

        decoder = DecoderMLP(z_dim=3, x_dim=3, hidden=128, depth=3)

        train_decoder(
            trainer=trainer,
            energy=energy,
            decoder=decoder,
            epochs=15,
            lr=1e-3,
            weight_decay=0.0,
            patience=20,
            log_every=1,
            save_plot_path=f"./loss_pic_3d/grid_{i+1}_rmin{r_min}_k{k_steps}_step{step_size}_noise{noise_scale}_seed{base_seed}_lr{lr}/recon_vs_gt.png" if loaded_step is None else f"./maybe/grid_{i+1}_.../{loaded_step}_recon_vs_gt.png",
            data_save_path=f"loss_pic_3d/grid_{i+1}_rmin{r_min}_k{k_steps}_step{step_size}_noise{noise_scale}_seed{base_seed}_lr{lr}/recon_data.pt" if loaded_step is None else f"2d_best_res/grid_1_rmin0.0_k25_step0.0008_noise1.0_seed277_lr0.002/{loaded_step}_recon_data.pt"
        )
