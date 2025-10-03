from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import random
import numpy as np
import matplotlib.pyplot as plt
import itertools
import sys
import logging

# 配置 logger（只需在程序入口处配置一次）
logging.basicConfig(
    level=logging.INFO,  # 控制日志等级
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

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


# ------------------------------
# Dataset: Latent-Controlled Rings (LCR)
# ------------------------------

class LatentControlledRings(Dataset):
    """
    Synthetic 2D rings controlled by latent (r, phi). Observations:
      x = [r cos(phi), r sin(phi)] + noise, with r ~ N(R_c, sigma_r^2), phi ~ Uniform[0, 2pi).
    """
    def __init__(
        self,
        n:int=50000,
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
        phi = torch.rand(n, generator=g) * 2 * math.pi
        x = torch.stack([r * torch.cos(phi), r * torch.sin(phi)], dim=-1)
        x = x + torch.normal(mean=0.0, std=sigma_x, size=x.shape, generator=g)
        self.x = x
        self.z = torch.stack([r, phi], dim=-1)
        self.return_z = return_z

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        # IMPORTANT: keep returning the global index i, to remain compatible with banks.
        if self.return_z:
            return self.x[i], self.z[i], i
        return self.x[i], i


# ------------------------------
# Energy network
# ------------------------------

class EnergyMLP(nn.Module):
    def __init__(self, x_dim, z_dim, hidden=128, depth=3, lam_x: float = 0.0, lam_r: float = 1e-3):
        super().__init__()
        self.use_circular = True
        in_dim = x_dim + (3 if self.use_circular else z_dim)
        layers = []
        for d in range(depth):
            layers += [nn.Linear(in_dim if d==0 else hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)
        self.lam_x, self.lam_r = lam_x, lam_r
        self.register_buffer("alpha_const", torch.tensor(1.0))

        self.r_floor = 0.05
        self.floor_margin = 0.05
        self.lam_wall = 4

    def forward(self, x, z):
        if self.use_circular:
            r, phi = z[:, 0], z[:, 1]
            z_feat = torch.stack([r, torch.sin(phi), torch.cos(phi)], dim=-1)
        else:
            z_feat = z
            r = z[:, 0]
        core = self.net(torch.cat([x, z_feat], dim=-1)).squeeze(-1)

        # --- soft lower wall: penalize r close to r_floor ---
        # penalty grows when r < r_floor + margin, ~0 otherwise
        # wall_low = F.softplus((self.r_floor + self.floor_margin) - r)
        # barrier = self.lam_wall * wall_low.pow(2)

        # No x-regularization; add mild quadratic prior on r only
        prior = 0.5 * self.lam_r * (r**2)
        # alpha = self.alpha_const
        return self.alpha_const * core + prior# + barrier


# ------------------------------
# Decoder network (NEW)
# ------------------------------

class DecoderMLP(nn.Module):
    """
    Simple MLP decoder: input z=(r, phi) (circular features [r, sin(phi), cos(phi)]) -> output x in R^2.
    """
    def __init__(self, z_dim=2, x_dim=2, hidden=128, depth=3):
        super().__init__()
        self.use_circular = True
        in_dim = 3 if self.use_circular else z_dim
        layers = []
        for d in range(depth):
            layers += [nn.Linear(in_dim if d==0 else hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, x_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        if self.use_circular:
            r, phi = z[:, 0], z[:, 1]
            z_feat = torch.stack([r, torch.sin(phi), torch.cos(phi)], dim=-1)
        else:
            z_feat = z
        return self.net(z_feat)


# ------------------------------
# Particle buffers
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
    """Persistent buffer for negative (x,z) particles; updated via short-run Langevin."""
    def __init__(self, cfg: JointBufferConfig, x_dim:int, z_dim:int, device=None, seed:int=0):
        self.cfg = cfg
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.device = device or default_device()
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

        self.x = beta_dist((cfg.size, x_dim), cfg.dist_alpha, cfg.dist_beta, cfg.x_min, cfg.x_max, self.device)
        r = beta_dist((cfg.size, 1), cfg.dist_alpha, cfg.dist_beta, cfg.r_min, cfg.r_max, self.device)
        phi = beta_dist((cfg.size, 1), cfg.dist_alpha, cfg.dist_beta, -math.pi, math.pi, self.device)
        self.z = torch.cat([r, phi], dim=-1)

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
        
            self.z[:, 0].clamp_(self.cfg.r_min, self.cfg.r_max)
            self.z[:, 1] = (self.z[:, 1] + math.pi) % (2 * math.pi) - math.pi

    @torch.no_grad()
    def part_reinit(self, mask:torch.Tensor=None):
        # Random reinit
        if mask is None:
            mask = torch.rand(self.x.shape[0], device=self.device, generator=self.gen) < self.cfg.reinit_prob

        if mask.any():
            num = int(mask.sum().item())
            self.x[mask] = beta_dist((num, self.x_dim), self.cfg.dist_alpha, self.cfg.dist_beta, self.cfg.x_min, self.cfg.x_max, self.device)
            r_new = beta_dist((num,), self.cfg.dist_alpha, self.cfg.dist_beta, self.cfg.r_min, self.cfg.r_max, self.device)
            phi_new = beta_dist((num,), self.cfg.dist_alpha, self.cfg.dist_beta, -math.pi, math.pi, self.device)
            self.z[mask, :] = torch.stack([r_new, phi_new], dim=-1)

    def sample(self, n:int) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(low=0, high=self.cfg.size, size=(n,), device=self.device, generator=self.gen)
        return self.x[idx], self.z[idx]

@dataclass
class LatentBankConfig:
    m_per_example: int = 4
    k_steps: int = 10
    step_size: float = 1e-2
    noise_scale: float = 1.0
    z_init_std: float = 1.0           # std for unconstrained init

    r_min: float = 0.2
    r_max: float = 2.0
    reinit_prob: float = 0.10
    dist_alpha: float = 2.0
    dist_beta: float = 2.0    

class LatentParticleBank:
    """
    Stores M latent particles per example. 
    """
    def __init__(self, cfg: LatentBankConfig, n_examples:int, z_dim:int, device=None, seed:int=0):
        self.cfg = cfg
        self.n_examples = n_examples
        self.z_dim = z_dim
        self.device = device or default_device()
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        
        B, M = n_examples, cfg.m_per_example
        r = beta_dist((B, M), cfg.dist_alpha, cfg.dist_beta, cfg.r_min, cfg.r_max, self.device)
        phi = beta_dist((B, M), cfg.dist_alpha, cfg.dist_beta, -math.pi, math.pi, self.device)
        self.z = torch.stack([r, phi], dim=-1)  # [B, M, 2] -> (r, phi)        

    @torch.no_grad()
    def update_for_batch(self, energy: EnergyMLP, x_batch: torch.Tensor, idx_batch: torch.Tensor):
        M, K, eta = self.cfg.m_per_example, self.cfg.k_steps, self.cfg.step_size
        noise = math.sqrt(2.0 * eta) * self.cfg.noise_scale

        z_part = self.z[idx_batch]  # [B, M, 2] (r, phi)
        B = x_batch.shape[0]
        x_rep = x_batch[:, None, :].expand(B, M, x_batch.shape[-1]).reshape(B * M, -1)
        z_flat = z_part.reshape(B * M, 2).contiguous()

        for _ in range(K):
            z_flat = z_flat.detach().requires_grad_(True)
            with torch.enable_grad():
                e = energy(x_rep, z_flat).sum()  # energy expects z as (r, phi)
                (grad_z,) = torch.autograd.grad(e, (z_flat,), create_graph=False)

            z_flat = (z_flat - eta * grad_z + noise * torch.randn(z_flat.shape, device=z_flat.device, dtype=z_flat.dtype, generator=self.gen)).detach()

            # Constrain r, wrap phi
            z_flat[:, 0].clamp_(self.cfg.r_min, self.cfg.r_max)
            z_flat[:, 1] = (z_flat[:, 1] + math.pi) % (2 * math.pi) - math.pi

        self.z[idx_batch] = z_flat.view(B, M, -1)

    @torch.no_grad()
    def part_reinit(self, mask:torch.Tensor=None):
        B, M = self.z.shape[0], self.z.shape[1]
        z_flat = self.z.reshape(B*M, -1)

        # Random reinit
        if mask is None:
            mask = torch.rand(z_flat.shape[0], device=self.device, generator=self.gen) < self.cfg.reinit_prob
        # Assign reinit idx
        else:
            mask = mask.to(self.device)

        if mask.any():
            num = int(mask.sum().item())
            r_new = beta_dist((num,), self.cfg.dist_alpha, self.cfg.dist_beta, self.cfg.r_min, self.cfg.r_max, self.device)
            phi_new = beta_dist((num,), self.cfg.dist_alpha, self.cfg.dist_beta, -math.pi, math.pi, self.device)
            z_flat[mask, :] = torch.stack([r_new, phi_new], dim=-1)
        
        self.z = z_flat.reshape(B, M, -1)

    def sample_for_batch(self, idx_batch: torch.Tensor) -> torch.Tensor:
        """Return transformed latents [r, phi] for the batch indices."""
        return self.z[idx_batch]


# ------------------------------
# Trainer
# ------------------------------

@dataclass
class TrainerConfig:
    x_dim: int = 2
    z_dim: int = 2
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
    # NEW: define output_dir (needed because it's used by training loop and set in __main__)
    output_dir: str = "."

class LVEBMTrainer:
    def __init__(
        self,
        energy: EnergyMLP,
        dataset: Dataset,
        cfg: TrainerConfig = TrainerConfig(),
        # NEW: allow externally provided split so we can prepare it *before* any training starts
        train_idx: Optional[torch.Tensor] = None,
        test_idx: Optional[torch.Tensor] = None,
    ):
        self.energy = energy
        self.cfg = cfg
        self.device = cfg.device or default_device()
        self.energy.to(self.device)
        set_seed(cfg.seed, deterministic=True)

        # ---------- Train/Test split (80/20) ----------
        # If user provided the split, use it; otherwise fall back to original internal split.
        if (train_idx is not None) and (test_idx is not None):
            # Ensure CPU tensors for samplers
            self.train_idx = train_idx.clone().cpu()
            self.test_idx  = test_idx.clone().cpu()
        else:
            gsplit = torch.Generator().manual_seed(cfg.seed + 2024)
            n = len(dataset)
            perm = torch.randperm(n, generator=gsplit)
            n_test = max(1, int(0.2 * n))
            self.test_idx = perm[:n_test].cpu()
            self.train_idx = perm[n_test:].cpu()

        # Samplers that pass global indices to Dataset.__getitem__
        self.train_sampler = SubsetRandomSampler(self.train_idx.tolist(), generator=torch.Generator().manual_seed(cfg.seed + 555))
        self.test_sampler  = SubsetRandomSampler(self.test_idx.tolist(),  generator=torch.Generator().manual_seed(cfg.seed + 556))

        self.dataset = dataset
        self.train_loader = DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=True,
            sampler=self.train_sampler, num_workers=0
        )
        self.test_loader = DataLoader(
            dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
            sampler=self.test_sampler, num_workers=0
        )

        # Banks (unchanged)
        self.joint = JointParticleBuffer(cfg.joint, x_dim=cfg.x_dim, z_dim=cfg.z_dim, device=self.device, seed=cfg.seed+1)
        self.bank  = LatentParticleBank(cfg.bank, n_examples=len(dataset), z_dim=cfg.z_dim, device=self.device, seed=cfg.seed+2)

        # Optimizer & cosine LR with warmup (unchanged)
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
        Given a batch of x, initialize (r,phi) via independent Beta mappings and run K-step ULA
        in latent space. Return z_post with shape [B, M, 2]. Does NOT modify internal buffers.
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
        phi = beta_dist((B, M), self.bank.cfg.dist_alpha, self.bank.cfg.dist_beta, -math.pi, math.pi, device)
        z = torch.stack([r, phi], dim=-1).reshape(B * M, 2)  # [B*M, 2]

        x_rep = x_batch[:, None, :].expand(B, M, x_batch.shape[-1]).reshape(B * M, -1)
        noise = math.sqrt(2.0 * eta) * noise_scale

        for _ in range(K):
            z = z.detach().requires_grad_(True)
            with torch.enable_grad():
                e = self.energy(x_rep, z).sum()
                (grad_z,) = torch.autograd.grad(e, (z,), create_graph=False)

            z = (z - eta * grad_z + noise * torch.randn(z.shape, device=device, dtype=z.dtype, generator=g)).detach()

            z[:, 0].clamp_(r_min, r_max)                                     # r
            z[:, 1] = (z[:, 1] + math.pi) % (2 * math.pi) - math.pi          # phi

        return z.view(B, M, 2)

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

        # Construct mismatched negatives (kept same as your original logic)
        z_neg_eval = z_pos_eval.detach().clone()
        g_dev = torch.Generator(device=self.device).manual_seed(self.cfg.seed + 12346)
        r   = z_pos_eval[:, 0]
        phi = z_pos_eval[:, 1]
        alpha = 0.35; k = 2
        scale = 1.0 + alpha * torch.cos(k * phi)
        x1 = r * scale * torch.cos(phi)
        x2 = r * scale * torch.sin(phi)
        x_neg_eval = torch.stack([x1, x2], dim=-1)
        x_neg_eval = x_neg_eval + torch.normal(0.0, 0.2, size=x_neg_eval.shape, generator=g_dev, device=self.device)

        self.eval_set = (x_pos_eval, z_pos_eval, x_neg_eval, z_neg_eval)

    # ------------------------------
    # Core energy training loop (UNCHANGED except using self.train_loader)
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
            z_pos = self.bank.sample_for_batch(idx)                          # [B, M, 2] (r, phi)
            x_pos = x[:, None, :].expand(x.shape[0], M, x.shape[-1])         # [B, M, 2]

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
                z_soft = (w.unsqueeze(-1) * z_pos).sum(dim=1)         # [B,2]
                B = x.shape[0]
                X_rep = x[:,None,:].expand(B,B,2).reshape(-1,2)
                Z_rep = z_soft[None,:,:].expand(B,B,2).reshape(-1,2)
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
                z_bank_b = self.bank.z[idx]              # [B, M, 2]
                r_bank_b = z_bank_b[..., 0]              # [B, M]
                low_b = torch.full_like(r_bank_b, low_val).to(self.device)
                high_b = torch.full_like(r_bank_b, high_val).to(self.device)
                bad_bank_b = ((r_bank_b <= low_b) | (r_bank_b >= high_b))   # [B, M] bool

                if bad_bank_b.any():
                    # Build a global mask over [n_examples, M], but only mark current batch positions
                    N_examples = self.bank.z.shape[0]
                    M_bank = self.bank.z.shape[1]
                    mask_global = torch.zeros((N_examples, M_bank),
                                            dtype=torch.bool, device=self.device)
                    mask_global[idx] = bad_bank_b
                    mask_flat = mask_global.reshape(-1)   # [N_examples*M]
                    self.bank.part_reinit(mask_flat)                    

            if step % (0.5*self.cfg.log_interval) == 0:
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
                
                if step % 100 == 0:
                    model_save_path = os.path.join(self.cfg.output_dir, f"energy_model_step_{step}.pt")
                    torch.save(self.energy.state_dict(), model_save_path)
                    print(f"[INFO] Model saved to {model_save_path}")

                print_str += f"  | eval_gap={obj_eval: .4f}  rank_acc={rank_acc: .3f}  "
                print(print_str)
        print("Training finished.")


@torch.no_grad()
def visualize_distributions(trainer: "LVEBMTrainer", step:int=0, output_dir: str = "."):
    """
    Visualization now *consistently* uses the test split only (indices provided by trainer.test_idx)
    so that evaluation never touches the training portion.
    """
    device = trainer.device
    ds = trainer.dataset

    # --- Use only the test split for visualization ---
    test_idx = trainer.test_idx
    X = ds.x[test_idx].to(device)
    Z0 = ds.z[test_idx].to(device)
    r0 = Z0[:, 0].detach().cpu().numpy()

    # Collect posterior particles on the *test split only*
    Z_particles = collect_posterior_particles(
        trainer.energy, ds, trainer.cfg.bank, device=device,
        seed=trainer.cfg.seed + 777, indices=test_idx
    )  # [N_test, M, 2]

    z_mean = aggregate_from_particles(X, Z_particles, trainer.energy, how="mean")
    z_map  = aggregate_from_particles(X, Z_particles, trainer.energy, how="map")

    x_hat_mean = reconstruct_x(z_mean[:, 0], z_mean[:, 1])  # [N_test,2]
    x_hat_map  = reconstruct_x(z_map[:, 0],  z_map[:, 1])  

    plot_xy(f"{step} x̂ from z' (mean)", x_hat_mean, output_dir=output_dir)
    plot_xy(f"{step} x̂ from z' (MAP)",  x_hat_map, output_dir=output_dir)

@torch.no_grad()
def reconstruct_x(r: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Given r, phi (r in R+, phi in radians), return x=[r cos phi, r sin phi]."""
    phi = wrap_0_2pi(phi)
    x0 = r * torch.cos(phi)
    x1 = r * torch.sin(phi)
    return torch.stack([x0, x1], dim=-1)

def plot_xy(title: str, X: torch.Tensor, max_points: int = 10000, output_dir : str = "."):
    """Scatter plot for 2D points (tensor of shape [N,2])."""
    xs = X.detach().cpu()
    if xs.shape[0] > max_points:
        idx = torch.randperm(xs.shape[0])[:max_points]
        xs = xs[idx]
    plt.figure(figsize=(4.2, 4.2))
    plt.scatter(xs[:, 0].numpy(), xs[:, 1].numpy(), s=3, alpha=0.6)
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    plt.title(title)
    plt.xlabel('x[0]'); plt.ylabel('x[1]')
    plt.grid(True, ls=':', alpha=0.3)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{title}.png"))
    plt.close()

def wrap_0_2pi(phi: torch.Tensor) -> torch.Tensor:
    return (phi % (2 * math.pi))

@torch.no_grad()
def aggregate_from_particles(x: torch.Tensor, z_M: torch.Tensor, energy: EnergyMLP, how: str = "map"):
    N, M, _ = z_M.shape
    if how == "map":
        x_rep = x[:, None, :].expand(N, M, x.shape[-1]).reshape(-1, x.shape[-1])
        z_flat = z_M.reshape(-1, 2)
        e = energy(x_rep, z_flat).view(N, M)
        idx_min = e.argmin(dim=1)
        z_hat = z_M[torch.arange(N, device=z_M.device), idx_min, :]
        z_hat = torch.stack([z_hat[:, 0], wrap_0_2pi(z_hat[:, 1])], dim=-1)
        return z_hat
    elif how == "mean":
        r_hat = z_M[:, :, 0].mean(dim=1)
        phi_hat = circular_mean(z_M[:, :, 1], dim=1)
        return torch.stack([r_hat, phi_hat], dim=-1)
    else:
        raise ValueError("how must be 'map' or 'mean'")

def circular_mean(phi_M: torch.Tensor, dim: int = 1) -> torch.Tensor:
    s = torch.sin(phi_M).mean(dim=dim)
    c = torch.cos(phi_M).mean(dim=dim)
    return torch.atan2(s, c) % (2 * math.pi)

@torch.no_grad()
def collect_posterior_particles(
    energy: EnergyMLP,
    dataset: Dataset,
    bank_cfg: LatentBankConfig,
    device=None,
    seed: int = 123,
    # NEW: restrict to a given subset (e.g., test split) so evaluation does not touch training data
    indices: Optional[torch.Tensor] = None,
):
    """
    Run conditional Langevin over a chosen subset of the dataset (default: whole dataset)
    and return transformed particles with shape [N_subset, M, 2] *in the same order as indices*.
    """
    device = device or default_device()
    energy.eval()

    # We keep a bank sized to the full dataset for index compatibility, but we will
    # only *update* and *return* entries for `indices` (or all indices if None).
    full_N = len(dataset)
    bank = LatentParticleBank(bank_cfg, n_examples=full_N, z_dim=2, device=device, seed=seed)

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
        return bank.sample_for_batch(all_idx)  # [N_all, M, 2]
    else:
        return bank.sample_for_batch(indices.to(device))  # [N_subset, M, 2]

# ------------------------------
# Decoder training helpers (NEW)
# ------------------------------

@torch.no_grad()
def map_z_for_batch(energy: EnergyMLP, trainer: LVEBMTrainer, x_batch: torch.Tensor) -> torch.Tensor:
    """
    For each x in the batch, sample M candidates via trainer._posterior_probe(x),
    then pick the MAP z (argmin energy(x, z)).
    Returns: z_map [B, 2]
    """
    energy.eval()
    z_samps = trainer._posterior_probe(x_batch)  # [B, M, 2]
    B, M, _ = z_samps.shape
    x_rep = x_batch[:, None, :].expand(B, M, x_batch.shape[-1]).reshape(B * M, -1)
    z_flat = z_samps.reshape(B * M, -1)
    E = energy(x_rep, z_flat).view(B, M)  # [B, M]
    idx = torch.argmin(E, dim=1)          # [B]
    z_map = z_samps[torch.arange(B, device=x_batch.device), idx]
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
    Finally, reconstruct test set and save GT vs reconstruction plot.
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
                z_map = map_z_for_batch(energy, trainer, x)  # [B, 2]

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
    r = np.sqrt((X_gt ** 2).sum(axis=1))      # 注意 axis=1
    X_flag = np.empty(X_gt.shape[0], dtype=np.int64)
    X_flag[r <= 0.75] = 1
    mask2 = (r > 0.75) & (r <= 1.25)
    X_flag[mask2] = 2
    X_flag[r > 1.25] = 3

    os.makedirs(os.path.dirname(data_save_path) or ".", exist_ok=True)

    np.savez(
        data_save_path,   # 建议用 .npz 结尾
        X_gt=X_gt,
        X_rec=X_rec,
        X_flag=X_flag
    )
    print(f"[Data] Saved numpy arrays to: {data_save_path}")

    plt.figure(figsize=(6,6))
    plt.scatter(X_gt[:,0],  X_gt[:,1],  s=6, alpha=0.5, label="GT (test)")
    plt.scatter(X_rec[:,0], X_rec[:,1], s=6, alpha=0.5, label="Reconstruction")
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_plot_path) or ".", exist_ok=True)
    plt.savefig(save_plot_path, dpi=150)
    print(f"[Decoder] Saved reconstruction plot to: {save_plot_path}")


# ------------------------------
# Visualization & posterior extraction + Main
# ------------------------------

if __name__ == "__main__":
    device = default_device()
    print("Using device:", device)


    ds = LatentControlledRings(n=50000, return_z=False, seed=77)

    # NEW: Perform an explicit 80/20 split *before* any training starts (trainer or decoder).
    # This split will be reused by LVEBMTrainer and the decoder stage to ensure consistent usage.
    gsplit = torch.Generator().manual_seed(77)
    n_total = len(ds)
    perm = torch.randperm(n_total, generator=gsplit)
    n_test = max(1, int(0.2 * n_total))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    r_min_list = [0.0]
    r_max_list = [10.0]
    k_steps_list = [25]
    step_size_list = [8e-4]
    noise_scale_list = [1.0]
    seed_list = [277]
    lr_list = [2e-3]

    # r_min_list = [0.0]
    # r_max_list = [10.0]
    # k_steps_list = [20, 25, 30]
    # step_size_list = [1e-3, 2e-3, 8e-4]
    # noise_scale_list = [1.0]
    # seed_list = [77, 177, 277]
    # lr_list = [2e-3, 3e-3]

    # param_grid = list(itertools.product(r_min_list, r_max_list, k_steps_list, step_size_list, noise_scale_list, seed_list, lr_list))
    param_grid = [
        (0.0, 10, 25, 8e-4, 1.0, 77, 3e-3),
        (0.0, 10, 25, 8e-4, 1.0, 277, 2e-3),
        (0.0, 10, 30, 1e-3, 1.0, 277, 3e-3)
    ]

    for i, (r_min, r_max, k_steps, step_size, noise_scale, base_seed, lr) in enumerate(param_grid):
        print(f"\n=== Grid {i+1}/{len(param_grid)}: r_min={r_min}, k_steps={k_steps}, step_size={step_size}, noise_scale={noise_scale}, seed={base_seed}, lr={lr} ===")

        output_dir = f"loss_pic_2d/grid_{i+1}_rmin{r_min}_k{k_steps}_step{step_size}_noise{noise_scale}_seed{base_seed}_lr{lr}/"
        # if os.path.exists(output_dir):
        #     print(f"Output directory {output_dir} already exists. Skipping this grid.")
        #     continue

        set_seed(base_seed, deterministic=True)
        energy = EnergyMLP(x_dim=2, z_dim=2, hidden=128, depth=3, lam_x=0.0, lam_r=0.0)
        
        loaded_step = None
        # if len(sys.argv) > 1:
        #     loaded_step = sys.argv[1]
        #     energy.load_state_dict(torch.load(f"2d_best_res/grid_1_rmin0.0_k25_step0.0008_noise1.0_seed277_lr0.002/energy_model_step_{loaded_step}.pt"))        

        cfg = TrainerConfig(
            x_dim=2, z_dim=2,
            steps=4000,
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
            output_dir=f"loss_pic_2d/grid_{i+1}_rmin{r_min}_k{k_steps}_step{step_size}_noise{noise_scale}_seed{base_seed}_lr{lr}/"
        )

        # Pass the precomputed split to the trainer to guarantee consistent usage across all stages.
        trainer = LVEBMTrainer(energy=energy, dataset=ds, cfg=cfg, train_idx=train_idx, test_idx=test_idx)
        trainer.train()

        # -------- After energy training: freeze energy and train decoder --------
        energy.eval()
        for p in energy.parameters():
            p.requires_grad_(False)

        decoder = DecoderMLP(z_dim=2, x_dim=2, hidden=128, depth=3)

        train_decoder(
            trainer=trainer,
            energy=energy,
            decoder=decoder,
            epochs=15,          # can be adjusted
            lr=1e-3,
            weight_decay=0.0,
            patience=20,         # early stopping patience
            log_every=1,
            save_plot_path=f"loss_pic_2d/grid_{i+1}_rmin{r_min}_k{k_steps}_step{step_size}_noise{noise_scale}_seed{base_seed}_lr{lr}/recon_vs_gt.png" if loaded_step is None else f"2d_best_res/grid_1_rmin0.0_k25_step0.0008_noise1.0_seed277_lr0.002/{loaded_step}_recon_vs_gt.png",
            data_save_path=f"loss_pic_2d/grid_{i+1}_rmin{r_min}_k{k_steps}_step{step_size}_noise{noise_scale}_seed{base_seed}_lr{lr}/recon_data.png" if loaded_step is None else f"2d_best_res/grid_1_rmin0.0_k25_step0.0008_noise1.0_seed277_lr0.002/{loaded_step}_recon_data.pt"
        )
