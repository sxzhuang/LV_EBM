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

# ==============================
# Utils
# ==============================

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
    samples = beta_d.sample(gen_shape).to(device)
    gen_data = min_bound + (max_bound - min_bound) * samples
    return gen_data


# ==============================
# Dataset: LCFR-2D
# ==============================

class LatentControlledFourierRings2D(Dataset):
    """
    2D Latent Controlled Fourier Rings (LCFR-2D)
    Variant: each ring center Rc shares a fixed (a, b) coefficient set.

    Radius modulation by Fourier series:
        r_base(phi) = Rc + sum_{k=1..K} [ a_k*cos(k*phi) + b_k*sin(k*phi) ]
        r ~ N(r_base(phi), sigma_r^2)

    Observation:
        x = [ r*cos(phi), r*sin(phi) ] + N(0, sigma_x^2 I)

    Latent z structure (THIS VERSION):
        z = [ Rc, phi, a_1..a_K, b_1..b_K ]
        dim(z) = 1 + 1 + K + K = 2K + 2

    Key properties:
      - (a, b) are shared per center Rc: each Rc has one (a, b) reused by all samples from that center.
      - Center-wise coefficients are stored in self.coeffs_a/self.coeffs_b with shape [n_centers, K].
      - r and center_id are *not* part of z. r is sampled from the distribution implied by z; center_id is an auxiliary index.
    """
    def __init__(
        self,
        n: int = 50_000,
        centers=(0.5, 1.0, 1.5),
        K: int = 5,
        sigma_coeff: float = 0.06,
        coeffs_override_a=None, coeffs_override_b=None,
        sigma_r: float = 0.02,
        sigma_x: float = 0.02,
        r_min: float = 1e-3,
        r_max: float = float("inf"),
        return_z: bool = False,
        seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device=None,
    ):
        super().__init__()
        self.K = int(K)
        self.return_z = return_z
        self.r_min = r_min
        self.r_max = r_max
        self.dtype = dtype

        g = torch.Generator(device=device).manual_seed(seed)

        # Centers and count
        centers = torch.as_tensor(list(centers), dtype=dtype, device=device)
        n_centers = centers.numel()
        self.centers = centers

        # Sample center-wise coefficients
        if self.K > 0:
            self.coeffs_a = torch.zeros(n_centers, self.K, device=device, dtype=dtype)
            self.coeffs_b = torch.zeros(n_centers, self.K, device=device, dtype=dtype)
            if sigma_coeff > 0:
                self.coeffs_a += torch.normal(
                    0, sigma_coeff, size=self.coeffs_a.shape, generator=g, device=device, dtype=dtype
                )
                self.coeffs_b += torch.normal(
                    0, sigma_coeff, size=self.coeffs_b.shape, generator=g, device=device, dtype=dtype
                )
        else:
            # Keep empty last dimension when K=0 for safe concatenations
            self.coeffs_a = torch.zeros((n_centers, 0), device=device, dtype=dtype)
            self.coeffs_b = torch.zeros((n_centers, 0), device=device, dtype=dtype)

        # Optional overrides: keys are center indices; values are {k: value}, with k starting at 1
        if coeffs_override_a:
            for ci, mapping in coeffs_override_a.items():
                for k, val in mapping.items():
                    self.coeffs_a[ci, k - 1] = float(val)
        if coeffs_override_b:
            for ci, mapping in coeffs_override_b.items():
                for k, val in mapping.items():
                    self.coeffs_b[ci, k - 1] = float(val)

        # Assign a center to each sample and gather its Rc and (a, b)
        idx = torch.randint(low=0, high=n_centers, size=(n,), generator=g, device=device)  # [n]
        Rc = centers[idx]              # [n]
        a = self.coeffs_a[idx]         # [n, K]
        b = self.coeffs_b[idx]         # [n, K]

        # Angle phi
        phi = torch.rand(n, generator=g, device=device, dtype=dtype) * (2.0 * math.pi)  # [n]

        # Compute r_base(phi)
        if self.K > 0:
            ks = torch.arange(1, self.K + 1, device=device, dtype=dtype).view(1, -1)   # [1, K]
            kphi = ks * phi.view(-1, 1)                                                 # [n, K]
            fourier_sum = (a * torch.cos(kphi) + b * torch.sin(kphi)).sum(dim=-1)       # [n]
            r_base = Rc + fourier_sum                                                   # [n]
        else:
            r_base = Rc

        # Sample r from N(r_base, sigma_r^2)
        r = torch.normal(mean=r_base, std=torch.full_like(r_base, sigma_r), generator=g)

        # Clip r to valid range
        if math.isfinite(self.r_max):
            r = r.clamp_min(self.r_min).clamp_max(self.r_max)
        else:
            r = r.clamp_min(self.r_min)

        # Build observation x with isotropic noise
        x = torch.stack([r * torch.cos(phi), r * torch.sin(phi)], dim=-1)               # [n, 2]
        x = x + torch.normal(mean=0.0, std=sigma_x, size=x.shape, generator=g, device=device, dtype=dtype)

        # ===== Save fields =====
        self.x = x
        self.r = r               # sampled radius (not part of z)
        self.Rc = Rc             # per-sample center radius (part of z)
        self.center_id = idx     # auxiliary: center index (not part of z)

        # z = [Rc, phi, a_1..a_K, b_1..b_K]
        z_parts = [Rc.view(-1, 1), phi.view(-1, 1), a, b]
        self.z = torch.cat(z_parts, dim=-1)

        # Convenient slices to access parts of z
        self.slices = {
            "Rc": slice(0, 1),
            "phi": slice(1, 2),
            "a": slice(2, 2 + self.K),
            "b": slice(2 + self.K, 2 + 2 * self.K),
        }

        # Metadata
        self.meta = {
            "z_names": (
                ["Rc", "phi"]
                + [f"a_{k}" for k in range(1, self.K + 1)]
                + [f"b_{k}" for k in range(1, self.K + 1)]
            ),
            "z_dim": 2 * self.K + 2,
            "K": self.K,
            "sigma_coeff": sigma_coeff,
            "sigma_r": sigma_r,
            "sigma_x": sigma_x,
            "centers": centers.detach().cpu().tolist(),
            "coeffs_a": self.coeffs_a.detach().cpu().numpy(),  # fixed a_k per center
            "coeffs_b": self.coeffs_b.detach().cpu().numpy(),  # fixed b_k per center
            "aux_names": ["center_id"],                        # auxiliary fields (not in z)
        }

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        if self.return_z:
            return self.x[i], self.z[i], i
        return self.x[i], i

    @staticmethod
    def fourier_radius(phi: torch.Tensor, Rc: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        Given batched phi, Rc, and (a, b) with shape [n, K], return r_base(phi).
        """
        if a.numel() == 0 and b.numel() == 0:
            return Rc
        K = a.shape[-1]
        device, dtype = phi.device, phi.dtype
        ks = torch.arange(1, K + 1, device=device, dtype=dtype).view(1, -1)   # [1, K]
        kphi = ks * phi.view(-1, 1)                                           # [n, K]
        return Rc + (a * torch.cos(kphi) + b * torch.sin(kphi)).sum(dim=-1)


# ==============================
# Energy network (adapted to LCFR-2D z)
# ==============================

class EnergyMLP(nn.Module):
    """
    Energy model over joint (x, z) where z = [Rc, phi, a_1..a_K, b_1..b_K].
    Uses circular embedding for phi and concatenates Rc, sin(phi), cos(phi), a, b.
    Adds mild L2 priors on Rc and coefficients (a, b).
    """
    def __init__(self, x_dim: int, z_dim: int, K: int, hidden=128, depth=3,
                 lam_x: float = 0.0, lam_coeff: float = 1e-3, lam_rc: float = 1e-4):
        super().__init__()
        self.use_circular = True
        self.K = int(K)
        # z features: Rc (1), sin(phi) (1), cos(phi) (1), a (K), b (K) => 2K + 3
        z_feat_dim = 3 + 2 * self.K
        in_dim = x_dim + z_feat_dim

        layers = []
        for d in range(depth):
            layers += [nn.Linear(in_dim if d==0 else hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

        self.lam_x = lam_x
        self.lam_coeff = lam_coeff
        self.lam_rc = lam_rc
        self.register_buffer("alpha_const", torch.tensor(1.0))

    def forward(self, x, z):
        # Parse z
        Rc = z[:, 0]
        phi = z[:, 1]
        a = z[:, 2:2 + self.K]
        b = z[:, 2 + self.K: 2 + 2 * self.K]

        if self.use_circular:
            z_feat = torch.cat([
                Rc.unsqueeze(-1),
                torch.sin(phi).unsqueeze(-1),
                torch.cos(phi).unsqueeze(-1),
                a, b
            ], dim=-1)
        else:
            z_feat = z

        core = self.net(torch.cat([x, z_feat], dim=-1)).squeeze(-1)

        # Mild Gaussian priors: Rc ~ N(0, lam_rc^{-1}), a,b ~ N(0, lam_coeff^{-1})
        prior = 0.5 * self.lam_rc * (Rc ** 2)
        if a.numel() > 0:
            prior = prior + 0.5 * self.lam_coeff * (a.pow(2).sum(dim=-1) + b.pow(2).sum(dim=-1))

        return self.alpha_const * core + prior


# ==============================
# Decoder network (adapted to new z)
# ==============================

class DecoderMLP(nn.Module):
    """
    Simple MLP decoder: input z=[Rc, phi, a..., b...] (with circular features for phi)
    -> output x in R^2.
    """
    def __init__(self, K: int, z_dim: int, x_dim: int = 2, hidden: int = 128, depth: int = 3):
        super().__init__()
        self.use_circular = True
        self.K = int(K)
        in_dim = 3 + 2 * self.K if self.use_circular else z_dim  # Rc + sin + cos + a + b
        layers = []
        for d in range(depth):
            layers += [nn.Linear(in_dim if d==0 else hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, x_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        Rc = z[:, 0]
        phi = z[:, 1]
        a = z[:, 2:2 + self.K]
        b = z[:, 2 + self.K: 2 + 2 * self.K]

        if self.use_circular:
            z_feat = torch.cat([
                Rc.unsqueeze(-1),
                torch.sin(phi).unsqueeze(-1),
                torch.cos(phi).unsqueeze(-1),
                a, b
            ], dim=-1)
        else:
            z_feat = z
        return self.net(z_feat)


# ==============================
# Particle buffers (adapted to LCFR-2D z)
# ==============================

@dataclass
class JointBufferConfig:
    size: int = 8192
    k_steps: int = 40
    step_size_x: float = 1e-2
    step_size_z: float = 1e-2
    noise_scale: float = 1.0
    reinit_prob: float = 0.05
    x_init_std: float = 1.0

    # Interpreted for LCFR-2D as bounds for Rc (formerly r_min/max)
    r_min: float = 0.2
    r_max: float = 2.5

    # New for LCFR-2D
    K: int = 5
    coeff_clip: float = 0.6          # clamp range for a_k/b_k to [-coeff_clip, coeff_clip]
    coeff_std_init: float = 0.1       # init std for a_k/b_k
    centers: Optional[Tuple[float, ...]] = None  # if provided, Rc samples from this discrete set

    dist_alpha: float = 2.0
    dist_beta: float = 2.0
    boundary_margin: float = 0.05

class JointParticleBuffer:
    """Persistent buffer for negative (x,z) particles; updated via short-run Langevin."""
    def __init__(self, cfg: JointBufferConfig, x_dim:int, z_dim:int, device=None, seed:int=0):
        self.cfg = cfg
        self.x_dim = x_dim
        self.z_dim = z_dim
        self.K = int(cfg.K)
        self.device = device or default_device()
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

        # Initialize x from a broad Beta box
        self.x = beta_dist((cfg.size, x_dim), cfg.dist_alpha, cfg.dist_beta, -2.0, 2.0, self.device)

        # Initialize z = [Rc, phi, a..., b...]
        Rc = self._sample_Rc(cfg.size)
        phi = beta_dist((cfg.size, 1), cfg.dist_alpha, cfg.dist_beta, -math.pi, math.pi, self.device)
        a = torch.randn(cfg.size, self.K, device=self.device, generator=self.gen) * cfg.coeff_std_init
        b = torch.randn(cfg.size, self.K, device=self.device, generator=self.gen) * cfg.coeff_std_init
        self.z = torch.cat([Rc, phi, a, b], dim=-1)

    def _sample_Rc(self, n:int):
        if self.cfg.centers is not None and len(self.cfg.centers) > 0:
            centers = torch.tensor(self.cfg.centers, device=self.device, dtype=torch.float32)
            idx = torch.randint(low=0, high=len(centers), size=(n,1), device=self.device, generator=self.gen)
            return centers[idx]
        # Fallback to continuous Beta over [r_min, r_max] as Rc range
        Rc = beta_dist((n, 1), self.cfg.dist_alpha, self.cfg.dist_beta, self.cfg.r_min, self.cfg.r_max, self.device)
        return Rc

    @torch.no_grad()
    def refresh(self, energy: "EnergyMLP"):
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

            # Constrain z
            self._project_inplace()

    def _project_inplace(self):
        # Rc bounds
        Rc = self.z[:, 0]
        # Rc.clamp_(self.cfg.r_min, self.cfg.r_max)
        # Wrap phi to (-pi, pi]
        self.z[:, 1] = (self.z[:, 1] + math.pi) % (2 * math.pi) - math.pi
        # Clamp coefficients
        if self.K > 0:
            a = self.z[:, 2:2 + self.K]
            b = self.z[:, 2 + self.K: 2 + 2 * self.K]
            # a.clamp_(-self.cfg.coeff_clip, self.cfg.coeff_clip)
            # b.clamp_(-self.cfg.coeff_clip, self.cfg.coeff_clip)
            self.z[:, 2:2 + self.K] = a
            self.z[:, 2 + self.K: 2 + 2 * self.K] = b

    @torch.no_grad()
    def part_reinit(self, mask:torch.Tensor=None):
        # Random reinit
        if mask is None:
            mask = torch.rand(self.x.shape[0], device=self.device, generator=self.gen) < self.cfg.reinit_prob

        if mask.any():
            num = int(mask.sum().item())
            self.x[mask] = beta_dist((num, self.x_dim), self.cfg.dist_alpha, self.cfg.dist_beta, -2.0, 2.0, self.device)

            Rc_new = self._sample_Rc(num)
            phi_new = beta_dist((num, 1), self.cfg.dist_alpha, self.cfg.dist_beta, -math.pi, math.pi, self.device)
            a_new = torch.randn(num, self.K, device=self.device, generator=self.gen) * self.cfg.coeff_std_init
            b_new = torch.randn(num, self.K, device=self.device, generator=self.gen) * self.cfg.coeff_std_init
            self.z[mask, :] = torch.cat([Rc_new, phi_new, a_new, b_new], dim=-1)

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

    # Interpreted for LCFR-2D as bounds for Rc (formerly r_min/max)
    r_min: float = 0.2
    r_max: float = 2.5

    # New for LCFR-2D
    K: int = 5
    coeff_clip: float = 0.6
    coeff_std_init: float = 0.1
    centers: Optional[Tuple[float, ...]] = None

    reinit_prob: float = 0.10
    dist_alpha: float = 2.0
    dist_beta: float = 2.0

class LatentParticleBank:
    """
    Stores M latent particles per example for LCFR-2D: z = [Rc, phi, a..., b...].
    """
    def __init__(self, cfg: LatentBankConfig, n_examples:int, z_dim:int, device=None, seed:int=0):
        self.cfg = cfg
        self.n_examples = n_examples
        self.z_dim = z_dim
        self.K = int(cfg.K)
        self.device = device or default_device()
        self.gen = torch.Generator(device=self.device).manual_seed(seed)

        B, M = n_examples, cfg.m_per_example
        Rc = self._sample_Rc((B, M))
        phi = beta_dist((B, M), cfg.dist_alpha, cfg.dist_beta, -math.pi, math.pi, self.device)
        a = torch.randn(B, M, self.K, device=self.device, generator=self.gen) * cfg.coeff_std_init
        b = torch.randn(B, M, self.K, device=self.device, generator=self.gen) * cfg.coeff_std_init
        self.z = torch.cat([Rc.unsqueeze(-1), phi.unsqueeze(-1), a, b], dim=-1)  # [B, M, 2+2K]

    def _sample_Rc(self, shape, device=None):
        """
        Flexible Rc sampler.
        Accepts shape as:
          - int          -> returns tensor with shape [n]
          - (n,)         -> returns tensor with shape [n]
          - (B, M)       -> returns tensor with shape [B, M]
        If cfg.centers is provided, sample Rc from discrete centers; otherwise from Beta in [r_min, r_max].
        """
        device = device or self.device

        # Normalize shape
        if isinstance(shape, int):
            size = (shape,)                  # 1-D
        elif isinstance(shape, tuple):
            if len(shape) == 1:
                size = shape                 # 1-D
            elif len(shape) == 2:
                size = shape                 # 2-D
            else:
                raise ValueError(f"Unsupported shape length for Rc: {shape}")
        else:
            raise TypeError(f"Unsupported shape type for Rc: {type(shape)}")

        # Sample
        if self.cfg.centers is not None and len(self.cfg.centers) > 0:
            centers = torch.tensor(self.cfg.centers, device=device, dtype=torch.float32)
            idx = torch.randint(low=0, high=len(centers), size=size, device=device, generator=self.gen)
            Rc = centers[idx]                # shape == size
        else:
            Rc = beta_dist(size, self.cfg.dist_alpha, self.cfg.dist_beta, self.cfg.r_min, self.cfg.r_max, device)

        return Rc  # [n] or [B,M], matching `size`

    @torch.no_grad()
    def update_for_batch(self, energy: EnergyMLP, x_batch: torch.Tensor, idx_batch: torch.Tensor):
        M, Ksteps, eta = self.cfg.m_per_example, self.cfg.k_steps, self.cfg.step_size
        noise = math.sqrt(2.0 * eta) * self.cfg.noise_scale

        z_part = self.z[idx_batch]  # [B, M, z_dim]
        B = x_batch.shape[0]
        x_rep = x_batch[:, None, :].expand(B, M, x_batch.shape[-1]).reshape(B * M, -1)
        z_flat = z_part.reshape(B * M, -1).contiguous()

        for _ in range(Ksteps):
            z_flat = z_flat.detach().requires_grad_(True)
            with torch.enable_grad():
                e = energy(x_rep, z_flat).sum()  # energy expects full z
                (grad_z,) = torch.autograd.grad(e, (z_flat,), create_graph=False)

            z_flat = (z_flat - eta * grad_z + noise * torch.randn(z_flat.shape, device=z_flat.device, dtype=z_flat.dtype, generator=self.gen)).detach()

            # Constrain: Rc bounds, wrap phi, clamp coeffs
            # Rc
            # z_flat[:, 0].clamp_(self.cfg.r_min, self.cfg.r_max)
            # phi
            z_flat[:, 1] = (z_flat[:, 1] + math.pi) % (2 * math.pi) - math.pi
            # coeffs
            if self.K > 0:
                a = z_flat[:, 2:2 + self.K]
                b = z_flat[:, 2 + self.K: 2 + 2 * self.K]
                # a.clamp_(-self.cfg.coeff_clip, self.cfg.coeff_clip)
                # b.clamp_(-self.cfg.coeff_clip, self.cfg.coeff_clip)
                z_flat[:, 2:2 + self.K] = a
                z_flat[:, 2 + self.K: 2 + 2 * self.K] = b

        self.z[idx_batch] = z_flat.view(B, M, -1)

    @torch.no_grad()
    def part_reinit(self, mask: torch.Tensor = None):
        B, M = self.z.shape[0], self.z.shape[1]
        z_flat = self.z.reshape(B * M, -1)

        # Random reinit
        if mask is None:
            mask = torch.rand(z_flat.shape[0], device=self.device, generator=self.gen) < self.cfg.reinit_prob
        else:
            mask = mask.to(self.device)

        if mask.any():
            num = int(mask.sum().item())
            # Use the flexible sampler; it returns shape [num]
            Rc_new  = self._sample_Rc(num)                 # [num]
            phi_new = beta_dist((num,), self.cfg.dist_alpha, self.cfg.dist_beta, -math.pi, math.pi, self.device)  # [num]
            a_new   = torch.randn(num, self.K, device=self.device, generator=self.gen) * self.cfg.coeff_std_init  # [num,K]
            b_new   = torch.randn(num, self.K, device=self.device, generator=self.gen) * self.cfg.coeff_std_init  # [num,K]

            z_flat[mask, :] = torch.cat([Rc_new.view(-1, 1), phi_new.view(-1, 1), a_new, b_new], dim=-1)

        self.z = z_flat.reshape(B, M, -1)

    def sample_for_batch(self, idx_batch: torch.Tensor) -> torch.Tensor:
        """Return latent particles z for the batch indices."""
        return self.z[idx_batch]


# ==============================
# Trainer (minimal edits to support new z)
# ==============================

@dataclass
class TrainerConfig:
    x_dim: int = 2
    z_dim: int = 2  # will be overwritten to 2K+2 for LCFR-2D
    K: int = 5      # number of harmonics
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
        Given a batch of x, initialize z=[Rc,phi,a...,b...] via simple priors and run K-step ULA
        in latent space. Return z_post with shape [B, M, z_dim]. Does NOT modify internal buffers.
        """
        device = x_batch.device
        B = x_batch.shape[0]
        M = self.cfg.bank.m_per_example
        Ksteps = self.cfg.bank.k_steps
        eta = self.cfg.bank.step_size
        noise_scale = self.cfg.bank.noise_scale

        g = torch.Generator(device=device).manual_seed(self.cfg.seed + 33333)

        # Init z
        Rc = self._probe_sample_Rc(B, M, device)
        phi = beta_dist((B, M), self.bank.cfg.dist_alpha, self.bank.cfg.dist_beta, -math.pi, math.pi, device)
        a = torch.randn(B, M, self.cfg.K, device=device, generator=g) * self.bank.cfg.coeff_std_init
        b = torch.randn(B, M, self.cfg.K, device=device, generator=g) * self.bank.cfg.coeff_std_init
        z = torch.cat([Rc.unsqueeze(-1), phi.unsqueeze(-1), a, b], dim=-1).reshape(B * M, -1)

        x_rep = x_batch[:, None, :].expand(B, M, x_batch.shape[-1]).reshape(B * M, -1)
        noise = math.sqrt(2.0 * eta) * noise_scale

        for _ in range(Ksteps):
            z = z.detach().requires_grad_(True)
            with torch.enable_grad():
                e = self.energy(x_rep, z).sum()
                (grad_z,) = torch.autograd.grad(e, (z,), create_graph=False)

            z = (z - eta * grad_z + noise * torch.randn(z.shape, device=device, dtype=z.dtype, generator=g)).detach()

            # Constrain
            # z[:, 0].clamp_(self.bank.cfg.r_min, self.bank.cfg.r_max)              # Rc
            z[:, 1] = (z[:, 1] + math.pi) % (2 * math.pi) - math.pi               # phi
            if self.cfg.K > 0:
                a = z[:, 2:2 + self.cfg.K]; b = z[:, 2 + self.cfg.K: 2 + 2 * self.cfg.K]
                # a.clamp_(-self.bank.cfg.coeff_clip, self.bank.cfg.coeff_clip)
                # b.clamp_(-self.bank.cfg.coeff_clip, self.bank.cfg.coeff_clip)
                z[:, 2:2 + self.cfg.K] = a; z[:, 2 + self.cfg.K: 2 + 2 * self.cfg.K] = b

        return z.view(B, M, -1)

    def _probe_sample_Rc(self, B, M, device):
        if self.bank.cfg.centers is not None and len(self.bank.cfg.centers) > 0:
            centers = torch.tensor(self.bank.cfg.centers, device=device, dtype=torch.float32)
            idx = torch.randint(low=0, high=len(centers), size=(B, M), device=device, generator=torch.Generator(device=device).manual_seed(self.cfg.seed+1234))
            return centers[idx]
        return beta_dist((B, M), self.bank.cfg.dist_alpha, self.bank.cfg.dist_beta, self.bank.cfg.r_min, self.bank.cfg.r_max, device)

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

        # Construct mismatched negatives: distort mean radius r_base by a sinusoidal factor and add noise
        z_neg_eval = z_pos_eval.detach().clone()
        phi = z_pos_eval[:, 1]
        # Compute r_base from z_pos_eval
        a = z_pos_eval[:, 2:2 + self.cfg.K]
        b = z_pos_eval[:, 2 + self.cfg.K: 2 + 2 * self.cfg.K]
        Rc = z_pos_eval[:, 0]
        r_base = fourier_radius_from_parts(phi, Rc, a, b)  # [N]
        alpha = 0.35; k = 2
        scale = 1.0 + alpha * torch.cos(k * phi)
        r_neg = r_base * scale
        x1 = r_neg * torch.cos(phi)
        x2 = r_neg * torch.sin(phi)
        g_dev = torch.Generator(device=self.device).manual_seed(self.cfg.seed + 12346)
        x_neg_eval = torch.stack([x1, x2], dim=-1)
        x_neg_eval = x_neg_eval + torch.normal(0.0, 0.2, size=x_neg_eval.shape, generator=g_dev, device=self.device)

        self.eval_set = (x_pos_eval, z_pos_eval, x_neg_eval, z_neg_eval)

    # ------------------------------
    # Core energy training loop (kept intact conceptually, only adapted shapes)
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
            z_pos = self.bank.sample_for_batch(idx)                          # [B, M, z_dim]
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
                # InfoNCE (kept): soft posterior average z_soft (linear avg, note phi periodicity is ignored here as before)
                w = torch.softmax(-Epos / tau, dim=1)                 # [B,M]
                z_soft = (w.unsqueeze(-1) * z_pos).sum(dim=1)         # [B, z_dim]
                B = x.shape[0]
                X_rep = x[:,None,:].expand(B,B,2).reshape(-1,2)
                Z_rep = z_soft[None,:,:].expand(B,B,z_soft.shape[-1]).reshape(-1,z_soft.shape[-1])
                S = -self.energy(X_rep, Z_rep).view(B,B)              # [B,B]
                loss_nce = (torch.logsumexp(S, dim=1) - S.diag()).mean()
                loss = loss + self.cfg.lambda_nce * loss_nce

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.energy.parameters(), max_norm=5.0)
            self.opt.step()
            self.sched.step()

            logger.info(f"Step {step}: loss = {loss.item():.6f}")

            # Reinit out-of-bound particles for stability (now check Rc range)
            with torch.no_grad():
                low_val = self.joint.cfg.r_min * (1 + self.joint.cfg.boundary_margin)
                high_val = self.joint.cfg.r_max * (1 - self.joint.cfg.boundary_margin)

                # Joint buffer reinit
                z_batch = self.joint.z
                Rc = z_batch[:, 0]
                bad_examples = ((Rc <= low_val) | (Rc >= high_val))
                if bad_examples.any():
                    self.joint.part_reinit(bad_examples)

                # LatentBank reinit for current batch
                z_bank_b = self.bank.z[idx]              # [B, M, z_dim]
                Rc_bank_b = z_bank_b[..., 0]             # [B, M]
                low_b = torch.full_like(Rc_bank_b, low_val).to(self.device)
                high_b = torch.full_like(Rc_bank_b, high_val).to(self.device)
                bad_bank_b = ((Rc_bank_b <= low_b) | (Rc_bank_b >= high_b))   # [B, M] bool

                if bad_bank_b.any():
                    N_examples = self.bank.z.shape[0]
                    M_bank = self.bank.z.shape[1]
                    mask_global = torch.zeros((N_examples, M_bank),
                                              dtype=torch.bool, device=self.device)
                    mask_global[idx] = bad_bank_b
                    mask_flat = mask_global.reshape(-1)
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

                if step % 500 == 0:
                    model_save_path = os.path.join(self.cfg.output_dir, f"energy_model_step_{step}.pt")
                    torch.save(self.energy.state_dict(), model_save_path)
                    print(f"[INFO] Model saved to {model_save_path}")

                print_str += f"  | eval_gap={obj_eval: .4f}  rank_acc={rank_acc: .3f}  "
                print(print_str)
        print("Training finished.")


# ==============================
# Visualization & posterior extraction
# ==============================

@torch.no_grad()
def visualize_distributions(trainer: "LVEBMTrainer", step:int=0, output_dir: str = "."):
    """
    Visualization now uses the test split only (indices provided by trainer.test_idx).
    """
    device = trainer.device
    ds = trainer.dataset
    K = trainer.cfg.K

    # Use only the test split for visualization
    test_idx = trainer.test_idx
    X = ds.x[test_idx].to(device)
    Z0 = ds.z[test_idx].to(device)

    # Collect posterior particles on the test split only
    Z_particles = collect_posterior_particles(
        trainer.energy, ds, trainer.cfg.bank, device=device,
        seed=trainer.cfg.seed + 777, indices=test_idx
    )  # [N_test, M, z_dim]

    # z_mean = aggregate_from_particles(X, Z_particles, trainer.energy, how="mean")   # [N, z_dim]
    z_map  = aggregate_from_particles(X, Z_particles, trainer.energy, how="map")    # [N, z_dim]

    # x_hat_mean = reconstruct_x_from_z(z_mean, K=K)  # [N_test,2]
    x_hat_map  = reconstruct_x_from_z(z_map,  K=K)

    # plot_xy(f"{step} x̂ from z' (mean)", x_hat_mean, output_dir=output_dir)
    plot_xy(f"{step} x̂ from z' (MAP)",  x_hat_map,  output_dir=output_dir)


def fourier_radius_from_parts(phi: torch.Tensor, Rc: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute r_base(phi) given Rc, (a,b) and angle phi for LCFR-2D."""
    if a.numel() == 0 and b.numel() == 0:
        return Rc
    K = a.shape[-1]
    ks = torch.arange(1, K + 1, device=phi.device, dtype=phi.dtype).view(1, -1)   # [1,K]
    kphi = ks * phi.view(-1, 1)                                                   # [N,K]
    return Rc + (a * torch.cos(kphi) + b * torch.sin(kphi)).sum(dim=-1)

@torch.no_grad()
def reconstruct_x_from_z(z: torch.Tensor, K: Optional[int] = None) -> torch.Tensor:
    """
    Deterministic reconstruction from z via mean radius r_base (ignores sigma_r/x noise):
      x_hat = r_base(phi; Rc,a,b) * [cos(phi), sin(phi)].
    """
    if K is None:
        K = (z.shape[-1] - 2) // 2
    Rc  = z[:, 0]
    phi = z[:, 1]
    a   = z[:, 2:2 + K]
    b   = z[:, 2 + K: 2 + 2 * K]
    r_base = fourier_radius_from_parts(phi, Rc, a, b)
    x0 = r_base * torch.cos(phi)
    x1 = r_base * torch.sin(phi)
    return torch.stack([x0, x1], dim=-1)

def plot_xy(title: str, X: torch.Tensor, max_points: int = 10000, output_dir : str = "."):
    """Scatter plot for 2D points (tensor of shape [N,2])."""
    xs = X.detach().cpu()
    if xs.shape[0] > max_points:
        idx = torch.randperm(xs.shape[0])[:max_points]
        xs = xs[idx]
    plt.figure(figsize=(4.6, 4.6))
    plt.scatter(xs[:, 0].numpy(), xs[:, 1].numpy(), s=3, alpha=0.6)
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    plt.title(title)
    plt.xlabel('x[0]'); plt.ylabel('x[1]')
    plt.grid(True, ls=':', alpha=0.3)
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    # sanitize file name
    safe_title = title.replace(" ", "_").replace("/", "_")
    plt.savefig(os.path.join(output_dir, f"{safe_title}.png"))
    plt.close()

def wrap_0_2pi(phi: torch.Tensor) -> torch.Tensor:
    return (phi % (2 * math.pi))

@torch.no_grad()
def aggregate_from_particles(x: torch.Tensor, z_M: torch.Tensor, energy: EnergyMLP, how: str = "map"):
    """
    Aggregate multiple latent candidates per x either by MAP (argmin energy) or mean:
      - For "map": pick z with minimal energy.
      - For "mean": Rc and coefficients are arithmetic mean; phi uses circular mean.
    """
    N, M, Zdim = z_M.shape
    K = (Zdim - 2) // 2
    if how == "map":
        x_rep = x[:, None, :].expand(N, M, x.shape[-1]).reshape(-1, x.shape[-1])
        z_flat = z_M.reshape(-1, Zdim)
        e = energy(x_rep, z_flat).view(N, M)
        idx_min = e.argmin(dim=1)
        z_hat = z_M[torch.arange(N, device=z_M.device), idx_min, :]
        # normalize phi to [0, 2pi)
        z_hat = torch.cat([
            z_hat[:, :1],                       # Rc
            wrap_0_2pi(z_hat[:, 1:2]),          # phi
            z_hat[:, 2:]
        ], dim=-1)
        return z_hat
    elif how == "mean":
        Rc_hat = z_M[:, :, 0].mean(dim=1)
        phi_hat = circular_mean(z_M[:, :, 1], dim=1)
        a_hat = z_M[:, :, 2:2 + K].mean(dim=1)
        b_hat = z_M[:, :, 2 + K: 2 + 2 * K].mean(dim=1)
        return torch.cat([Rc_hat.unsqueeze(-1), phi_hat.unsqueeze(-1), a_hat, b_hat], dim=-1)
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
    indices: Optional[torch.Tensor] = None,
):
    """
    Run conditional Langevin over a chosen subset of the dataset (default: whole dataset)
    and return particles with shape [N_subset, M, z_dim] *in the same order as indices*.
    """
    device = device or default_device()
    energy.eval()

    full_N = len(dataset)
    z_dim = dataset.z.shape[1]
    bank = LatentParticleBank(bank_cfg, n_examples=full_N, z_dim=z_dim, device=device, seed=seed)

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
        return bank.sample_for_batch(all_idx)
    else:
        return bank.sample_for_batch(indices.to(device))


# ==============================
# Decoder training helpers (adapted shapes)
# ==============================

@torch.no_grad()
def map_z_for_batch(energy: EnergyMLP, trainer: LVEBMTrainer, x_batch: torch.Tensor) -> torch.Tensor:
    """
    For each x in the batch, sample M candidates via trainer._posterior_probe(x),
    then pick the MAP z (argmin energy(x, z)).
    Returns: z_map [B, z_dim]
    """
    energy.eval()
    z_samps = trainer._posterior_probe(x_batch)  # [B, M, z_dim]
    B, M, Zdim = z_samps.shape
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
                z_map = map_z_for_batch(energy, trainer, x)  # [B, z_dim]

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

    os.makedirs(os.path.dirname(data_save_path) or ".", exist_ok=True)
    np.savez(
        data_save_path,   # 建议用 .npz 结尾
        X_gt=X_gt,
        X_rec=X_rec,
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


# ==============================
# Main
# ==============================

if __name__ == "__main__":
    device = default_device()
    print("Using device:", device)

    # ----- LCFR-2D dataset as requested -----
    ds = LatentControlledFourierRings2D(
        n=100_000,
        centers=(0.6, 1.5, 2.4),
        K=10,                 # contain at least k=10 to sharpen star points
        sigma_coeff=0.0,      # disable random coeffs; shape fully controlled
        sigma_r=0.02,
        sigma_x=0.02,
        return_z=True,
        seed=123,
        coeffs_override_a={
            1: {3: 0.22},             # center-2: a_3=0.22 -> trefoil (r = R + a3*cos(3φ))
            2: {5: 0.30, 10: -0.12},  # outer ring: a_5>0, a_10<0 -> star (sharper tips)
        },
        # All b_k default to 0; add b_k != 0 to rotate shapes if desired
    )

    # Explicit 80/20 split before any training starts
    gsplit = torch.Generator().manual_seed(77)
    n_total = len(ds)
    perm = torch.randperm(n_total, generator=gsplit)
    n_test = max(1, int(0.2 * n_total))
    test_idx = perm[:n_test]
    train_idx = perm[n_test:]

    # Hyper grids (kept similar; r_min/r_max now interpreted as Rc bounds)
    r_min_list = [0.0]
    r_max_list = [10.0]
    k_steps_list = [20]
    step_size_list = [1e-3]
    noise_scale_list = [1.0]
    seed_list = [77]
    lr_list = [2e-3]

    # r_min_list = [0.0]
    # r_max_list = [10.0]
    # k_steps_list = [20, 25, 30]
    # step_size_list = [1e-3, 2e-3, 8e-4]
    # noise_scale_list = [1.0]
    # seed_list = [77, 177, 277]
    # lr_list = [2e-3, 3e-3]

    # Determine z_dim and K from dataset
    K = ds.meta["K"]
    z_dim = ds.z.shape[1]
    centers_tuple = tuple(float(c) for c in ds.meta["centers"])
    rc_min, rc_max = min(centers_tuple), max(centers_tuple)
    coeff_std_default = float(ds.meta["sigma_coeff"])

    param_grid = list(itertools.product(r_min_list, r_max_list, k_steps_list, step_size_list, noise_scale_list, seed_list, lr_list))
    # param_grid = [
    #     (0.0, 10, 20, 1e-3, 1.0, 77, 2e-3),
    #     (0.0, 10, 25, 1e-3, 1.0, 177, 2e-3),
    #     (0.0, 10, 30, 2e-3, 1.0, 77, 2e-3)
    # ]      

    for i, (r_min, r_max, k_steps, step_size, noise_scale, base_seed, lr) in enumerate(param_grid):
        print(f"\n=== Grid {i+1}/{len(param_grid)}: Rc_min={r_min}, k_steps={k_steps}, step_size={step_size}, noise_scale={noise_scale}, seed={base_seed}, lr={lr} ===")

        output_dir = f"loss_pic_fourier/grid_{i+1}_rcmin{r_min}_k{k_steps}_step{step_size}_noise{noise_scale}_seed{base_seed}_lr{lr}/"
        # if os.path.exists(output_dir):
        #     print(f"Output directory {output_dir} already exists. Skipping this grid.")
        #     continue

        set_seed(base_seed, deterministic=True)
        energy = EnergyMLP(x_dim=2, z_dim=z_dim, K=K, hidden=128, depth=3, lam_x=0.0, lam_coeff=1e-4, lam_rc=1e-5)

        loaded_step = None
        if len(sys.argv) > 1:
            loaded_step = sys.argv[1]
            energy.load_state_dict(torch.load(f"lcfr_noclamp/grid_37_rcmin0.0_k30_step0.001_noise1.0_seed77_lr0.002/energy_model_step_{loaded_step}.pt"))          
        output_dir = f"lcfr_noclamp/grid_37_rcmin0.0_k30_step0.001_noise1.0_seed77_lr0.002/"

        cfg = TrainerConfig(
            x_dim=2, z_dim=z_dim, K=K,
            steps=7000,
            lr=lr,
            batch_size=512,
            log_interval=100,
            seed=base_seed,
            use_infoNCE=True,
            joint=JointBufferConfig(
                size=4096, k_steps=k_steps, step_size_x=step_size, step_size_z=step_size,
                noise_scale=noise_scale, reinit_prob=0.02,
                r_min=r_min if r_min is not None else rc_min,
                r_max=r_max if r_max is not None else rc_max,
                K=K,
                coeff_clip=0.6,
                coeff_std_init=max(1e-3, coeff_std_default if coeff_std_default > 0 else 0.1),
                centers=centers_tuple
            ),
            bank=LatentBankConfig(
                m_per_example=16, k_steps=k_steps, step_size=step_size,
                noise_scale=noise_scale,
                r_min=r_min if r_min is not None else rc_min,
                r_max=r_max if r_max is not None else rc_max,
                K=K,
                coeff_clip=0.6,
                coeff_std_init=max(1e-3, coeff_std_default if coeff_std_default > 0 else 0.1),
                centers=centers_tuple,
                reinit_prob=0.1
            ),
            output_dir=output_dir
        )

        trainer = LVEBMTrainer(energy=energy, dataset=ds, cfg=cfg, train_idx=train_idx, test_idx=test_idx)
        # trainer.train()

        # -------- After energy training: freeze energy and train decoder --------
        energy.eval()
        for p in energy.parameters():
            p.requires_grad_(False)

        decoder = DecoderMLP(K=K, z_dim=z_dim, x_dim=2, hidden=128, depth=3)

        train_decoder(
            trainer=trainer,
            energy=energy,
            decoder=decoder,
            epochs=15,
            lr=1e-3,
            weight_decay=0.0,
            patience=20,
            log_every=1,
            save_plot_path=os.path.join(output_dir, "recon_vs_gt.png") if loaded_step is None else os.path.join(output_dir, f"{loaded_step}_recon_vs_gt.png",),
            data_save_path=os.path.join(output_dir, "recon_data.pt") if loaded_step is None else f"lcfr_noclamp/grid_37_rcmin0.0_k30_step0.001_noise1.0_seed77_lr0.002/{loaded_step}_recon_data.pt"
        )
