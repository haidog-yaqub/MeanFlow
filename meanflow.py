import torch
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import numpy as np


class Normalizer:
    # minmax for raw image, mean_std for vae latent
    def __init__(self, mode='minmax', mean=None, std=None):
        assert mode in ['minmax', 'mean_std'], "mode must be 'minmax' or 'mean_std'"
        self.mode = mode

        if mode == 'mean_std':
            if mean is None or std is None:
                raise ValueError("mean and std must be provided for 'mean_std' mode")
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

    @classmethod
    def from_list(cls, config):
        """
        config: [mode, mean, std]
        """
        mode, mean, std = config
        return cls(mode, mean, std)

    def norm(self, x):
        if self.mode == 'minmax':
            return x * 2 - 1
        elif self.mode == 'mean_std':
            return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def unnorm(self, x):
        if self.mode == 'minmax':
            x = x.clip(-1, 1)
            return (x + 1) * 0.5
        elif self.mode == 'mean_std':
            return x * self.std.to(x.device) + self.mean.to(x.device)


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.0, c=1e-2):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.mean(error ** 2, dim=(1, 2, 3), keepdim=False)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq
    return (stopgrad(w) * loss).mean()


class MeanFlow:
    def __init__(
        self,
        channels=1,
        image_size=32,
        num_classes=10,
        normalizer=['minmax', None, None],
        mode='i-meanflow',
        # time distribution, mu, sigma
        time_dist=['lognorm', -0.4, 1.0],
        cfg_ratio=0.10,
        # scalar or [low, high] range; None disables CFG
        cfg_scale=[1.0, 5.0],
        cfg_scale_ratio=0.75,
        adaptive_l2_gamma=0.0,
        adaptive_l2_c=1e-2,
    ):
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_cond = num_classes is not None

        self.normer = Normalizer.from_list(normalizer)

        assert mode in ('meanflow', 'i-meanflow')
        self.mode = mode
        self.time_dist = time_dist
        self.cfg_ratio = cfg_ratio
        self.cfg_scale_ratio = cfg_scale_ratio
        self.use_cfg_w = cfg_scale is not None
        if self.use_cfg_w and isinstance(cfg_scale, (int, float)):
            self.cfg_w_fixed = float(cfg_scale)
            self.cfg_w_range = None
        elif self.use_cfg_w:
            self.cfg_w_fixed = None
            self.cfg_w_range = (float(cfg_scale[0]), float(cfg_scale[1]))
        else:
            self.cfg_w_fixed = None
            self.cfg_w_range = None
        self.adaptive_l2_gamma = adaptive_l2_gamma
        self.adaptive_l2_c = adaptive_l2_c

    def sample_t_r(self, batch_size, device, sample_cfg=True):
        if self.time_dist[0] == 'uniform':
            samples = np.random.rand(batch_size, 2).astype(np.float32)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
            samples = 1 / (1 + np.exp(-normal_samples))

        t_np = np.maximum(samples[:, 0], samples[:, 1])
        r_np = np.minimum(samples[:, 0], samples[:, 1])

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)

        if not self.use_cfg_w:
            w = torch.ones(batch_size, device=device)
        elif self.cfg_w_fixed is not None:
            w = torch.full((batch_size,), self.cfg_w_fixed, device=device)
        else:
            low, high = self.cfg_w_range
            if sample_cfg:
                w = torch.ones(batch_size, device=device)
                num = int(self.cfg_scale_ratio * batch_size)
                if num > 0:
                    indices = torch.randperm(batch_size, device=device)[:num]
                    w[indices] = torch.empty(num, device=device).uniform_(low, high)
            else:
                w = torch.full((batch_size,), (low + high) / 2, device=device)

        return t, r, w

    def loss(self, model, x, c=None):
        batch_size = x.shape[0]
        device = x.device

        t, r, w = self.sample_t_r(batch_size, device)

        t_ = rearrange(t, "b -> b 1 1 1").detach().clone()
        r_ = rearrange(r, "b -> b 1 1 1").detach().clone()

        e = torch.randn_like(x)
        x = self.normer.norm(x)

        z = (1 - t_) * x + t_ * e
        v = e - x
        v_hat = v

        if c is not None and self.use_cfg_w:
            uncond = torch.ones_like(c) * self.num_classes
            uncond_w = torch.ones_like(w)
            cfg_mask = torch.rand_like(c.float()) < self.cfg_ratio
            c = torch.where(cfg_mask, uncond, c)
            w = torch.where(cfg_mask, torch.ones_like(w), w)
            w_ = rearrange(w, "b -> b 1 1 1")

            with torch.no_grad():
                _, v_c = model(z, t, r, y=c, w=w)
                _, v_uc = model(z, t, r, y=uncond, w=uncond_w)
                v_hat = v + (1 - 1 / w_) * (v_c - v_uc)
                # cfg_mask_ = rearrange(cfg_mask, "b -> b 1 1 1").bool()
                # v_hat = torch.where(cfg_mask_, v, v_hat)  # redundant: w=1 on uncond => (1 - 1/w_)=0
        else:
            with torch.no_grad():
                _, v_c = model(z, t, r, y=c, w=w)

        with torch.no_grad():
            model_partial = partial(
                model, y=c, w=w, return_v=False, use_flash_attention=False
            )
            _, dudt = torch.autograd.functional.jvp(
                model_partial,
                (z, t, r),
                (v_c, torch.ones_like(t), torch.zeros_like(r)),
                create_graph=False,
            )

        u_p, v_p = model(z, t, r, y=c, w=w)

        fm_loss = adaptive_l2_loss(
            v_p - stopgrad(v_hat),
            gamma=self.adaptive_l2_gamma,
            c=self.adaptive_l2_c,
        )

        v_est = u_p + (t_ - r_) * stopgrad(dudt)

        if self.mode == 'meanflow':
            u_tgt = v_hat - (t_ - r_) * dudt
            mf_loss = adaptive_l2_loss(
                u_p - stopgrad(u_tgt),
                gamma=self.adaptive_l2_gamma,
                c=self.adaptive_l2_c,
            )
        else:
            mf_loss = adaptive_l2_loss(
                v_est - stopgrad(v_hat),
                gamma=self.adaptive_l2_gamma,
                c=self.adaptive_l2_c,
            )

        mse_val = {
            'fm_loss': stopgrad(fm_loss),
            'mf_loss': stopgrad(mf_loss),
            'mf_v_mse': (stopgrad(v_est - v_hat) ** 2).mean(),
        }
        return mf_loss + fm_loss, mse_val

    @torch.no_grad()
    def sample_each_class(self, model, n_per_class, classes=None,
                          sample_steps=5, device='cuda'):
        model.eval()

        if classes is None:
            c = torch.arange(self.num_classes, device=device).repeat(n_per_class)
        else:
            c = torch.tensor(classes, device=device).repeat(n_per_class)

        z = torch.randn(
            c.shape[0], self.channels, self.image_size, self.image_size, device=device
        )

        t_vals = torch.linspace(1.0, 0.0, sample_steps + 1, device=device)
        _, _, w = self.sample_t_r(c.shape[0], device, sample_cfg=False)

        for i in range(sample_steps):
            t = torch.full((z.size(0),), t_vals[i], device=device)
            r = torch.full((z.size(0),), t_vals[i + 1], device=device)

            t_ = rearrange(t, "b -> b 1 1 1")
            r_ = rearrange(r, "b -> b 1 1 1")

            _, v = model(z, t, r, y=c, w=w)
            z = z - (t_ - r_) * v

        z = self.normer.unnorm(z)
        return z
