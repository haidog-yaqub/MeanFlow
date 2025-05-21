import torch
import torch.nn.functional as F
from einops import rearrange
from functools import partial
import numpy as np


def normalize_to_neg1_1(x):
    return x * 2 - 1


def unnormalize_to_0_1(x):
    return (x + 1) * 0.5


def stopgrad(x):
    return x.detach()


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (batch, dim)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    delta_sq = torch.sum(error ** 2, dim=-1)  # ||Δ||^2 per sample
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()


class MeanFlow:
    def __init__(
        self,
        channels=1,
        image_size=32,
        num_classes=10,
        flow_ratio=0.50,
        cfg_ratio=0.20,
        cfg_scale=2.0,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_classes = num_classes
        self.use_cond = num_classes is not None
        self.flow_ratio = flow_ratio
        self.cfg_ratio = cfg_ratio
        self.w = cfg_scale

    def loss(self, model, x, c=None):
        batch_size = x.shape[0]
        device = x.device

        t_np = np.random.rand(batch_size).astype(np.float32)
        r_np = np.random.rand(batch_size).astype(np.float32)
        num_selected = int(self.flow_ratio * batch_size)
        indices = np.random.permutation(batch_size)[:num_selected]
        r_np[indices] = t_np[indices]

        t = torch.tensor(t_np, device=device)
        r = torch.tensor(r_np, device=device)

        t_ = rearrange(t, "b -> b 1 1 1")
        r_ = rearrange(r, "b -> b 1 1 1")

        e = torch.randn_like(x)
        x = normalize_to_neg1_1(x)

        z = (1 - t_) * x + t_ * e
        v = e - x

        if self.w is not None:
            uncond = torch.ones_like(c) * self.num_classes
            with torch.no_grad():
                u_t = model(z, t, t, uncond)
            v_hat = self.w * v + (1 - self.w) * u_t
            cfg_mask = torch.rand_like(c.float()) < self.cfg_ratio
            c = torch.where(cfg_mask, uncond, c)
        else:
            v_hat = v

        model_partial = partial(model, y=c)
        u, dudt = torch.autograd.functional.jvp(
            lambda z, t, r: model_partial(z, t, r),
            # model,
            (z, t, r),
            (v_hat, torch.ones_like(t), torch.zeros_like(r)),
            create_graph=True
        )

        # u = model(z, t, r)
        # u_tgt = v
        u_tgt = v - (t_ - r_) * dudt

        error = u - stopgrad(u_tgt)
        loss = adaptive_l2_loss(error)
        # loss = F.mse_loss(u, stopgrad(u_tgt))

        mse_val = (stopgrad(error) ** 2).mean()
        return loss, mse_val

    @torch.no_grad()
    def sample_each_class(self, model, n_per_class,
                          sample_steps=1, device='cuda'):
        model.eval()

        c = torch.arange(self.num_classes, device=device).repeat(n_per_class)
        z = torch.randn(self.num_classes * n_per_class, self.channels,
                        self.image_size, self.image_size, device=device)

        t = torch.ones((c.shape[0],), device=c.device)
        r = torch.zeros((c.shape[0],), device=c.device)

        z = z - model(z, t, r, c)

        z = unnormalize_to_0_1(z.clip(-1, 1))

        return z