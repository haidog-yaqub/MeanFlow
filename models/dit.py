import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.attention import sdpa_kernel, SDPBackend
from timm.models.vision_transformer import PatchEmbed, Mlp


def expand_ada_params(param, x):
    """Expand global (B, D) adaLN params to (B, T, D); pass through token-level params."""
    if param.ndim == 2:
        return param.unsqueeze(1)
    return param


def modulate(x, scale, shift):
    """
    Apply adaLN modulation.
    x: (B, T, D)
    scale, shift: (B, D) for global adaLN or (B, T, D) for token-level adaLN
    """
    scale = expand_ada_params(scale, x)
    shift = expand_ada_params(shift, x)
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    def __init__(self, dim, nfreq=256, scale=1000.0):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nfreq = nfreq
        self.scale = scale

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t = t * self.scale
        t_freq = self.timestep_embedding(t, self.nfreq)
        t_emb = self.mlp(t_freq)
        return t_emb

    def initialize_weights(self):
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)


class LabelEmbedder(nn.Module):
    def __init__(self, num_classes, dim):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, dim)
        self.num_classes = num_classes

    def forward(self, labels):
        embeddings = self.embedding(labels)
        return embeddings


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def _native_attention(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return attn @ v

    def _flash_attention(self, q, k, v):
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, scale=self.scale)

    def forward(self, x, use_flash_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if (
            use_flash_attention
            and q.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
        ):
            x = self._flash_attention(q, k, v)
        else:
            x = self._native_attention(q, k, v)

        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, qk_norm=True)
        self.norm2 = nn.RMSNorm(dim, elementwise_affine=False)
        mlp_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=dim, hidden_features=mlp_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c, use_flash_attention=False):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + expand_ada_params(gate_msa, x) * self.attn(
            modulate(self.norm1(x), scale_msa, shift_msa),
            use_flash_attention=use_flash_attention,
        )
        x = x + expand_ada_params(gate_mlp, x) * self.mlp(
            modulate(self.norm2(x), scale_mlp, shift_mlp)
        )
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_dim):
        super().__init__()
        self.norm_final = nn.RMSNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(dim, patch_size * patch_size * out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), scale, shift)
        x = self.linear(x)
        return x


class MFDiT(nn.Module):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        dim=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        num_classes=1000,
        max_cfg=4.0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.max_cfg = max_cfg

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, dim)
        self.t_embedder = TimestepEmbedder(dim)
        self.r_embedder = TimestepEmbedder(dim)
        self.w_embedder = TimestepEmbedder(dim)

        self.use_cond = num_classes is not None
        self.y_embedder = LabelEmbedder(num_classes, dim) if self.use_cond else None

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, dim), requires_grad=True)

        self.blocks = nn.ModuleList([
            DiTBlock(dim, num_heads, mlp_ratio) for _ in range(depth)
        ])
        self.final_layer_v = FinalLayer(dim, patch_size, self.out_channels)
        self.final_layer_u = FinalLayer(dim, patch_size, self.out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.embedding.weight, std=0.02)

        # Initialize timestep embedding MLP (t, r, w):
        for embedder in (self.t_embedder, self.r_embedder, self.w_embedder):
            embedder.initialize_weights()

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        for final_layer in (self.final_layer_v, self.final_layer_u):
            nn.init.constant_(final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(final_layer.linear.weight, 0)
            nn.init.constant_(final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, r, y=None, w=None, return_v=True, use_flash_attention=False):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        r: (N,) tensor of auxiliary timesteps
        y: (N,) tensor of class labels
        w: (N,) CFG scale; normalized by max_cfg before embedding. Defaults to 1.0.
        return_v: if False, return u only; otherwise return (u, v)
        use_flash_attention: if True, use flash SDPA when dtype/device allow
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2

        if w is None:
            w = torch.ones_like(t)
        w = w / self.max_cfg

        t = self.t_embedder(t)                   # (N, D)
        r = self.r_embedder(r)
        w = self.w_embedder(w)
        t = t + r + w

        c = t
        if self.use_cond:
            y = self.y_embedder(y)               # (N, D)
            c = c + y                            # (N, D)

        for block in self.blocks:
            x = block(x, c, use_flash_attention=use_flash_attention)

        u = self.unpatchify(self.final_layer_u(x, c))
        if not return_v:
            return u

        v = self.unpatchify(self.final_layer_v(x, c))
        return u, v


# Positional embedding from:
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb