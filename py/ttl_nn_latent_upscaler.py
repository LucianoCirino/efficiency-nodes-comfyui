import torch
#from .latent_resizer import LatentResizer
from comfy import model_management
import os

import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

def normalization(channels):
    return nn.GroupNorm(32, channels)


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = normalization(in_channels)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q, k, v = map(
            lambda x: rearrange(x, "b c h w -> b 1 (h w) c").contiguous(), (q, k, v)
        )
        h_ = nn.functional.scaled_dot_product_attention(
            q, k, v
        )  # scale is dim ** -0.5 per default

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x, **kwargs):
        h_ = x
        h_ = self.attention(h_)
        h_ = self.proj_out(h_)
        return x + h_


def make_attn(in_channels, attn_kwargs=None):
    return AttnBlock(in_channels)


class ResBlockEmb(nn.Module):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout=0,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        kernel_size=3,
        exchange_temb_dims=False,
        skip_t_emb=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.exchange_temb_dims = exchange_temb_dims

        padding = kernel_size // 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, self.out_channels, kernel_size, padding=padding),
        )

        self.skip_t_emb = skip_t_emb
        self.emb_out_channels = (
            2 * self.out_channels if use_scale_shift_norm else self.out_channels
        )
        if self.skip_t_emb:
            print(f"Skipping timestep embedding in {self.__class__.__name__}")
            assert not self.use_scale_shift_norm
            self.emb_layers = None
            self.exchange_temb_dims = False
        else:
            self.emb_layers = nn.Sequential(
                nn.SiLU(),
                nn.Linear(
                    emb_channels,
                    self.emb_out_channels,
                ),
            )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                nn.Conv2d(
                    self.out_channels,
                    self.out_channels,
                    kernel_size,
                    padding=padding,
                )
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv2d(
                channels, self.out_channels, kernel_size, padding=padding
            )
        else:
            self.skip_connection = nn.Conv2d(channels, self.out_channels, 1)

    def forward(self, x, emb):
        h = self.in_layers(x)

        if self.skip_t_emb:
            emb_out = torch.zeros_like(h)
        else:
            emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            if self.exchange_temb_dims:
                emb_out = rearrange(emb_out, "b t c ... -> b c t ...")
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class LatentResizer(nn.Module):
    def __init__(self, in_blocks=10, out_blocks=10, channels=128, dropout=0, attn=True):
        super().__init__()
        self.conv_in = nn.Conv2d(4, channels, 3, padding=1)

        self.channels = channels
        embed_dim = 32
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.in_blocks = nn.ModuleList([])
        for b in range(in_blocks):
            if (b == 1 or b == in_blocks - 1) and attn:
                self.in_blocks.append(make_attn(channels))
            self.in_blocks.append(ResBlockEmb(channels, embed_dim, dropout))

        self.out_blocks = nn.ModuleList([])
        for b in range(out_blocks):
            if (b == 1 or b == out_blocks - 1) and attn:
                self.out_blocks.append(make_attn(channels))
            self.out_blocks.append(ResBlockEmb(channels, embed_dim, dropout))

        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv2d(channels, 4, 3, padding=1)

    @classmethod
    def load_model(cls, filename, device="cpu", dtype=torch.float32, dropout=0):
        if not 'weights_only' in torch.load.__code__.co_varnames:
            weights = torch.load(filename, map_location=torch.device("cpu"))
        else:
            weights = torch.load(filename, map_location=torch.device("cpu"), weights_only=True)
        in_blocks = 0
        out_blocks = 0
        in_tfs = 0
        out_tfs = 0
        channels = weights["conv_in.bias"].shape[0]
        for k in weights.keys():
            k = k.split(".")
            if k[0] == "in_blocks":
                in_blocks = max(in_blocks, int(k[1]))
                if k[2] == "q" and k[3] == "weight":
                    in_tfs += 1
            if k[0] == "out_blocks":
                out_blocks = max(out_blocks, int(k[1]))
                if k[2] == "q" and k[3] == "weight":
                    out_tfs += 1
        in_blocks = in_blocks + 1 - in_tfs
        out_blocks = out_blocks + 1 - out_tfs
        resizer = cls(
            in_blocks=in_blocks,
            out_blocks=out_blocks,
            channels=channels,
            dropout=dropout,
            attn=(out_tfs != 0),
        )
        resizer.load_state_dict(weights)
        resizer.eval()
        resizer.to(device, dtype=dtype)
        return resizer

    def forward(self, x, scale=None, size=None):
        if scale is None and size is None:
            raise ValueError("Either scale or size needs to be not None")
        if scale is not None and size is not None:
            raise ValueError("Both scale or size can't be not None")
        if scale is not None:
            size = (x.shape[-2] * scale, x.shape[-1] * scale)
            size = tuple([int(round(i)) for i in size])
        else:
            scale = size[-1] / x.shape[-1]

        # Output is the same size as input
        if size == x.shape[-2:]:
            return x

        scale = torch.tensor([scale - 1], dtype=x.dtype).to(x.device).unsqueeze(0)
        emb = self.embed(scale)

        x = self.conv_in(x)

        for b in self.in_blocks:
            if isinstance(b, ResBlockEmb):
                x = b(x, emb)
            else:
                x = b(x)
        x = F.interpolate(x, size=size, mode="bilinear")
        for b in self.out_blocks:
            if isinstance(b, ResBlockEmb):
                x = b(x, emb)
            else:
                x = b(x)

        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x

########################################################
class NNLatentUpscale:
    """
    Upscales SDXL latent using neural network
    """

    def __init__(self):
        self.local_dir = os.path.dirname(os.path.realpath(__file__))
        self.scale_factor = 0.13025
        self.dtype = torch.float32
        if model_management.should_use_fp16():
            self.dtype = torch.float16
        self.weight_path = {
            "SDXL": os.path.join(self.local_dir, "sdxl_resizer.pt"),
            "SD 1.x": os.path.join(self.local_dir, "sd15_resizer.pt"),
        }
        self.version = "none"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "latent": ("LATENT",),
                "version": (["SDXL", "SD 1.x"],),
                "upscale": (
                    "FLOAT",
                    {
                        "default": 1.5,
                        "min": 1.0,
                        "max": 2.0,
                        "step": 0.01,
                        "display": "number",
                    },
                ),
            },
        }

    RETURN_TYPES = ("LATENT",)

    FUNCTION = "upscale"

    CATEGORY = "latent"

    def upscale(self, latent, version, upscale):
        device = model_management.get_torch_device()
        samples = latent["samples"].to(device=device, dtype=self.dtype)

        if version != self.version:
            self.model = LatentResizer.load_model(self.weight_path[version], device, self.dtype)
            self.version = version

        self.model.to(device=device)
        latent_out = (self.model(self.scale_factor * samples, scale=upscale) / self.scale_factor)

        if self.dtype != torch.float32:
            latent_out = latent_out.to(dtype=torch.float32)

        latent_out = latent_out.to(device="cpu")

        self.model.to(device=model_management.vae_offload_device())
        return ({"samples": latent_out},)