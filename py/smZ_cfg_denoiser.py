# https://github.com/shiimizu/ComfyUI_smZNodes
import os
import re
import inspect
import itertools
import functools
import importlib
from typing import List, Any, Optional, Union, Tuple, Dict, Callable, Type, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

import comfy
import comfy.sample
from comfy import model_management
from comfy.samplers import KSampler, KSamplerX0Inpaint

import nodes


def adjust_tensor_shape(
    tensor_small: torch.Tensor, tensor_big: torch.Tensor
) -> torch.Tensor:
    replication_factor = (
        tensor_big.size(1) + tensor_small.size(1) - 1
    ) // tensor_small.size(1)
    return tensor_small.repeat_interleave(replication_factor, dim=1)[
        :, : tensor_big.size(1), :
    ]


def expand(
    tensor1: torch.Tensor, tensor2: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if tensor1.size(1) != tensor2.size(1):
        if tensor1.size(1) < tensor2.size(1):
            tensor1 = adjust_tensor_shape(tensor1, tensor2)
        else:
            tensor2 = adjust_tensor_shape(tensor2, tensor1)
    return tensor1, tensor2


def bounded_modulo(number: int, modulo_value: int) -> int:
    return number % modulo_value


def calc_cond(c: List[Any], current_step: int) -> List[Any]:
    grouped_by_smZ = {
        k: list(g)
        for k, g in itertools.groupby(
            sorted(c, key=lambda x: x[1].get("from_smZ")),
            key=lambda x: x[1].get("from_smZ"),
        )
    }
    result_cond = []
    for from_smZ, group in grouped_by_smZ.items():
        grouped_by_smZid = {
            k: list(g)
            for k, g in itertools.groupby(
                sorted(group, key=lambda x: x[1].get("smZid")),
                key=lambda x: x[1].get("smZid"),
            )
        }
        for smZid, group_smZid in grouped_by_smZid.items():
            if smZid is not None:
                orig_len = group_smZid[0][1].get("orig_len", 1)
                i = bounded_modulo(current_step, orig_len)
                result_cond.extend(group_smZid[i : i + 1])
            else:
                result_cond.extend(group_smZid)
    return result_cond


def catenate_conds(conds: List[Any]) -> Any:
    if not isinstance(conds[0], dict):
        return torch.cat(conds)
    return {key: torch.cat([x[key] for x in conds]) for key in conds[0].keys()}


def subscript_cond(cond: Any, a: int, b: int) -> Any:
    if not isinstance(cond, dict):
        return cond[a:b]
    return {key: vec[a:b] for key, vec in cond.items()}


def pad_cond(tensor: torch.Tensor, repeats: int, empty: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, dict):
        return F.pad(tensor, (0, 0, 0, repeats), "constant", 0)
    tensor["crossattn"] = pad_cond(tensor["crossattn"], repeats, empty)
    return tensor


class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
        self.mask = None
        self.nmask = None
        self.init_latent = None
        self.steps = None
        self.total_steps = None
        self.step = 0
        self.image_cfg_scale = None
        self.padded_cond_uncond = False
        self.sampler = None
        self.p = None
        self.mask_before_denoising = False

    def combine_denoised(self, x_out, conds_list, uncond, cond_scale):
        denoised_uncond = x_out[-uncond.shape[0] :]
        denoised = denoised_uncond.clone()

        for i, conds in enumerate(conds_list):
            denoised[i] += sum(
                (x_out[cond_index] - denoised_uncond[i]) * weight * cond_scale
                for cond_index, weight in conds
            )

        return denoised

    def combine_denoised_for_edit_model(self, x_out, cond_scale):
        out_cond, out_img_cond, out_uncond = x_out.chunk(3)
        return (
            out_uncond
            + cond_scale * (out_cond - out_img_cond)
            + self.image_cfg_scale * (out_img_cond - out_uncond)
        )

    def forward(self, x, sigma, uncond, cond, cond_scale, s_min_uncond, image_cond):
        model_management.throw_exception_if_processing_interrupted()

        conds_list, tensor = cond
        batch_size = len(conds_list)
        repeats = [len(conds) for conds in conds_list]

        x_in, sigma_in, image_cond_in = self.prepare_inputs(
            x, sigma, image_cond, repeats, uncond, s_min_uncond
        )

        cond_in = self.prepare_cond(tensor, uncond, repeats, s_min_uncond)

        x_out = self.apply_model(x_in, sigma_in, cond_in, image_cond_in, batch_size)

        denoised = self.post_process(
            x_out, conds_list, uncond, cond_scale, s_min_uncond
        )

        return denoised

    def prepare_inputs(self, x, sigma, image_cond, repeats, uncond, s_min_uncond):
        x_in = torch.cat(
            [x[i].expand(n, -1, -1, -1) for i, n in enumerate(repeats)] + [x]
        )
        sigma_in = torch.cat(
            [sigma[i].expand(n) for i, n in enumerate(repeats)] + [sigma]
        )
        image_cond_in = torch.cat(
            [image_cond[i].expand(n, -1, -1, -1) for i, n in enumerate(repeats)]
            + [image_cond]
        )

        if self.step % 2 and s_min_uncond > 0 and sigma[0] < s_min_uncond:
            x_in = x_in[: -len(repeats)]
            sigma_in = sigma_in[: -len(repeats)]

        return x_in, sigma_in, image_cond_in

    def prepare_cond(self, tensor, uncond, repeats, s_min_uncond):
        if self.step % 2 and s_min_uncond > 0 and tensor.shape[1] == uncond.shape[1]:
            return torch.cat([tensor] * sum(repeats) + [uncond])
        return torch.cat([tensor] * sum(repeats))

    def apply_model(self, x_in, sigma_in, cond_in, image_cond_in, batch_size):
        x_out = torch.zeros_like(x_in)
        for batch_offset in range(0, x_in.shape[0], batch_size):
            batch_slice = slice(batch_offset, batch_offset + batch_size)
            x_out[batch_slice] = self.inner_model(
                x_in[batch_slice],
                sigma_in[batch_slice],
                cond_in[batch_slice],
                image_cond_in[batch_slice],
            )
        return x_out

    def post_process(self, x_out, conds_list, uncond, cond_scale, s_min_uncond):
        if self.step % 2 and s_min_uncond > 0:
            denoised_image_indexes = [x[0][0] for x in conds_list]
            fake_uncond = torch.cat([x_out[i : i + 1] for i in denoised_image_indexes])
            x_out = torch.cat([x_out, fake_uncond])

        denoised = self.combine_denoised(x_out, conds_list, uncond, cond_scale)

        if not self.mask_before_denoising and self.mask is not None:
            denoised = self.init_latent * self.mask + self.nmask * denoised

        self.step += 1
        return denoised


class CFGNoisePredictor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.ksampler = next(
            obj for obj in gc.get_objects() if isinstance(obj, KSampler)
        )
        self.step = 0
        self.orig = model
        self.inner_model = model
        self.inner_model2 = model.apply_model
        self.inner_model2.num_timesteps = model.num_timesteps
        self.inner_model2.device = getattr(self.ksampler, "device", None)
        self.s_min_uncond = 0.0
        self.alphas_cumprod = model.alphas_cumprod
        self.c_adm = None
        self.init_cond = None
        self.init_uncond = None
        self.is_prompt_editing_u = False
        self.is_prompt_editing_c = False

    def apply_model(
        self,
        x,
        timestep,
        cond,
        uncond,
        cond_scale,
        cond_concat=None,
        model_options={},
        seed=None,
    ):
        cc, uu = self.prepare_conditions(cond, uncond)
        self.step += 1
        self.update_model_options(cc, uu, model_options)
        c, u = cc[1], uu[1]
        _cc, _uu = self.expand_conditions(cc[0], uu[0], c)
        x.c_adm = self.c_adm
        conds_list = c.get("conds_list", [[(0, 1.0)]])
        image_cond = self.txt2img_image_conditioning(x)
        out = self.inner_model2(
            x,
            timestep,
            cond=(conds_list, _cc),
            uncond=_uu,
            cond_scale=cond_scale,
            s_min_uncond=self.s_min_uncond,
            image_cond=image_cond,
        )
        return out

    def prepare_conditions(self, cond, uncond):
        cc = calc_cond(cond, self.step)
        uu = calc_cond(uncond, self.step)
        cc = next((c for c in cc if c[1].get("from_smZ", False)), cc[0])
        uu = next((u for u in uu if u[1].get("from_smZ", False)), uu[0])
        return cc, uu

    def update_model_options(self, cc, uu, model_options):
        if any(p[1].get("from_smZ", False) for p in cc + uu):
            model_options.setdefault("transformer_options", {})["from_smZ"] = True

    def expand_conditions(self, _cc, _uu, c):
        if c.get("adm_encoded", None) is not None:
            self.c_adm = torch.cat([c["adm_encoded"], _uu["adm_encoded"]])
            _cc, _uu = torch.broadcast_tensors(_cc, _uu)
        return _cc, _uu

    @staticmethod
    def txt2img_image_conditioning(x):
        return x.new_zeros((x.shape[0], 5, 1, 1), dtype=x.dtype)


class SDKSampler(KSampler):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_denoise = CFGNoisePredictor(model)
        self.model_wrap = self.set_model_wrap(self.model_denoise, model)
        self.model_k = KSamplerX0Inpaint(self.model_wrap)

    def set_model_wrap(self, model_denoise, model):
        parameterization = getattr(model, "parameterization", "eps")
        quantize = True
        return KSamplerX0Inpaint(
            model_denoise, quantize=quantize, parameterization=parameterization
        )
