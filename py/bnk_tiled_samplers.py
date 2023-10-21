# https://github.com/BlenderNeko/ComfyUI_TiledKSampler
import sys
import os
import itertools
import numpy as np
from tqdm.auto import tqdm
import torch

#sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
import comfy.sd
import comfy.controlnet
import comfy.model_management
import comfy.sample
#from . import tiling
import latent_preview
#import torch
#import itertools
#import numpy as np
MAX_RESOLUTION=8192

def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def create_batches(n, iterable):
    groups = itertools.groupby(iterable, key=lambda x: (x[1], x[3]))
    for _, x in groups:
        for y in grouper(n, x):
            yield y


def get_slice(tensor, h, h_len, w, w_len):
    t = tensor.narrow(-2, h, h_len)
    t = t.narrow(-1, w, w_len)
    return t


def set_slice(tensor1, tensor2, h, h_len, w, w_len, mask=None):
    if mask is not None:
        tensor1[:, :, h:h + h_len, w:w + w_len] = tensor1[:, :, h:h + h_len, w:w + w_len] * (1 - mask) + tensor2 * mask
    else:
        tensor1[:, :, h:h + h_len, w:w + w_len] = tensor2


def get_tiles_and_masks_simple(steps, latent_shape, tile_height, tile_width):
    latent_size_h = latent_shape[-2]
    latent_size_w = latent_shape[-1]
    tile_size_h = int(tile_height // 8)
    tile_size_w = int(tile_width // 8)

    h = np.arange(0, latent_size_h, tile_size_h)
    w = np.arange(0, latent_size_w, tile_size_w)

    def create_tile(hs, ws, i, j):
        h = int(hs[i])
        w = int(ws[j])
        h_len = min(tile_size_h, latent_size_h - h)
        w_len = min(tile_size_w, latent_size_w - w)
        return (h, h_len, w, w_len, steps, None)

    passes = [
        [[create_tile(h, w, i, j) for i in range(len(h)) for j in range(len(w))]],
    ]
    return passes


def get_tiles_and_masks_padded(steps, latent_shape, tile_height, tile_width):
    batch_size = latent_shape[0]
    latent_size_h = latent_shape[-2]
    latent_size_w = latent_shape[-1]

    tile_size_h = int(tile_height // 8)
    tile_size_h = int((tile_size_h // 4) * 4)
    tile_size_w = int(tile_width // 8)
    tile_size_w = int((tile_size_w // 4) * 4)

    # masks
    mask_h = [0, tile_size_h // 4, tile_size_h - tile_size_h // 4, tile_size_h]
    mask_w = [0, tile_size_w // 4, tile_size_w - tile_size_w // 4, tile_size_w]
    masks = [[] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            mask = torch.zeros((batch_size, 1, tile_size_h, tile_size_w), dtype=torch.float32, device='cpu')
            mask[:, :, mask_h[i]:mask_h[i + 1], mask_w[j]:mask_w[j + 1]] = 1.0
            masks[i].append(mask)

    def create_mask(h_ind, w_ind, h_ind_max, w_ind_max, mask_h, mask_w, h_len, w_len):
        mask = masks[1][1]
        if not (h_ind == 0 or h_ind == h_ind_max or w_ind == 0 or w_ind == w_ind_max):
            return get_slice(mask, 0, h_len, 0, w_len)
        mask = mask.clone()
        if h_ind == 0 and mask_h:
            mask += masks[0][1]
        if h_ind == h_ind_max and mask_h:
            mask += masks[2][1]
        if w_ind == 0 and mask_w:
            mask += masks[1][0]
        if w_ind == w_ind_max and mask_w:
            mask += masks[1][2]
        if h_ind == 0 and w_ind == 0 and mask_h and mask_w:
            mask += masks[0][0]
        if h_ind == 0 and w_ind == w_ind_max and mask_h and mask_w:
            mask += masks[0][2]
        if h_ind == h_ind_max and w_ind == 0 and mask_h and mask_w:
            mask += masks[2][0]
        if h_ind == h_ind_max and w_ind == w_ind_max and mask_h and mask_w:
            mask += masks[2][2]
        return get_slice(mask, 0, h_len, 0, w_len)

    h = np.arange(0, latent_size_h, tile_size_h)
    h_shift = np.arange(tile_size_h // 2, latent_size_h - tile_size_h // 2, tile_size_h)
    w = np.arange(0, latent_size_w, tile_size_w)
    w_shift = np.arange(tile_size_w // 2, latent_size_w - tile_size_h // 2, tile_size_w)

    def create_tile(hs, ws, mask_h, mask_w, i, j):
        h = int(hs[i])
        w = int(ws[j])
        h_len = min(tile_size_h, latent_size_h - h)
        w_len = min(tile_size_w, latent_size_w - w)
        mask = create_mask(i, j, len(hs) - 1, len(ws) - 1, mask_h, mask_w, h_len, w_len)
        return (h, h_len, w, w_len, steps, mask)

    passes = [
        [[create_tile(h, w, True, True, i, j) for i in range(len(h)) for j in range(len(w))]],
        [[create_tile(h_shift, w, False, True, i, j) for i in range(len(h_shift)) for j in range(len(w))]],
        [[create_tile(h, w_shift, True, False, i, j) for i in range(len(h)) for j in range(len(w_shift))]],
        [[create_tile(h_shift, w_shift, False, False, i, j) for i in range(len(h_shift)) for j in range(len(w_shift))]],
    ]

    return passes


def mask_at_boundary(h, h_len, w, w_len, tile_size_h, tile_size_w, latent_size_h, latent_size_w, mask, device='cpu'):
    tile_size_h = int(tile_size_h // 8)
    tile_size_w = int(tile_size_w // 8)

    if (h_len == tile_size_h or h_len == latent_size_h) and (w_len == tile_size_w or w_len == latent_size_w):
        return h, h_len, w, w_len, mask
    h_offset = min(0, latent_size_h - (h + tile_size_h))
    w_offset = min(0, latent_size_w - (w + tile_size_w))
    new_mask = torch.zeros((1, 1, tile_size_h, tile_size_w), dtype=torch.float32, device=device)
    new_mask[:, :, -h_offset:h_len if h_offset == 0 else tile_size_h,
    -w_offset:w_len if w_offset == 0 else tile_size_w] = 1.0 if mask is None else mask
    return h + h_offset, tile_size_h, w + w_offset, tile_size_w, new_mask


def get_tiles_and_masks_rgrid(steps, latent_shape, tile_height, tile_width, generator):
    def calc_coords(latent_size, tile_size, jitter):
        tile_coords = int((latent_size + jitter - 1) // tile_size + 1)
        tile_coords = [np.clip(tile_size * c - jitter, 0, latent_size) for c in range(tile_coords + 1)]
        tile_coords = [(c1, c2 - c1) for c1, c2 in zip(tile_coords, tile_coords[1:])]
        return tile_coords

    # calc stuff
    batch_size = latent_shape[0]
    latent_size_h = latent_shape[-2]
    latent_size_w = latent_shape[-1]
    tile_size_h = int(tile_height // 8)
    tile_size_w = int(tile_width // 8)

    tiles_all = []

    for s in range(steps):
        rands = torch.rand((2,), dtype=torch.float32, generator=generator, device='cpu').numpy()

        jitter_w1 = int(rands[0] * tile_size_w)
        jitter_w2 = int(((rands[0] + .5) % 1.0) * tile_size_w)
        jitter_h1 = int(rands[1] * tile_size_h)
        jitter_h2 = int(((rands[1] + .5) % 1.0) * tile_size_h)

        # calc number of tiles
        tiles_h = [
            calc_coords(latent_size_h, tile_size_h, jitter_h1),
            calc_coords(latent_size_h, tile_size_h, jitter_h2)
        ]
        tiles_w = [
            calc_coords(latent_size_w, tile_size_w, jitter_w1),
            calc_coords(latent_size_w, tile_size_w, jitter_w2)
        ]

        tiles = []
        if s % 2 == 0:
            for i, h in enumerate(tiles_h[0]):
                for w in tiles_w[i % 2]:
                    tiles.append((int(h[0]), int(h[1]), int(w[0]), int(w[1]), 1, None))
        else:
            for i, w in enumerate(tiles_w[0]):
                for h in tiles_h[i % 2]:
                    tiles.append((int(h[0]), int(h[1]), int(w[0]), int(w[1]), 1, None))
        tiles_all.append(tiles)
    return [tiles_all]

#######################

def recursion_to_list(obj, attr):
    current = obj
    yield current
    while True:
        current = getattr(current, attr, None)
        if current is not None:
            yield current
        else:
            return

def copy_cond(cond):
    return [[c1,c2.copy()] for c1,c2 in cond]

def slice_cond(tile_h, tile_h_len, tile_w, tile_w_len, cond, area):
    tile_h_end = tile_h + tile_h_len
    tile_w_end = tile_w + tile_w_len
    coords = area[0] #h_len, w_len, h, w,
    mask = area[1]
    if coords is not None:
        h_len, w_len, h, w = coords
        h_end = h + h_len
        w_end = w + w_len
        if h < tile_h_end and h_end > tile_h and w < tile_w_end and w_end > tile_w:
            new_h = max(0, h - tile_h)
            new_w = max(0, w - tile_w)
            new_h_end = min(tile_h_end, h_end - tile_h)
            new_w_end = min(tile_w_end, w_end - tile_w)
            cond[1]['area'] = (new_h_end - new_h, new_w_end - new_w, new_h, new_w)
        else:
            return (cond, True)
    if mask is not None:
        new_mask = get_slice(mask, tile_h,tile_h_len,tile_w,tile_w_len)
        if new_mask.sum().cpu() == 0.0 and 'mask' in cond[1]:
            return (cond, True)
        else:
            cond[1]['mask'] = new_mask
    return (cond, False)

def slice_gligen(tile_h, tile_h_len, tile_w, tile_w_len, cond, gligen):
    tile_h_end = tile_h + tile_h_len
    tile_w_end = tile_w + tile_w_len
    if gligen is None:
        return
    gligen_type = gligen[0]
    gligen_model = gligen[1]
    gligen_areas = gligen[2]
    
    gligen_areas_new = []
    for emb, h_len, w_len, h, w in gligen_areas:
        h_end = h + h_len
        w_end = w + w_len
        if h < tile_h_end and h_end > tile_h and w < tile_w_end and w_end > tile_w:
            new_h = max(0, h - tile_h)
            new_w = max(0, w - tile_w)
            new_h_end = min(tile_h_end, h_end - tile_h)
            new_w_end = min(tile_w_end, w_end - tile_w)
            gligen_areas_new.append((emb, new_h_end - new_h, new_w_end - new_w, new_h, new_w))

    if len(gligen_areas_new) == 0:
        del cond['gligen']
    else:
        cond['gligen'] = (gligen_type, gligen_model, gligen_areas_new)

def slice_cnet(h, h_len, w, w_len, model:comfy.controlnet.ControlBase, img):
    if img is None:
        img = model.cond_hint_original
    model.cond_hint = get_slice(img, h*8, h_len*8, w*8, w_len*8).to(model.control_model.dtype).to(model.device)

def slices_T2I(h, h_len, w, w_len, model:comfy.controlnet.ControlBase, img):
    model.control_input = None
    if img is None:
        img = model.cond_hint_original
    model.cond_hint = get_slice(img, h*8, h_len*8, w*8, w_len*8).float().to(model.device)

# TODO: refactor some of the mess

from PIL import Image

def sample_common(model, add_noise, noise_seed, tile_width, tile_height, tiling_strategy, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0, preview=False):
    end_at_step = min(end_at_step, steps)
    device = comfy.model_management.get_torch_device()
    samples = latent_image["samples"]
    noise_mask = latent_image["noise_mask"] if "noise_mask" in latent_image else None
    force_full_denoise = return_with_leftover_noise == "enable"
    if add_noise == "disable":
        noise = torch.zeros(samples.size(), dtype=samples.dtype, layout=samples.layout, device="cpu")
    else:
        skip = latent_image["batch_index"] if "batch_index" in latent_image else None
        noise = comfy.sample.prepare_noise(samples, noise_seed, skip)

    if noise_mask is not None:
        noise_mask = comfy.sample.prepare_mask(noise_mask, noise.shape, device='cpu')

    shape = samples.shape
    samples = samples.clone()
    
    tile_width = min(shape[-1] * 8, tile_width)
    tile_height = min(shape[2] * 8, tile_height)

    real_model = None
    modelPatches, inference_memory = comfy.sample.get_additional_models(positive, negative, model.model_dtype())
    comfy.model_management.load_models_gpu([model] + modelPatches, comfy.model_management.batch_area_memory(noise.shape[0] * noise.shape[2] * noise.shape[3]) + inference_memory)
    real_model = model.model

    sampler = comfy.samplers.KSampler(real_model, steps=steps, device=device, sampler=sampler_name, scheduler=scheduler, denoise=denoise, model_options=model.model_options)

    if tiling_strategy != 'padded':
        if noise_mask is not None:
            samples += sampler.sigmas[start_at_step].cpu() * noise_mask * model.model.process_latent_out(noise).cpu()
        else:
            samples += sampler.sigmas[start_at_step].cpu() * model.model.process_latent_out(noise).cpu()

    #cnets
    cnets =  comfy.sample.get_models_from_cond(positive, 'control') + comfy.sample.get_models_from_cond(negative, 'control')
    cnets = [m for m in cnets if isinstance(m, comfy.controlnet.ControlNet)]
    cnets = list(set([x for m in cnets for x in recursion_to_list(m, "previous_controlnet")]))
    cnet_imgs = [
        torch.nn.functional.interpolate(m.cond_hint_original, (shape[-2] * 8, shape[-1] * 8), mode='nearest-exact').to('cpu')
        if m.cond_hint_original.shape[-2] != shape[-2] * 8 or m.cond_hint_original.shape[-1] != shape[-1] * 8 else None
        for m in cnets]

    #T2I
    T2Is =  comfy.sample.get_models_from_cond(positive, 'control') + comfy.sample.get_models_from_cond(negative, 'control')
    T2Is = [m for m in T2Is if isinstance(m, comfy.controlnet.T2IAdapter)]
    T2Is = [x for m in T2Is for x in recursion_to_list(m, "previous_controlnet")]
    T2I_imgs = [
        torch.nn.functional.interpolate(m.cond_hint_original, (shape[-2] * 8, shape[-1] * 8), mode='nearest-exact').to('cpu')
        if m.cond_hint_original.shape[-2] != shape[-2] * 8 or m.cond_hint_original.shape[-1] != shape[-1] * 8 or (m.channels_in == 1 and m.cond_hint_original.shape[1] != 1) else None
        for m in T2Is
    ]
    T2I_imgs = [
        torch.mean(img, 1, keepdim=True) if img is not None and m.channels_in == 1 and m.cond_hint_original.shape[1] else img
        for m, img in zip(T2Is, T2I_imgs)
    ]
    
    #cond area and mask
    spatial_conds_pos = [
        (c[1]['area'] if 'area' in c[1] else None, 
            comfy.sample.prepare_mask(c[1]['mask'], shape, device) if 'mask' in c[1] else None)
        for c in positive
    ]
    spatial_conds_neg = [
        (c[1]['area'] if 'area' in c[1] else None, 
            comfy.sample.prepare_mask(c[1]['mask'], shape, device) if 'mask' in c[1] else None)
        for c in negative
    ]

    #gligen
    gligen_pos = [
        c[1]['gligen'] if 'gligen' in c[1] else None
        for c in positive
    ]
    gligen_neg = [
        c[1]['gligen'] if 'gligen' in c[1] else None
        for c in negative
    ]

    positive_copy = comfy.sample.broadcast_cond(positive, shape[0], device)
    negative_copy = comfy.sample.broadcast_cond(negative, shape[0], device)

    gen = torch.manual_seed(noise_seed)
    if tiling_strategy == 'random' or tiling_strategy == 'random strict':
        tiles = get_tiles_and_masks_rgrid(end_at_step - start_at_step, samples.shape, tile_height, tile_width, gen)
    elif tiling_strategy == 'padded':
        tiles = get_tiles_and_masks_padded(end_at_step - start_at_step, samples.shape, tile_height, tile_width)
    else:
        tiles = get_tiles_and_masks_simple(end_at_step - start_at_step, samples.shape, tile_height, tile_width)

    total_steps = sum([num_steps for img_pass in tiles for steps_list in img_pass for _,_,_,_,num_steps,_ in steps_list])
    current_step = [0]

    preview_format = "JPEG"
    if preview_format not in ["JPEG", "PNG"]:
        preview_format = "JPEG"
    previewer = None
    if preview:
        previewer = latent_preview.get_previewer(device, model.model.latent_format)
    
    
    with tqdm(total=total_steps) as pbar_tqdm:
        pbar = comfy.utils.ProgressBar(total_steps)
        
        def callback(step, x0, x, total_steps):
            current_step[0] += 1
            preview_bytes = None
            if previewer:
                preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
            pbar.update_absolute(current_step[0], preview=preview_bytes)
            pbar_tqdm.update(1)
            
        if tiling_strategy == "random strict":
            samples_next = samples.clone()
        for img_pass in tiles:
            for i in range(len(img_pass)):
                for tile_h, tile_h_len, tile_w, tile_w_len, tile_steps, tile_mask in img_pass[i]:
                    tiled_mask = None
                    if noise_mask is not None:
                        tiled_mask = get_slice(noise_mask, tile_h, tile_h_len, tile_w, tile_w_len).to(device)
                    if tile_mask is not None:
                        if tiled_mask is not None:
                            tiled_mask *= tile_mask.to(device)
                        else:
                            tiled_mask = tile_mask.to(device)
                    
                    if tiling_strategy == 'padded' or tiling_strategy == 'random strict':
                        tile_h, tile_h_len, tile_w, tile_w_len, tiled_mask = mask_at_boundary(   tile_h, tile_h_len, tile_w, tile_w_len,
                                                                                                        tile_height, tile_width, samples.shape[-2], samples.shape[-1],
                                                                                                        tiled_mask, device)
                        

                    if tiled_mask is not None and tiled_mask.sum().cpu() == 0.0:
                            continue
                            
                    tiled_latent = get_slice(samples, tile_h, tile_h_len, tile_w, tile_w_len).to(device)
                    
                    if tiling_strategy == 'padded':
                        tiled_noise = get_slice(noise, tile_h, tile_h_len, tile_w, tile_w_len).to(device)
                    else:
                        if tiled_mask is None or noise_mask is None:
                            tiled_noise = torch.zeros_like(tiled_latent)
                        else:
                            tiled_noise = get_slice(noise, tile_h, tile_h_len, tile_w, tile_w_len).to(device) * (1 - tiled_mask)
                    
                    #TODO: all other condition based stuff like area sets and GLIGEN should also happen here

                    #cnets
                    for m, img in zip(cnets, cnet_imgs):
                        slice_cnet(tile_h, tile_h_len, tile_w, tile_w_len, m, img)
                    
                    #T2I
                    for m, img in zip(T2Is, T2I_imgs):
                        slices_T2I(tile_h, tile_h_len, tile_w, tile_w_len, m, img)

                    pos = copy_cond(positive_copy)
                    neg = copy_cond(negative_copy)

                    #cond areas
                    pos = [slice_cond(tile_h, tile_h_len, tile_w, tile_w_len, c, area) for c, area in zip(pos, spatial_conds_pos)]
                    pos = [c for c, ignore in pos if not ignore]
                    neg = [slice_cond(tile_h, tile_h_len, tile_w, tile_w_len, c, area) for c, area in zip(neg, spatial_conds_neg)]
                    neg = [c for c, ignore in neg if not ignore]

                    #gligen
                    for (_, cond), gligen in zip(pos, gligen_pos):
                        slice_gligen(tile_h, tile_h_len, tile_w, tile_w_len, cond, gligen)
                    for (_, cond), gligen in zip(neg, gligen_neg):
                        slice_gligen(tile_h, tile_h_len, tile_w, tile_w_len, cond, gligen)

                    tile_result = sampler.sample(tiled_noise, pos, neg, cfg=cfg, latent_image=tiled_latent, start_step=start_at_step + i * tile_steps, last_step=start_at_step + i*tile_steps + tile_steps, force_full_denoise=force_full_denoise and i+1 == end_at_step - start_at_step, denoise_mask=tiled_mask, callback=callback, disable_pbar=True, seed=noise_seed)
                    tile_result = tile_result.cpu()
                    if tiled_mask is not None:
                        tiled_mask = tiled_mask.cpu()
                    if tiling_strategy == "random strict":
                        set_slice(samples_next, tile_result, tile_h, tile_h_len, tile_w, tile_w_len, tiled_mask)
                    else:
                        set_slice(samples, tile_result, tile_h, tile_h_len, tile_w, tile_w_len, tiled_mask)
                if tiling_strategy == "random strict":
                    samples = samples_next.clone()
                    

    comfy.sample.cleanup_additional_models(modelPatches)

    out = latent_image.copy()
    out["samples"] = samples.cpu()
    return (out, )

class TiledKSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "tile_width": ("INT", {"default": 512, "min": 256, "max": MAX_RESOLUTION, "step": 64}),
                    "tile_height": ("INT", {"default": 512, "min": 256, "max": MAX_RESOLUTION, "step": 64}),
                    "tiling_strategy": (["random", "random strict", "padded", 'simple'], ),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, seed, tile_width, tile_height, tiling_strategy, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise):
        steps_total = int(steps / denoise)
        return sample_common(model, 'enable', seed, tile_width, tile_height, tiling_strategy, steps_total, cfg, sampler_name, scheduler, positive, negative, latent_image, steps_total-steps, steps_total, 'disable', denoise=1.0, preview=True)

class TiledKSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "tile_width": ("INT", {"default": 512, "min": 256, "max": MAX_RESOLUTION, "step": 64}),
                    "tile_height": ("INT", {"default": 512, "min": 256, "max": MAX_RESOLUTION, "step": 64}),
                    "tiling_strategy": (["random", "random strict", "padded", 'simple'], ),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    "preview": (["disable", "enable"], ),
                    }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"

    CATEGORY = "sampling"

    def sample(self, model, add_noise, noise_seed, tile_width, tile_height, tiling_strategy, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, preview, denoise=1.0):
        return sample_common(model, add_noise, noise_seed, tile_width, tile_height, tiling_strategy, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0, preview= preview == 'enable')
