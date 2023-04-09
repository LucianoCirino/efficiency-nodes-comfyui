# Efficiency Nodes - A collection of my ComfyUI custom nodes to help streamline workflows and reduce total node count.
#  by Luciano Cirino (Discord: TSC#9184) - April 2023

from nodes import common_ksampler
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
import numpy as np
import torch

import os
import sys
import json
import copy
import folder_paths


# Get the absolute path of the parent directory of the current script
my_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the ComfyUI directory
comfy_dir = os.path.abspath(os.path.join(my_dir, '..', '..'))

# Add the ComfyUI directory path to the sys.path list
sys.path.append(comfy_dir)

# Import functions from nodes.py in the ComfyUI directory
import comfy.samplers
import comfy.sd
import comfy.utils

MAX_RESOLUTION=8192

# Tensor to PIL (grabbed from WAS Suite)
def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor (grabbed from WAS Suite)
def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


# TSC Efficient Loader
# Track what objects have already been loaded into memory (*only for instances of this node)
loaded_objects = {
    "ckpt": [], # (ckpt_name, location)
    "clip": [], # (ckpt_name, location)
    "bvae": [], # (ckpt_name, location)
    "vae": []   # (vae_name, location)
}
class TSC_EfficientLoader:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                              "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                              "clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
                              "positive": ("STRING", {"default": "Positive","multiline": True}),
                              "negative": ("STRING", {"default": "Negative", "multiline": True}),
                              "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                              "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})
                             }}

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP" ,)
    RETURN_NAMES = ("MODEL", "CONDITIONING+", "CONDITIONING-", "LATENT", "VAE", "CLIP", )
    FUNCTION = "efficientloader"

    CATEGORY = "Efficiency Nodes/Loaders"

    def efficientloader(self, ckpt_name, vae_name, clip_skip, positive, negative, empty_latent_width, empty_latent_height, batch_size,
                        output_vae=False, output_clip=True):

        # Baked VAE setup
        if vae_name == "Baked VAE":
            output_vae = True

        # Search for tuple index that contains ckpt_name in "ckpt" array of loaded_lbjects
        checkpoint_found = False
        for i, entry in enumerate(loaded_objects["ckpt"]):
            if entry[0] == ckpt_name:
                # Extract the second element of the tuple at 'i' in the "ckpt", "clip", "bvae" arrays
                model = loaded_objects["ckpt"][i][1]
                clip = loaded_objects["clip"][i][1]
                vae = loaded_objects["bvae"][i][1]
                checkpoint_found = True
                break

        # If not found, load ckpt
        if checkpoint_found == False:
            # Load Checkpoint
            ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
            out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
            model = out[0]
            clip = out[1]
            vae = out[2]

            # Update loaded_objects[] array
            loaded_objects["ckpt"].append((ckpt_name, out[0]))
            loaded_objects["clip"].append((ckpt_name, out[1]))
            loaded_objects["bvae"].append((ckpt_name, out[2]))

        # Create Empty Latent
        latent = torch.zeros([batch_size, 4, empty_latent_height // 8, empty_latent_width // 8]).cpu()

        # Check for custom VAE
        if vae_name != "Baked VAE":
            # Check if vae_name exists in "vae" array
            if any(entry[0] == vae_name for entry in loaded_objects["vae"]):
                # Extract the second tuple entry of the checkpoint
                vae = [entry[1] for entry in loaded_objects["vae"] if entry[0] == vae_name][0]
            else:
                vae_path = folder_paths.get_full_path("vae", vae_name)
                vae = comfy.sd.VAE(ckpt_path=vae_path)
                # Update loaded_objects[] array
                loaded_objects["vae"].append((vae_name, vae))

        # CLIP skip
        clip = clip.clone()
        clip.clip_layer(clip_skip)

        return (model, [[clip.encode(positive), {}]], [[clip.encode(negative), {}]], {"samples":latent}, vae, clip, )


# TSC KSampler (Efficient)
last_helds = {
    "results": [None for _ in range(15)],
    "latent": [None for _ in range(15)],
    "images": [None for _ in range(15)]
}
class TSC_KSampler:

    def __init__(self):
        self.output_dir = os.path.join(comfy_dir, 'temp')
        self.type = "temp"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"sampler_state": (["Sample", "Hold"], ),
                     "my_unique_id": ("INT", {"default": 0, "min": 0, "max": 15}),
                     "model": ("MODEL",),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     "latent_image": ("LATENT",),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "preview_image": (["Disabled", "Enabled"],),
                     },
                "optional": { "optional_vae": ("VAE",), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "IMAGE", )
    RETURN_NAMES = ("MODEL", "CONDITIONING+", "CONDITIONING-", "LATENT", "VAE", "IMAGE", )
    FUNCTION = "sample"
    OUTPUT_NODE = True
    CATEGORY = "Efficiency Nodes/Sampling"

    def sample(self, sampler_state, my_unique_id, model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
               latent_image, preview_image, denoise=1.0, prompt=None, extra_pnginfo=None, optional_vae=(None,)):

        empty_image = pil2tensor(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))
        vae = optional_vae

        # Preview check
        preview = True
        if vae == (None,) or preview_image == "Disabled":
            preview = False
            last_helds["results"][my_unique_id]  = None
            last_helds["images"][my_unique_id] = None
            if vae == (None,):
                print('\033[32mKSampler(Efficient)[{}]:\033[0m No vae input detected, preview image disabled'.format(my_unique_id))

        # Init last_results
        if last_helds["results"][my_unique_id] == None:
            last_results = list()
        else:
            last_results = last_helds["results"][my_unique_id]

        # Init last_latent
        if last_helds["latent"][my_unique_id] == None:
            last_latent = latent_image
        else:
            last_latent = {"samples": None}
            last_latent["samples"] = last_helds["latent"][my_unique_id]

        # Init last_images
        if last_helds["images"][my_unique_id] == None:
            last_images = empty_image
        else:
            last_images = last_helds["images"][my_unique_id]

        if sampler_state == "Sample":
            samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
            latent = samples[0]["samples"]
            last_helds["latent"][my_unique_id] =  latent
            if preview == False:
                return {"ui": {"images": list()}, "result": (model, positive, negative, {"samples": latent}, vae, empty_image,)}

        # Adjust for KSampler states
        elif sampler_state == "Hold":
            print('\033[32mKSampler(Efficient)[{}] outputs on hold\033[0m'.format(my_unique_id))
            if preview == False:
                return {"ui": {"images": last_results}, "result": (model, positive, negative, last_latent, vae, last_images,)}
            else:
                latent = last_latent["samples"]

        images = vae.decode(latent).cpu()
        last_helds["images"][my_unique_id] = images

        filename_prefix = "TSC_KS_{:02d}".format(my_unique_id)

        def map_filename(filename):
            prefix_len = len(os.path.basename(filename_prefix))
            prefix = filename[:prefix_len + 1]
            try:
                digits = int(filename[prefix_len + 1:].split('_')[0])
            except:
                digits = 0
            return (digits, prefix)

        def compute_vars(input):
            input = input.replace("%width%", str(images[0].shape[1]))
            input = input.replace("%height%", str(images[0].shape[0]))
            return input

        filename_prefix = compute_vars(filename_prefix)

        subfolder = os.path.dirname(os.path.normpath(filename_prefix))
        filename = os.path.basename(os.path.normpath(filename_prefix))

        full_output_folder = os.path.join(self.output_dir, subfolder)

        try:
            counter = max(filter(lambda a: a[1][:-1] == filename and a[1][-1] == "_",
                                 map(map_filename, os.listdir(full_output_folder))))[0] + 1
        except ValueError:
            counter = 1
        except FileNotFoundError:
            os.makedirs(full_output_folder, exist_ok=True)
            counter = 1

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        results = list()
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))
            file = f"{filename}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            });
            counter += 1
        last_helds["results"][my_unique_id] = results

        #if sampler_state == "Sample":
        # Output results to ui and node outputs
        return {"ui": {"images": results}, "result": (model, positive, negative, {"samples":latent}, vae, images, )}
        #if sampler_state == "Hold":
        #    return {"ui": {"images": last_results}, "result": (model, positive, negative, last_latent, vae, last_images,)}


# TSC Image Overlay
class TSC_ImageOverlay:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_image": ("IMAGE",),
                "overlay_image": ("IMAGE",),
                "overlay_resize": (["None", "Fit", "Resize by rescale_factor", "Resize to width & heigth"],),
                "resize_method": (["nearest-exact", "bilinear", "area"],),
                "rescale_factor": ("FLOAT", {"default": 1, "min": 0.01, "max": 16.0, "step": 0.1}),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 64}),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 64}),
                "x_offset": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 10}),
                "y_offset": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 10}),
                "rotation": ("INT", {"default": 0, "min": -180, "max": 180, "step": 5}),
                "opacity": ("FLOAT", {"default": 0, "min": 0, "max": 100, "step": 5}),
            },
            "optional": {"optional_mask": ("MASK",),}
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "overlayimage"
    CATEGORY = "Efficiency Nodes/Image"

    def overlayimage(self, base_image, overlay_image, overlay_resize, resize_method, rescale_factor, width, height, x_offset, y_offset, rotation, opacity, optional_mask=None):
        result = self.apply_overlay(tensor2pil(base_image), overlay_image, overlay_resize, resize_method, rescale_factor, (int(width), int(height)),
                                   (int(x_offset), int(y_offset)), int(rotation), opacity, optional_mask)
        return (pil2tensor(result),)

    def apply_overlay(self, base, overlay, size_option, resize_method, rescale_factor, size, location, rotation, opacity, mask):

        # Check for different sizing options
        if size_option != "None":
            #Extract overlay size and store in Tuple "overlay_size" (WxH)
            overlay_size = overlay.size()
            overlay_size = (overlay_size[2], overlay_size[1])
            if size_option == "Fit":
                overlay_size = (base.size[0],base.size[1])
            elif size_option == "Resize by rescale_factor":
                overlay_size = tuple(int(dimension * rescale_factor) for dimension in overlay_size)
            elif size_option == "Resize to width & heigth":
                overlay_size = (size[0], size[1])

            samples = overlay.movedim(-1, 1)
            overlay = comfy.utils.common_upscale(samples, overlay_size[0], overlay_size[1], resize_method, False)
            overlay = overlay.movedim(1, -1)
            
        overlay = tensor2pil(overlay)

         # Add Alpha channel to overlay
        overlay = overlay.convert('RGBA')
        overlay.putalpha(Image.new("L", overlay.size, 255))

        # If mask connected, check if the overlay image has an alpha channel
        if mask is not None:
            # Convert mask to pil and resize
            mask = tensor2pil(mask)
            mask = mask.resize(overlay.size)
            # Apply mask as overlay's alpha
            overlay.putalpha(ImageOps.invert(mask))

        # Rotate the overlay image
        overlay = overlay.rotate(rotation, expand=True)

        # Apply opacity on overlay image
        r, g, b, a = overlay.split()
        a = a.point(lambda x: max(0, int(x * (1 - opacity / 100))))
        overlay.putalpha(a)

        # Paste the overlay image onto the base image
        if mask is None:
            base.paste(overlay, location)
        else:
            base.paste(overlay, location, overlay)

        # Return the edited base image
        return base


# TSC Evaluate Integers
class TSC_EvaluateInts:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "python_expression": ("STRING", {"default": "((a + b) - c) / 2", "multiline": False}),
                    "print_to_console": (["False", "True"],),},
                "optional": {
                    "a": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 1}),
                    "b": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 1}),
                    "c": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 1}),}
                }
    RETURN_TYPES = ("INT", "FLOAT",)
    OUTPUT_NODE = True
    FUNCTION = "evaluate"
    CATEGORY = "Efficiency Nodes/Math"

    def evaluate(self, python_expression, print_to_console, a=0, b=0, c=0):
        int_result = int(eval(python_expression))
        float_result = float(eval(python_expression))
        if print_to_console=="True":
            print("\n\033[31mEvaluate Integers Debug:\033[0m")
            print(f"\033[90m{{a = {a} , b = {b} , c = {c}}} \033[0m")
            print(f"{python_expression} = \033[92m INT: " + str(int_result) + " , FLOAT: " + str(float_result) + "\033[0m")
        return (int_result, float_result,)


# TSC Evaluate Strings
class TSC_EvaluateStrs:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "python_expression": ("STRING", {"default": "a + b + c", "multiline": False}),
                    "print_to_console": (["False", "True"],)},
                "optional": {
                    "a": ("STRING", {"default": "Hello", "multiline": False}),
                    "b": ("STRING", {"default": " World", "multiline": False}),
                    "c": ("STRING", {"default": "!", "multiline": False}),}
                }
    RETURN_TYPES = ("STRING",)
    OUTPUT_NODE = True
    FUNCTION = "evaluate"
    CATEGORY = "Efficiency Nodes/Math"

    def evaluate(self, python_expression, print_to_console, a="", b="", c=""):
        result = str(eval(python_expression))
        if print_to_console=="True":
            print("\n\033[31mEvaluate Strings Debug:\033[0m")
            print(f"\033[90ma = {a} \nb = {b} \nc = {c}\033[0m")
            print(f"{python_expression} = \033[92m" + result + "\033[0m")
        return (result,)


# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "KSampler (Efficient)": TSC_KSampler,
    "Efficient Loader": TSC_EfficientLoader,
    "Image Overlay": TSC_ImageOverlay,
    "Evaluate Integers": TSC_EvaluateInts,
    "Evaluate Strings": TSC_EvaluateStrs,
}