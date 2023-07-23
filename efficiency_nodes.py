# Efficiency Nodes - A collection of my ComfyUI custom nodes to help streamline workflows and reduce total node count.
#  by Luciano Cirino (Discord: TSC#9184) - April 2023

from comfy.sd import ModelPatcher, CLIP, VAE
from nodes import common_ksampler, CLIPSetLastLayer

from torch import Tensor
from PIL import Image, ImageOps, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
import numpy as np
import torch

import ast
from pathlib import Path
import os
import sys
import subprocess
import json
import folder_paths
import psutil

# Get the absolute path of the parent directory of the current script
my_dir = os.path.dirname(os.path.abspath(__file__))

# Add the My directory path to the sys.path list
sys.path.append(my_dir)

# Construct the absolute path to the ComfyUI directory
comfy_dir = os.path.abspath(os.path.join(my_dir, '..', '..'))

# Add the ComfyUI directory path to the sys.path list
sys.path.append(comfy_dir)

# Construct the path to the font file
font_path = os.path.join(my_dir, 'arial.ttf')

# Import functions from ComfyUI
import comfy.samplers
import comfy.sd
import comfy.utils

# Import my util functions
from tsc_utils import *

MAX_RESOLUTION=8192

########################################################################################################################
# TSC Efficient Loader
class TSC_EfficientLoader:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"),),
                              "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                              "clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
                              "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                              "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "lora_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              "positive": ("STRING", {"default": "Positive","multiline": True}),
                              "negative": ("STRING", {"default": "Negative", "multiline": True}),
                              "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                              "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})},
                "optional": {"lora_stack": ("LORA_STACK", )},
                "hidden": { "prompt": "PROMPT",
                            "my_unique_id": "UNIQUE_ID",},
                }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "DEPENDENCIES",)
    RETURN_NAMES = ("MODEL", "CONDITIONING+", "CONDITIONING-", "LATENT", "VAE", "CLIP", "DEPENDENCIES", )
    FUNCTION = "efficientloader"
    CATEGORY = "Efficiency Nodes/Loaders"

    def efficientloader(self, ckpt_name, vae_name, clip_skip, lora_name, lora_model_strength, lora_clip_strength,
                        positive, negative, empty_latent_width, empty_latent_height, batch_size, lora_stack=None,
                        prompt=None, my_unique_id=None):

        model: ModelPatcher | None = None
        clip: CLIP | None = None
        vae: VAE | None = None

        # Create Empty Latent
        latent = torch.zeros([batch_size, 4, empty_latent_height // 8, empty_latent_width // 8]).cpu()

        # Clean globally stored objects
        globals_cleanup(prompt)

        # Retrieve cache numbers
        vae_cache, ckpt_cache, lora_cache = get_cache_numbers("Efficient Loader")

        if lora_name != "None":
            lora_params = [(lora_name, lora_model_strength, lora_clip_strength)]
            if lora_stack is not None:
                lora_params.extend(lora_stack)
            model, clip = load_lora(lora_params, ckpt_name, my_unique_id, cache=lora_cache, ckpt_cache=ckpt_cache, cache_overwrite=True)
            if vae_name == "Baked VAE":
                vae = get_bvae_by_ckpt_name(ckpt_name)
        else:
            model, clip, vae = load_checkpoint(ckpt_name, my_unique_id, cache=ckpt_cache, cache_overwrite=True)
            lora_params = None

        # Check for custom VAE
        if vae_name != "Baked VAE":
            vae = load_vae(vae_name, my_unique_id, cache=vae_cache, cache_overwrite=True)

        # Debugging
        ###print_loaded_objects_entries()

        # CLIP skip
        if not clip:
            raise Exception("No CLIP found")
        clip = clip.clone()
        clip.clip_layer(clip_skip)

        # Data for XY Plot
        dependencies = (vae_name, ckpt_name, clip, clip_skip, positive, negative, lora_params)

        return (model, [[clip.encode(positive), {}]], [[clip.encode(negative), {}]], {"samples":latent}, vae, clip, dependencies, )

########################################################################################################################
# TSC LoRA Stacker
class TSC_LoRA_Stacker:

    loras = ["None"] + folder_paths.get_filename_list("loras")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lora_name_1": (cls.loras,),
            "lora_wt_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_name_2": (cls.loras,),
            "lora_wt_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_name_3": (cls.loras,),
            "lora_wt_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})},
            "optional": {"lora_stack": ("LORA_STACK",)},
        }

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("LORA_STACK",)
    FUNCTION = "lora_stacker"
    CATEGORY = "Efficiency Nodes/Misc"

    def lora_stacker(self, lora_name_1, lora_wt_1, lora_name_2, lora_wt_2, lora_name_3, lora_wt_3, lora_stack=None):
        # Create a list of tuples using provided parameters, exclude tuples with lora_name as "None"
        loras = [(lora_name, lora_wt, lora_wt) for lora_name, lora_wt, lora_wt in
                 [(lora_name_1, lora_wt_1, lora_wt_1),
                  (lora_name_2, lora_wt_2, lora_wt_2),
                  (lora_name_3, lora_wt_3, lora_wt_3)]
                 if lora_name != "None"]

        # If lora_stack is not None, extend the loras list with lora_stack
        if lora_stack is not None:
            loras.extend([l for l in lora_stack if l[0] != "None"])

        return (loras,)

# TSC LoRA Stacker Advanced
class TSC_LoRA_Stacker_Adv:

    loras = ["None"] + folder_paths.get_filename_list("loras")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lora_name_1": (cls.loras,),
            "model_str_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "clip_str_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_name_2": (cls.loras,),
            "model_str_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "clip_str_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_name_3": (cls.loras,),
            "model_str_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "clip_str_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})},
            "optional": {"lora_stack": ("LORA_STACK",)},
        }

    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("LORA_STACK",)
    FUNCTION = "lora_stacker"
    CATEGORY = "Efficiency Nodes/Misc"

    def lora_stacker(self, lora_name_1, model_str_1, clip_str_1, lora_name_2, model_str_2, clip_str_2,
                     lora_name_3, model_str_3, clip_str_3, lora_stack=None):
        # Create a list of tuples using provided parameters, exclude tuples with lora_name as "None"
        loras = [(lora_name, model_str, clip_str) for lora_name, model_str, clip_str in
                 [(lora_name_1, model_str_1, clip_str_1),
                  (lora_name_2, model_str_2, clip_str_2),
                  (lora_name_3, model_str_3, clip_str_3)]
                 if lora_name != "None"]

        # If lora_stack is not None, extend the loras list with lora_stack
        if lora_stack is not None:
            loras.extend([l for l in lora_stack if l[0] != "None"])

        return (loras,)

########################################################################################################################
# TSC KSampler (Efficient)
class TSC_KSampler:
    
    empty_image = pil2tensor(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))

    def __init__(self):
        self.output_dir = os.path.join(comfy_dir, 'temp')
        self.type = "temp"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"sampler_state": (["Sample", "Hold", "Script"], ),
                     "model": ("MODEL",),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     "latent_image": ("LATENT",),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "preview_image": (["Disabled", "Enabled", "Output Only"],),
                     },
                "optional": { "optional_vae": ("VAE",),
                              "script": ("SCRIPT",),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",},
                }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "IMAGE", )
    RETURN_NAMES = ("MODEL", "CONDITIONING+", "CONDITIONING-", "LATENT", "VAE", "IMAGE", )
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "Efficiency Nodes/Sampling"
    
    def sample(self, sampler_state, model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
               latent_image, preview_image, denoise=1.0, prompt=None, extra_pnginfo=None, my_unique_id=None,
               optional_vae=(None,), script=None):

        # Extract node_settings from json
        def get_settings():
            # Get the directory path of the current file
            my_dir = os.path.dirname(os.path.abspath(__file__))
            # Construct the file path for node_settings.json
            settings_file = os.path.join(my_dir, 'node_settings.json')
            # Load the settings from the JSON file
            with open(settings_file, 'r') as file:
                node_settings = json.load(file)
            # Retrieve the settings
            kse_vae_tiled = node_settings.get("KSampler (Efficient)", {}).get('vae_tiled', False)
            xy_vae_tiled = node_settings.get("XY Plot", {}).get('vae_tiled', False)
            return kse_vae_tiled, xy_vae_tiled

        kse_vae_tiled, xy_vae_tiled = get_settings()

        # Functions for previewing images in Ksampler
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

        def preview_images(images, filename_prefix):
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
            return results

        def get_value_by_id(key: str, my_unique_id):
            global last_helds
            for value, id_ in last_helds[key]:
                if id_ == my_unique_id:
                    return value
            return None

        def update_value_by_id(key: str, my_unique_id, new_value):
            global last_helds

            for i, (value, id_) in enumerate(last_helds[key]):
                if id_ == my_unique_id:
                    last_helds[key][i] = (new_value, id_)
                    return True

            last_helds[key].append((new_value, my_unique_id))
            return True

        # Clean globally stored objects of non-existant nodes
        globals_cleanup(prompt)

        # Convert ID string to an integer
        my_unique_id = int(my_unique_id)

        # Vae input check
        vae = optional_vae
        if vae == (None,):
            print('\033[33mKSampler(Efficient) Warning:\033[0m No vae input detected, preview and output image disabled.\n')
            preview_image = "Disabled"

        # Init last_results
        if get_value_by_id("results", my_unique_id) is None:
            last_results = list()
        else:
            last_results = get_value_by_id("results", my_unique_id)

        # Init last_latent
        if get_value_by_id("latent", my_unique_id) is None:
            last_latent = latent_image
        else:
            last_latent = {"samples": None}
            last_latent["samples"] = get_value_by_id("latent", my_unique_id)

        # Init last_images
        if get_value_by_id("images", my_unique_id) == None:
            last_images = TSC_KSampler.empty_image
        else:
            last_images = get_value_by_id("images", my_unique_id)

        # Initialize latent
        latent: Tensor|None = None

        # Define filename_prefix
        filename_prefix = "KSeff_{:02d}".format(my_unique_id)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Check the current sampler state
        if sampler_state == "Sample":

            # Sample using the common KSampler function and store the samples
            samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                      latent_image, denoise=denoise)

            # Extract the latent samples from the returned samples dictionary
            latent = samples[0]["samples"]

            # Store the latent samples in the 'last_helds' dictionary with a unique ID
            update_value_by_id("latent", my_unique_id, latent)

            # If not in preview mode, return the results in the specified format
            if preview_image == "Disabled":
                # Enable vae decode on next Hold
                update_value_by_id("vae_decode", my_unique_id, True)
                return {"ui": {"images": list()},
                        "result": (model, positive, negative, {"samples": latent}, vae, TSC_KSampler.empty_image,)}
            else:
                # Decode images and store
                if kse_vae_tiled == False:
                    images = vae.decode(latent).cpu()
                else:
                    images = vae.decode_tiled(latent).cpu()
                update_value_by_id("images", my_unique_id, images)

                # Disable vae decode on next Hold
                update_value_by_id("vae_decode", my_unique_id, False)

                # Generate image results and store
                results = preview_images(images, filename_prefix)
                update_value_by_id("results", my_unique_id, results)

                # Determine what the 'images' value should be
                images_value = list() if preview_image == "Output Only" else results

                # Output image results to ui and node outputs
                return {"ui": {"images": images_value},
                        "result": (model, positive, negative, {"samples": latent}, vae, images,)}


        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # If the sampler state is "Hold"
        elif sampler_state == "Hold":

            # If not in preview mode, return the results in the specified format
            if preview_image == "Disabled":
                return {"ui": {"images": list()},
                        "result": (model, positive, negative, last_latent, vae, TSC_KSampler.empty_image,)}

            else:
                latent = last_latent["samples"]

                if get_value_by_id("vae_decode", my_unique_id) == True:

                    # Decode images and store
                    if kse_vae_tiled == False:
                        images = vae.decode(latent).cpu()
                    else:
                        images = vae.decode_tiled(latent).cpu()
                    update_value_by_id("images", my_unique_id, images)

                    # Disable vae decode on next Hold
                    update_value_by_id("vae_decode", my_unique_id, False)

                    # Generate image results and store
                    results = preview_images(images, filename_prefix)
                    update_value_by_id("results", my_unique_id, results)

                else:
                    images = last_images
                    results = last_results

                # Determine what the 'images' value should be
                images_value = list() if preview_image == "Output Only" else results

                # Output image results to ui and node outputs
                return {"ui": {"images": images_value},
                        "result": (model, positive, negative, {"samples": latent}, vae, images,)}

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        elif sampler_state == "Script":

            # Store name of connected node to script input
            script_node_name, script_node_id = extract_node_info(prompt, my_unique_id, 'script')

            # If no valid script input connected, error out
            if script == None or script == (None,) or script_node_name!="XY Plot":
                if script_node_name!="XY Plot":
                    print('\033[31mKSampler(Efficient) Error:\033[0m No valid script input detected')
                return {"ui": {"images": list()},
                        "result": (model, positive, negative, last_latent, vae, last_images,)}

            # If no vae connected, throw errors
            if vae == (None,):
                print('\033[31mKSampler(Efficient) Error:\033[0m VAE must be connected to use Script mode.')
                return {"ui": {"images": list()},
                        "result": (model, positive, negative, last_latent, vae, last_images,)}

            # If preview_image set to disabled, run script anyways with message
            if preview_image == "Disabled":
                print('\033[33mKSampler(Efficient) Warning:\033[0m The preview image cannot be disabled when running'
                      ' the XY Plot script, proceeding as if it was enabled.\n')

            # Extract the 'samples' tensor and split it into individual image tensors
            image_tensors = torch.split(latent_image['samples'], 1, dim=0)

            # Get the shape of the first image tensor
            shape = image_tensors[0].shape

            # Extract the original height and width
            latent_height, latent_width = shape[2] * 8, shape[3] * 8

            # Set latent only to the first latent of batch
            latent_image = {'samples': image_tensors[0]}

            #___________________________________________________________________________________________________________
            # Initialize, unpack, and clean variables for the XY Plot script
            if script_node_name == "XY Plot":

                # Initialize variables
                vae_name = None
                ckpt_name = None
                clip = None
                lora_params = None
                positive_prompt = None
                negative_prompt = None
                clip_skip = None
                merge_ckpt_name = None
                merge_ckpt_weight = None
                merge_clip_weight = None

                # Unpack script Tuple (X_type, X_value, Y_type, Y_value, grid_spacing, Y_label_orientation, dependencies)
                X_type, X_value, Y_type, Y_value, grid_spacing, Y_label_orientation, cache_models, xyplot_as_output_image,\
                    flip_xy, dependencies = script

                # Unpack Effficient Loader dependencies
                if dependencies is not None:
                    vae_name, ckpt_name, clip, clip_skip, positive_prompt, negative_prompt, lora_params = dependencies

                # Helper function to process printout values
                def process_xy_for_print(value, replacement, type_):
                    if isinstance(value, tuple) and type_ == "Scheduler":
                        return value[0]  # Return only the first entry of the tuple
                    elif isinstance(value, tuple):
                        return tuple(replacement if v is None else v for v in value)
                    else:
                        return replacement if value is None else value

                # Determine the replacements based on X_type and Y_type
                replacement_X = scheduler if X_type == 'Sampler' else clip_skip if X_type == 'Checkpoint' else None
                replacement_Y = scheduler if Y_type == 'Sampler' else clip_skip if Y_type == 'Checkpoint' else None

                # Process X_value and Y_value
                X_value_processed = [process_xy_for_print(v, replacement_X, X_type) for v in X_value]
                Y_value_processed = [process_xy_for_print(v, replacement_Y, Y_type) for v in Y_value]

                # Print XY Plot Inputs
                print("-" * 40)
                print("XY Plot Script Inputs:")
                print(f"(X) {X_type}: {X_value_processed}")
                print(f"(Y) {Y_type}: {Y_value_processed}")
                print("-" * 40)

                # If not caching models, set to 1.
                if cache_models == "False":
                    vae_cache = ckpt_cache = lora_cache = 1
                else:
                    # Retrieve cache numbers
                    vae_cache, ckpt_cache, lora_cache = get_cache_numbers("XY Plot")
                # Pack cache numbers in a tuple
                cache = (vae_cache, ckpt_cache, lora_cache)

                # Embedd original prompts into prompt variables
                positive_prompt = (positive_prompt, positive_prompt)
                negative_prompt = (negative_prompt, negative_prompt)

                #_______________________________________________________________________________________________________
                #The below code will clean from the cache any ckpt/vae/lora models it will not be reusing.

                # Map the type names to the dictionaries
                dict_map = {"VAE": [], "Checkpoint": [], "LoRA": []}

                # Create a list of tuples with types and values
                type_value_pairs = [(X_type, X_value), (Y_type, Y_value)]

                # Iterate over type-value pairs
                for t, v in type_value_pairs:
                    if t in dict_map:
                        # Flatten the list of lists of tuples if the type is "LoRA"
                        if t == "LoRA":
                            dict_map[t] = [item for sublist in v for item in sublist]
                        else:
                            dict_map[t] = v

                ckpt_dict = [t[0] for t in dict_map.get("Checkpoint", [])] if dict_map.get("Checkpoint", []) else []

                lora_dict = [[t,] for t in dict_map.get("LoRA", [])] if dict_map.get("LoRA", []) else []

                # If both ckpt_dict and lora_dict are not empty, manipulate lora_dict as described
                if ckpt_dict and lora_dict:
                    lora_dict = [(lora_params, ckpt) for ckpt in ckpt_dict for lora_params in lora_dict]
                # If lora_dict is not empty and ckpt_dict is empty, insert ckpt_name into each tuple in lora_dict
                elif lora_dict:
                    lora_dict = [(lora_params, ckpt_name) for lora_params in lora_dict]

                vae_dict = dict_map.get("VAE", [])

                # prioritize Caching Checkpoints over LoRAs but not both.
                if X_type == "LoRA":
                    ckpt_dict = []
                if X_type == "Checkpoint":
                    lora_dict = []

                # Print dict_arrays for debugging
                ###print(f"vae_dict={vae_dict}\nckpt_dict={ckpt_dict}\nlora_dict={lora_dict}")

                # Clean values that won't be reused
                clear_cache_by_exception(script_node_id, vae_dict=vae_dict, ckpt_dict=ckpt_dict, lora_dict=lora_dict)

                # Print loaded_objects for debugging
                ###print_loaded_objects_entries()

                #_______________________________________________________________________________________________________
                # Function that changes appropiate variables for next processed generations (also generates XY_labels)
                def define_variable(var_type, var, seed, steps, cfg, sampler_name, scheduler, denoise, vae_name, ckpt_name,
                                    clip_skip, positive_prompt, negative_prompt, lora_params, var_label, num_label, merge_ckpt_name, merge_ckpt_weight, merge_clip_weight):

                    # Define default max label size limit
                    max_label_len = 36
                    text = ""

                    # If var_type is "Seeds++ Batch", update var and seed, and generate labels
                    if var_type == "Seeds++ Batch":
                        text = f"Seed: {seed}"

                    # If var_type is "Steps", update steps and generate labels
                    elif var_type == "Steps":
                        steps = var
                        text = f"steps: {steps}"

                    # If var_type is "CFG Scale", update cfg and generate labels
                    elif var_type == "CFG Scale":
                        cfg = var
                        text = f"CFG: {round(cfg,2)}"

                    # If var_type is "Sampler", update sampler_name, scheduler, and generate labels
                    elif var_type == "Sampler":
                        sampler_name = var[0]
                        if var[1] == "":
                            text = f"{sampler_name}"
                        else:
                            if var[1] != None:
                                scheduler = (var[1], scheduler[1])
                            else:
                                scheduler = (scheduler[1], scheduler[1])
                            text = f"{sampler_name} ({scheduler[0]})"
                        text = text.replace("ancestral", "a").replace("uniform", "u").replace("exponential","exp")

                    # If var_type is "Scheduler", update scheduler and generate labels
                    elif var_type == "Scheduler":
                        if len(var) == 2:
                            scheduler = (var[0], scheduler[1])
                            text = f"{sampler_name} ({scheduler[0]})"
                        else:
                            scheduler = (var, scheduler[1])
                            text = f"{scheduler[0]}"
                        text = text.replace("ancestral", "a").replace("uniform", "u").replace("exponential","exp")

                    # If var_type is "Denoise", update denoise and generate labels
                    elif var_type == "Denoise":
                        denoise = var
                        text = f"denoise: {round(denoise, 2)}"

                    # If var_type is "VAE", update vae_name and generate labels
                    elif var_type == "VAE":
                        vae_name = var
                        vae_filename = os.path.splitext(os.path.basename(vae_name))[0]
                        text = f"VAE: {vae_filename}"

                    # If var_type is "Positive Prompt S/R", update positive_prompt and generate labels
                    elif var_type == "Positive Prompt S/R":
                        search_txt, replace_txt = var
                        if replace_txt != None:
                            positive_prompt = (positive_prompt[1].replace(search_txt, replace_txt, 1), positive_prompt[1])
                        else:
                            positive_prompt = (positive_prompt[1], positive_prompt[1])
                            replace_txt = search_txt
                        text = f"{replace_txt}"

                    # If var_type is "Negative Prompt S/R", update negative_prompt and generate labels
                    elif var_type == "Negative Prompt S/R":
                        search_txt, replace_txt = var
                        if replace_txt:
                            negative_prompt = (negative_prompt[1].replace(search_txt, replace_txt, 1), negative_prompt[1])
                        else:
                            negative_prompt = (negative_prompt[1], negative_prompt[1])
                            replace_txt = search_txt
                        text = f"(-) {replace_txt}"

                    # If var_type is "Checkpoint", update model and clip (if needed) and generate labels
                    elif var_type == "Checkpoint":
                        ckpt_name = var[0]
                        if var[1] == None:
                            clip_skip = (clip_skip[1],clip_skip[1])
                        else:
                            clip_skip = (var[1],clip_skip[1])
                        ckpt_filename = os.path.splitext(os.path.basename(ckpt_name))[0]
                        text = f"{ckpt_filename}"

                    elif var_type == "Checkpoint Merge" or var_type == "Checkpoint Merge Adv":
                        merge_ckpt_name = var[0]
                        merge_ckpt_weight = var[1] if var[1] is not None else 0.5
                        merge_clip_weight = var[2] if var[2] is not None else 0.5
                        merge_ckpt_filename = os.path.splitext(os.path.basename(merge_ckpt_name))[0]
                        
                        if merge_ckpt_weight != merge_clip_weight:
                            text = f"{merge_ckpt_filename} ({merge_ckpt_weight:.2f}|{merge_clip_weight:.2f})"
                        else:
                            text = f"{merge_ckpt_filename} ({merge_ckpt_weight:.2f})"

                    elif var_type == "Clip Skip":
                        clip_skip = (var, clip_skip[1])
                        text = f"Clip Skip ({clip_skip[0]})"

                    elif var_type == "LoRA":
                        lora_params = var
                        max_label_len = 30 + (12 * (len(lora_params)-1))
                        if len(lora_params) == 1:
                            lora_name, lora_model_wt, lora_clip_wt = lora_params[0]
                            lora_filename = os.path.splitext(os.path.basename(lora_name))[0]
                            lora_model_wt = format(float(lora_model_wt), ".2f").rstrip('0').rstrip('.')
                            lora_clip_wt = format(float(lora_clip_wt), ".2f").rstrip('0').rstrip('.')
                            lora_filename = lora_filename[:max_label_len - len(f"LoRA: ({lora_model_wt})")]
                            if lora_model_wt == lora_clip_wt:
                                text = f"LoRA: {lora_filename}({lora_model_wt})"
                            else:
                                text = f"LoRA: {lora_filename}({lora_model_wt},{lora_clip_wt})"
                        elif len(lora_params) > 1:
                            lora_filenames = [os.path.splitext(os.path.basename(lora_name))[0] for lora_name, _, _ in lora_params]
                            lora_details = [(format(float(lora_model_wt), ".2f").rstrip('0').rstrip('.'),
                                             format(float(lora_clip_wt), ".2f").rstrip('0').rstrip('.')) for _, lora_model_wt, lora_clip_wt in lora_params]
                            non_name_length = sum(len(f"({lora_details[i][0]},{lora_details[i][1]})") + 2 for i in range(len(lora_params)))
                            available_space = max_label_len - non_name_length
                            max_name_length = available_space // len(lora_params)
                            lora_filenames = [filename[:max_name_length] for filename in lora_filenames]
                            text_elements = [f"{lora_filename}({lora_details[i][0]})" if lora_details[i][0] == lora_details[i][1] else f"{lora_filename}({lora_details[i][0]},{lora_details[i][1]})" for i, lora_filename in enumerate(lora_filenames)]
                            text = " ".join(text_elements)

                    def truncate_texts(texts, num_label, max_label_len):
                        truncate_length = max(min(max(len(text) for text in texts), max_label_len), 24)

                        return [text if len(text) <= truncate_length else text[:truncate_length] + "..." for text in
                                texts]

                    # Add the generated text to var_label if it's not full
                    if len(var_label) < num_label:
                        var_label.append(text)

                    # If var_type VAE , truncate entries in the var_label list when it's full
                    if len(var_label) == num_label and (var_type == "VAE" or var_type == "Checkpoint" or var_type == "LoRA"):
                        var_label = truncate_texts(var_label, num_label, max_label_len)

                    # Return the modified variables
                    return steps, cfg, sampler_name, scheduler, denoise, vae_name, ckpt_name, clip_skip, \
                        positive_prompt, negative_prompt, lora_params, var_label, merge_ckpt_name, merge_ckpt_weight, merge_clip_weight

                # _______________________________________________________________________________________________________
                # The function below is used to smartly load Checkpoint/LoRA/VAE models between generations.
                def define_model(model, clip, positive, negative, positive_prompt, negative_prompt, clip_skip, vae,
                                 vae_name, ckpt_name, lora_params, index, types, script_node_id, cache, merge_ckpt_name, merge_ckpt_weight, merge_clip_weight):
        
                    # Encode prompt and apply clip_skip. Return new conditioning.
                    def encode_prompt(positive_prompt, negative_prompt, clip, clip_skip):
                        clip = CLIPSetLastLayer().set_last_layer(clip, clip_skip)[0]
                        return [[clip.encode(positive_prompt), {}]], [[clip.encode(negative_prompt), {}]]

                    # Variable to track wether to encode prompt or not
                    encode = False

                    # Unpack types tuple
                    X_type, Y_type = types

                    # Note: Index is held at 0 when Y_type == "Nothing"

                    # Load VAE if required
                    if (X_type == "VAE" and index == 0) or Y_type == "VAE":
                        vae = load_vae(vae_name, script_node_id, cache=cache[0])

                    # Load Checkpoint if required. If Y_type is LoRA, required models will be loaded by load_lora func.
                    if (X_type == "Checkpoint" and index == 0 and Y_type != "LoRA"):
                        if lora_params is None:
                            model, clip, _ = load_checkpoint(ckpt_name, script_node_id, output_vae=False, cache=cache[1])
                        else: # Load Efficient Loader LoRA
                            model, clip = load_lora(lora_params, ckpt_name, script_node_id,
                                                    cache=None, ckpt_cache=cache[1])
                        encode = True

                    # Load LoRA if required
                    elif (X_type == "LoRA" and index == 0):
                        # Don't cache Checkpoints
                        model, clip = load_lora(lora_params, ckpt_name, script_node_id, cache=cache[2])
                        encode = True
                        
                    elif Y_type == "LoRA":  # X_type must be Checkpoint, so cache those as defined
                        model, clip = load_lora(lora_params, ckpt_name, script_node_id,
                                                cache=None, ckpt_cache=cache[1])
                        encode = True

                    if (X_type == "Checkpoint Merge" or Y_type == "Checkpoint Merge"):
                        # merge the model
                        if lora_params is None:
                            model2, clip2, _ = load_checkpoint(merge_ckpt_name, script_node_id, output_vae=False, cache=cache[1])
                        else: # Load Efficient Loader LoRA
                            model2, clip2 = load_lora(lora_params, merge_ckpt_name, script_node_id,
                                                              cache=cache[2], ckpt_cache=cache[1])
                        model1 = None
                        clip1 = None
                        if encode:
                            model1 = model.clone()
                            if clip is not None:
                                clip1 = clip.clone()
                        else:
                            # load the model again so that we do not modify the previously modified one
                            if lora_params is None:
                                tmp_model, tmp_clip, _ = load_checkpoint(ckpt_name, script_node_id, output_vae=False, cache=cache[1])
                            else: # Load Efficient Loader LoRA
                                tmp_model, tmp_clip = load_lora(lora_params, ckpt_name, script_node_id,
                                                        cache=cache[2], ckpt_cache=cache[1])
                            model1 = tmp_model.clone()
                            if tmp_clip is not None:
                                clip1 = tmp_clip.clone()
                            
                        # merge model
                        kp = model2.get_key_patches("diffusion_model.")
                        for k in kp:
                            model1.add_patches({k: kp[k]}, merge_ckpt_weight, 1.0 - merge_ckpt_weight)
                            
                        # merge clip
                        if clip2 is not None and clip1 is not None:
                            kp = clip2.get_key_patches()
                            for k in kp:
                                if k.endswith(".position_ids") or k.endswith(".logit_scale"):
                                    continue
                                clip1.add_patches({k: kp[k]}, merge_clip_weight, 1-merge_clip_weight)
                            clip = clip1
                            encode = True
                        
                        model = model1
                        

                    # Encode Prompt if required
                    prompt_types = ["Positive Prompt S/R", "Negative Prompt S/R", "Clip Skip"]
                    if (X_type in prompt_types and index == 0) or Y_type in prompt_types:
                        encode = True

                    # Encode prompt if needed
                    if encode == True:
                        positive, negative = encode_prompt(positive_prompt[0], negative_prompt[0], clip, clip_skip)
                        
                    return model, positive, negative, vae

                # ______________________________________________________________________________________________________
                # The below function is used to generate the results based on all the processed variables
                def process_values(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                   denoise, vae, latent_list=[], image_tensor_list=[], image_pil_list=[]):

                    # Sample
                    samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                              latent_image, denoise=denoise)

                    # Decode images and store
                    latent = samples[0]["samples"]

                    # Add the latent tensor to the tensors list
                    latent_list.append(latent)

                    # Decode the latent tensor
                    if xy_vae_tiled == False:
                        image = vae.decode(latent).cpu()
                    else:
                        image = vae.decode_tiled(latent).cpu()

                    # Add the resulting image tensor to image_tensor_list
                    image_tensor_list.append(image)

                    # Convert the image from tensor to PIL Image and add it to the image_pil_list
                    image_pil_list.append(tensor2pil(image))

                    # Return the touched variables
                    return latent_list, image_tensor_list, image_pil_list

                # ______________________________________________________________________________________________________
                # The below section is the heart of the XY Plot image generation

                 # Initiate Plot label text variables X/Y_label
                X_label = []
                Y_label = []

                # Seed_updated for "Seeds++ Batch" incremental seeds
                seed_updated = seed

                # Store the KSamplers original scheduler inside the same scheduler variable
                scheduler = (scheduler, scheduler)

                # Store the Eff Loaders original clip_skip inside the same clip_skip variable
                clip_skip = (clip_skip, clip_skip)

                # Store types in a Tuple for easy function passing
                types = (X_type, Y_type)

                # Fill Plot Rows (X)
                for X_index, X in enumerate(X_value):

                    # Seed control based on loop index during Batch
                    if X_type == "Seeds++ Batch":
                        # Update seed based on the inner loop index
                        seed_updated = seed + X_index

                    # Define X parameters and generate labels
                    steps, cfg, sampler_name, scheduler, denoise, vae_name, ckpt_name, clip_skip, positive_prompt, negative_prompt, \
                        lora_params, X_label, merge_ckpt_name, merge_ckpt_weight, merge_clip_weight = \
                        define_variable(X_type, X, seed_updated, steps, cfg, sampler_name, scheduler, denoise, vae_name, ckpt_name,
                                        clip_skip, positive_prompt, negative_prompt, lora_params, X_label, len(X_value), merge_ckpt_name, merge_ckpt_weight, merge_clip_weight)

                    if X_type != "Nothing" and Y_type == "Nothing":

                        # Models & Conditionings
                        model, positive, negative , vae = \
                            define_model(model, clip, positive, negative, positive_prompt, negative_prompt, clip_skip[0], vae,
                                         vae_name, ckpt_name, lora_params, 0, types, script_node_id, cache, merge_ckpt_name, merge_ckpt_weight, merge_clip_weight)

                        # Generate Results
                        latent_list, image_tensor_list, image_pil_list = \
                            process_values(model, seed_updated, steps, cfg, sampler_name, scheduler[0],
                                           positive, negative, latent_image, denoise, vae)

                    elif X_type != "Nothing" and Y_type != "Nothing":
                        # Seed control based on loop index during Batch
                        for Y_index, Y in enumerate(Y_value):

                            if Y_type == "Seeds++ Batch":
                                # Update seed based on the inner loop index
                                seed_updated = seed + Y_index

                            # Define Y parameters and generate labels
                            steps, cfg, sampler_name, scheduler, denoise, vae_name, ckpt_name, clip_skip, positive_prompt, negative_prompt, lora_params, Y_label, merge_ckpt_name, merge_ckpt_weight, merge_clip_weight = \
                                define_variable(Y_type, Y, seed_updated, steps, cfg, sampler_name, scheduler, denoise, vae_name, ckpt_name,
                                                clip_skip, positive_prompt, negative_prompt, lora_params, Y_label, len(Y_value), merge_ckpt_name, merge_ckpt_weight, merge_clip_weight)

                            # Models & Conditionings
                            model, positive, negative, vae = \
                                define_model(model, clip, positive, negative, positive_prompt, negative_prompt, clip_skip[0], vae,
                                         vae_name, ckpt_name, lora_params, Y_index, types, script_node_id, cache, merge_ckpt_name, merge_ckpt_weight, merge_clip_weight)

                            # Generate Results
                            latent_list, image_tensor_list, image_pil_list = \
                                process_values(model, seed_updated, steps, cfg, sampler_name, scheduler[0],
                                               positive, negative, latent_image, denoise, vae)

                # Clean up cache
                if cache_models == "False":
                    clear_cache_by_exception(script_node_id, vae_dict=[], ckpt_dict=[], lora_dict=[])
                #
                else:
                    # Prioritrize Caching Checkpoints over LoRAs.
                    if X_type == "LoRA":
                        clear_cache_by_exception(script_node_id, ckpt_dict=[])
                    elif X_type == "Checkpoint":
                        clear_cache_by_exception(script_node_id, lora_dict=[])

                # ______________________________________________________________________________________________________
                def print_plot_variables(X_type, Y_type, X_value, Y_value, seed, ckpt_name, lora_params,
                                         vae_name, clip_skip, steps, cfg, sampler_name, scheduler, denoise,
                                         num_rows, num_cols, latent_height, latent_width):

                    print("-" * 40)  # Print an empty line followed by a separator line
                    print("\033[32mXY Plot Results:\033[0m")

                    def get_vae_name(X_type, Y_type, X_value, Y_value, vae_name):
                        if X_type == "VAE":
                            vae_name = ", ".join(map(lambda x: os.path.splitext(os.path.basename(str(x)))[0], X_value))
                        elif Y_type == "VAE":
                            vae_name = ", ".join(map(lambda y: os.path.splitext(os.path.basename(str(y)))[0], Y_value))
                        else:
                            vae_name = os.path.splitext(os.path.basename(str(vae_name)))[0]
                        return vae_name
                    
                    def get_clip_skip(X_type, Y_type, X_value, Y_value, clip_skip):
                        if X_type == "Clip Skip":
                            clip_skip = ", ".join(map(str, X_value))
                        elif Y_type == "Clip Skip":
                            clip_skip = ", ".join(map(str, Y_value))
                        else:
                            clip_skip = clip_skip[1]
                        return clip_skip

                    def get_checkpoint_name(ckpt_type, ckpt_values, clip_skip_type, clip_skip_values, ckpt_name, clip_skip):
                        if ckpt_type == "Checkpoint":
                            if clip_skip_type == "Clip Skip":
                                ckpt_name = ", ".join([os.path.splitext(os.path.basename(str(ckpt[0])))[0] for ckpt in ckpt_values])
                            else:
                                ckpt_name = ", ".join([f"{os.path.splitext(os.path.basename(str(ckpt[0])))[0]}({str(ckpt[1]) if ckpt[1] is not None else str(clip_skip_values)})"
                                                          for ckpt in ckpt_values])
                                clip_skip = "_"
                        else:
                            ckpt_name = os.path.splitext(os.path.basename(str(ckpt_name)))[0]

                        return ckpt_name, clip_skip

                    def get_lora_name(X_type, Y_type, X_value, Y_value, lora_params=None):
                        if X_type != "LoRA" and Y_type != "LoRA":
                            if lora_params:
                                return f"[{', '.join([f'{os.path.splitext(os.path.basename(name))[0]}({round(model_wt, 3)},{round(clip_wt, 3)})' for name, model_wt, clip_wt in lora_params])}]"
                            else:
                                return None
                        else:
                            return get_lora_sublist_name(X_type,
                                                         X_value) if X_type == "LoRA" else get_lora_sublist_name(Y_type, Y_value) if Y_type == "LoRA" else None

                    def get_lora_sublist_name(lora_type, lora_value):
                        return ", ".join([
                                             f"[{', '.join([f'{os.path.splitext(os.path.basename(str(x[0])))[0]}({round(x[1], 3)},{round(x[2], 3)})' for x in sublist])}]"
                                             for sublist in lora_value])

                    # use these functions:
                    ckpt_type, clip_skip_type = (X_type, Y_type) if X_type in ["Checkpoint", "Clip Skip"] else (Y_type, X_type)
                    ckpt_values, clip_skip_values = (X_value, Y_value) if X_type in ["Checkpoint", "Clip Skip"] else (Y_value, X_value)

                    clip_skip = get_clip_skip(X_type, Y_type, X_value, Y_value, clip_skip)
                    ckpt_name, clip_skip = get_checkpoint_name(ckpt_type, ckpt_values, clip_skip_type, clip_skip_values, ckpt_name, clip_skip)
                    vae_name = get_vae_name(X_type, Y_type, X_value, Y_value, vae_name)
                    lora_name = get_lora_name(X_type, Y_type, X_value, Y_value, lora_params)

                    seed_list = [seed + x for x in X_value] if X_type == "Seeds++ Batch" else\
                        [seed + y for y in Y_value] if Y_type == "Seeds++ Batch" else [seed]
                    seed = ", ".join(map(str, seed_list))

                    steps = ", ".join(map(str, X_value)) if X_type == "Steps" else ", ".join(
                        map(str, Y_value)) if Y_type == "Steps" else steps

                    cfg = ", ".join(map(str, X_value)) if X_type == "CFG Scale" else ", ".join(
                        map(str, Y_value)) if Y_type == "CFG Scale" else cfg

                    if X_type == "Sampler":
                        if Y_type == "Scheduler":
                            sampler_name = ", ".join([f"{x[0]}" for x in X_value])
                            scheduler = ", ".join([f"{y}" for y in Y_value])
                        else:
                            sampler_name = ", ".join(
                                [f"{x[0]}({x[1] if x[1] != '' and x[1] is not None else scheduler[1]})" for x in X_value])
                            scheduler = "_"
                    elif Y_type == "Sampler":
                        if X_type == "Scheduler":
                            sampler_name = ", ".join([f"{y[0]}" for y in Y_value])
                            scheduler = ", ".join([f"{x}" for x in X_value])
                        else:
                            sampler_name = ", ".join(
                                [f"{y[0]}({y[1] if y[1] != '' and y[1] is not None else scheduler[1]})" for y in Y_value])
                            scheduler = "_"
                    else:
                        scheduler = ", ".join([str(x[0]) if isinstance(x, tuple) else str(x) for x in X_value]) if X_type == "Scheduler" else \
                            ", ".join([str(y[0]) if isinstance(y, tuple) else str(y) for y in Y_value]) if Y_type == "Scheduler" else scheduler[0]

                    denoise = ", ".join(map(str, X_value)) if X_type == "Denoise" else ", ".join(
                        map(str, Y_value)) if Y_type == "Denoise" else denoise

                    # Printouts
                    print(f"img_count: {len(X_value)*len(Y_value)}")
                    print(f"img_dims: {latent_height} x {latent_width}")
                    print(f"plot_dim: {num_cols} x {num_rows}")
                    if clip_skip == "_":
                        print(f"ckpt(clipskip): {ckpt_name if ckpt_name is not None else ''}")
                    else:
                        print(f"ckpt: {ckpt_name if ckpt_name is not None else ''}")
                        print(f"clip_skip: {clip_skip if clip_skip is not None else ''}")
                    if lora_name:
                        print(f"lora(mod,clip): {lora_name if lora_name is not None else ''}")
                    print(f"vae: {vae_name if vae_name is not None else ''}")
                    print(f"seed: {seed}")
                    print(f"steps: {steps}")
                    print(f"cfg: {cfg}")
                    if scheduler == "_":
                        print(f"sampler(schr): {sampler_name}")
                    else:
                        print(f"sampler: {sampler_name}")
                        print(f"scheduler: {scheduler}")
                    print(f"denoise: {denoise}")

                    if X_type == "Positive Prompt S/R" or Y_type == "Positive Prompt S/R":
                        positive_prompt = ", ".join([str(x[0]) if i == 0 else str(x[1]) for i, x in enumerate(
                            X_value)]) if X_type == "Positive Prompt S/R" else ", ".join(
                            [str(y[0]) if i == 0 else str(y[1]) for i, y in
                             enumerate(Y_value)]) if Y_type == "Positive Prompt S/R" else positive_prompt
                        print(f"+prompt_s/r: {positive_prompt}")

                    if X_type == "Negative Prompt S/R" or Y_type == "Negative Prompt S/R":
                        negative_prompt = ", ".join([str(x[0]) if i == 0 else str(x[1]) for i, x in enumerate(
                            X_value)]) if X_type == "Negative Prompt S/R" else ", ".join(
                            [str(y[0]) if i == 0 else str(y[1]) for i, y in
                             enumerate(Y_value)]) if Y_type == "Negative Prompt S/R" else negative_prompt
                        print(f"-prompt_s/r: {negative_prompt}")

                # ______________________________________________________________________________________________________
                def adjusted_font_size(text, initial_font_size, latent_width):
                    font = ImageFont.truetype(str(Path(font_path)), initial_font_size)
                    text_width = font.getlength(text)

                    if text_width > (latent_width * 0.9):
                        scaling_factor = 0.9  # A value less than 1 to shrink the font size more aggressively
                        new_font_size = int(initial_font_size * (latent_width / text_width) * scaling_factor)
                    else:
                        new_font_size = initial_font_size

                    return new_font_size

                # ______________________________________________________________________________________________________
                
                # Disable vae decode on next Hold
                update_value_by_id("vae_decode", my_unique_id, False)

                def rearrange_list_A(arr, num_cols, num_rows):
                    new_list = []
                    for i in range(num_rows):
                        for j in range(num_cols):
                            index = j * num_rows + i
                            new_list.append(arr[index])
                    return new_list

                def rearrange_list_B(arr, num_rows, num_cols):
                    new_list = []
                    for i in range(num_rows):
                        for j in range(num_cols):
                            index = i * num_cols + j
                            new_list.append(arr[index])
                    return new_list

                # Extract plot dimensions
                num_rows = max(len(Y_value) if Y_value is not None else 0, 1)
                num_cols = max(len(X_value) if X_value is not None else 0, 1)

                # Flip X & Y results back if flipped earlier (for Checkpoint/LoRA For loop optimizations)
                if flip_xy == True:
                    X_type, Y_type = Y_type, X_type
                    X_value, Y_value = Y_value, X_value
                    X_label, Y_label = Y_label, X_label
                    num_rows, num_cols = num_cols, num_rows
                    image_pil_list = rearrange_list_A(image_pil_list, num_rows, num_cols)
                else:
                    image_pil_list = rearrange_list_B(image_pil_list, num_rows, num_cols)
                    image_tensor_list = rearrange_list_A(image_tensor_list, num_cols, num_rows)
                    latent_list = rearrange_list_A(latent_list, num_cols, num_rows)

                # Print XY Plot Results
                print_plot_variables(X_type, Y_type, X_value, Y_value, seed, ckpt_name, lora_params, vae_name,
                                     clip_skip, steps, cfg, sampler_name, scheduler, denoise,
                                     num_rows, num_cols, latent_height, latent_width)

                # Concatenate the tensors along the first dimension (dim=0)
                latent_list = torch.cat(latent_list, dim=0)

                # Store latent_list as last latent
                update_value_by_id("latent", my_unique_id, latent_list)

                # Calculate the dimensions of the white background image
                border_size_top = latent_width // 15

                # Longest Y-label length
                if len(Y_label) > 0:
                    Y_label_longest = max(len(s) for s in Y_label)
                else:
                    # Handle the case when the sequence is empty
                    Y_label_longest = 0  # or any other appropriate value

                Y_label_scale = min(Y_label_longest + 4,24) / 24

                if Y_label_orientation == "Vertical":
                    border_size_left = border_size_top
                else:  # Assuming Y_label_orientation is "Horizontal"
                    # border_size_left is now min(latent_width, latent_height) plus 20% of the difference between the two
                    border_size_left = min(latent_width, latent_height) + int(0.2 * abs(latent_width - latent_height))
                    border_size_left = int(border_size_left * Y_label_scale)

                # Modify the border size, background width and x_offset initialization based on Y_type and Y_label_orientation
                if Y_type == "Nothing":
                    bg_width = num_cols * latent_width + (num_cols - 1) * grid_spacing
                    x_offset_initial = 0
                else:
                    if Y_label_orientation == "Vertical":
                        bg_width = num_cols * latent_width + (num_cols - 1) * grid_spacing + 3 * border_size_left
                        x_offset_initial = border_size_left * 3
                    else:  # Assuming Y_label_orientation is "Horizontal"
                        bg_width = num_cols * latent_width + (num_cols - 1) * grid_spacing + border_size_left
                        x_offset_initial = border_size_left

                # Modify the background height based on X_type
                if X_type == "Nothing":
                    bg_height = num_rows * latent_height + (num_rows - 1) * grid_spacing
                    y_offset = 0
                else:
                    bg_height = num_rows * latent_height + (num_rows - 1) * grid_spacing + 3 * border_size_top
                    y_offset = border_size_top * 3

                # Create the white background image
                background = Image.new('RGBA', (int(bg_width), int(bg_height)), color=(255, 255, 255, 255))

                for row in range(num_rows):

                    # Initialize the X_offset
                    x_offset = x_offset_initial

                    for col in range(num_cols):
                        # Calculate the index for image_pil_list
                        index = col * num_rows + row
                        img = image_pil_list[index]

                        # Paste the image
                        background.paste(img, (x_offset, y_offset))

                        if row == 0 and X_type != "Nothing":
                            # Assign text
                            text = X_label[col]

                            # Add the corresponding X_value as a label above the image
                            initial_font_size = int(48 * img.width / 512)
                            font_size = adjusted_font_size(text, initial_font_size, img.width)
                            label_height = int(font_size*1.5)

                            # Create a white background label image
                            label_bg = Image.new('RGBA', (img.width, label_height), color=(255, 255, 255, 0))
                            d = ImageDraw.Draw(label_bg)

                            # Create the font object
                            font = ImageFont.truetype(str(Path(font_path)), font_size)

                            # Calculate the text size and the starting position
                            _, _, text_width, text_height = d.textbbox([0,0], text, font=font)
                            text_x = (img.width - text_width) // 2
                            text_y = (label_height - text_height) // 2

                            # Add the text to the label image
                            d.text((text_x, text_y), text, fill='black', font=font)

                            # Calculate the available space between the top of the background and the top of the image
                            available_space = y_offset - label_height

                            # Calculate the new Y position for the label image
                            label_y = available_space // 2

                            # Paste the label image above the image on the background using alpha_composite()
                            background.alpha_composite(label_bg, (x_offset, label_y))

                        if col == 0 and Y_type != "Nothing":
                            # Assign text
                            text = Y_label[row]

                            # Add the corresponding Y_value as a label to the left of the image
                            if Y_label_orientation == "Vertical":
                                initial_font_size = int(48 * latent_width / 512)  # Adjusting this to be same as X_label size
                                font_size = adjusted_font_size(text, initial_font_size, latent_width)
                            else:  # Assuming Y_label_orientation is "Horizontal"
                                initial_font_size = int(48 *  (border_size_left/Y_label_scale) / 512)  # Adjusting this to be same as X_label size
                                font_size = adjusted_font_size(text, initial_font_size,  int(border_size_left/Y_label_scale))

                            # Create a white background label image
                            label_bg = Image.new('RGBA', (img.height, int(font_size*1.2)), color=(255, 255, 255, 0))
                            d = ImageDraw.Draw(label_bg)

                            # Create the font object
                            font = ImageFont.truetype(str(Path(font_path)), font_size)

                            # Calculate the text size and the starting position
                            _, _, text_width, text_height = d.textbbox([0,0], text, font=font)
                            text_x = (img.height - text_width) // 2
                            text_y = (font_size - text_height) // 2

                            # Add the text to the label image
                            d.text((text_x, text_y), text, fill='black', font=font)

                            # Rotate the label_bg 90 degrees counter-clockwise only if Y_label_orientation is "Vertical"
                            if Y_label_orientation == "Vertical":
                                label_bg = label_bg.rotate(90, expand=True)

                            # Calculate the available space between the left of the background and the left of the image
                            available_space = x_offset - label_bg.width

                            # Calculate the new X position for the label image
                            label_x = available_space // 2

                            # Calculate the Y position for the label image based on its orientation
                            if Y_label_orientation == "Vertical":
                                label_y = y_offset + (img.height - label_bg.height) // 2
                            else:  # Assuming Y_label_orientation is "Horizontal"
                                label_y = y_offset + img.height - (img.height - label_bg.height) // 2

                            # Paste the label image to the left of the image on the background using alpha_composite()
                            background.alpha_composite(label_bg, (label_x, label_y))

                        # Update the x_offset
                        x_offset += img.width + grid_spacing

                    # Update the y_offset
                    y_offset += img.height + grid_spacing

                images = pil2tensor(background)

             # Generate image results and store
            results = preview_images(images, filename_prefix)
            update_value_by_id("results", my_unique_id, results)

            # Squeeze and Stack the tensors, and store results
            if xyplot_as_output_image == False:
                image_tensor_list = torch.stack([tensor.squeeze() for tensor in image_tensor_list])
            else:
                image_tensor_list = images
            update_value_by_id("images", my_unique_id, image_tensor_list)

            # Print cache if set to true
            if cache_models == "True":
                print_loaded_objects_entries(script_node_id, prompt)

            print("-" * 40)  # Print an empty line followed by a separator line

            images = list() if preview_image == "Output Only" else results

            return {
                "ui": {"images": images},
                "result": (model, positive, negative, {"samples": latent_list}, vae, image_tensor_list,)
            }

########################################################################################################################
# TSC XY Plot
class TSC_XYplot:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "grid_spacing": ("INT", {"default": 0, "min": 0, "max": 500, "step": 5}),
                    "XY_flip": (["False","True"],),
                    "Y_label_orientation": (["Horizontal", "Vertical"],),
                    "cache_models": (["True", "False"],),
                    "ksampler_output_image": (["Plot", "Images"],),},
                "optional": {
                    "dependencies": ("DEPENDENCIES", ),
                    "X": ("XY", ),
                    "Y": ("XY", ),},
        }

    RETURN_TYPES = ("SCRIPT",)
    RETURN_NAMES = ("SCRIPT",)
    FUNCTION = "XYplot"
    CATEGORY = "Efficiency Nodes/XY Plot"

    def XYplot(self, grid_spacing, XY_flip, Y_label_orientation, cache_models, ksampler_output_image, dependencies=None, X=None, Y=None):

        # Unpack X & Y Tuples if connected
        if X != None:
            X_type, X_value  = X
        else:
            X_type = "Nothing"
            X_value = [""]
        if Y != None:
            Y_type, Y_value = Y
        else:
            Y_type = "Nothing"
            Y_value = [""]

        # If types are the same exit. If one isn't "Nothing", print error
        if (X_type == Y_type):
            if X_type != "Nothing":
                print(f"\033[31mXY Plot Error:\033[0m X and Y input types must be different.")
            return (None,)

        # Check that dependencies is connected for Checkpoint and LoRA plots
        types = ("Checkpoint", "LoRA", "Positive Prompt S/R", "Negative Prompt S/R")
        if X_type in types or Y_type in types:
            if dependencies == None: # Not connected
                print(f"\033[31mXY Plot Error:\033[0m The dependencies input must be connected for certain plot types.")
                # Return None
                return (None,)

        # Define X/Y_values for "Seeds++ Batch"
        if X_type == "Seeds++ Batch":
            X_value = [i for i in range(X_value[0])]
        if Y_type == "Seeds++ Batch":
            Y_value = [i for i in range(Y_value[0])]

        # Clean Schedulers from Sampler data (if other type is Scheduler)
        if X_type == "Sampler" and Y_type == "Scheduler":
            # Clear X_value Scheduler's
            X_value = [(x[0], "") for x in X_value]
        elif Y_type == "Sampler" and X_type == "Scheduler":
            # Clear Y_value Scheduler's
            Y_value = [(y[0], "") for y in Y_value]

        # Embed information into "Scheduler" X/Y_values for text label
        if X_type == "Scheduler" and Y_type != "Sampler":
            # X_value second tuple value of each array entry = None
            X_value = [(x, None) for x in X_value]

        if Y_type == "Scheduler" and X_type != "Sampler":
            # Y_value second tuple value of each array entry = None
            Y_value = [(y, None) for y in Y_value]

        # Optimize image generation by prioritizing Checkpoint>LoRA>VAE>PromptSR as X in For Loop. Flip back when done.
        if Y_type == "Checkpoint" or \
                Y_type == "LoRA" and X_type not in {"Checkpoint"} or \
                Y_type == "VAE" and X_type not in {"Checkpoint", "LoRA"} or \
                Y_type == "Positive Prompt S/R" and X_type not in {"Checkpoint", "LoRA", "VAE",
                                                                   "Negative Prompt S/R"} or \
                Y_type == "Negative Prompt S/R" and X_type not in {"Checkpoint", "LoRA", "VAE",
                                                                   "Positive Prompt S/R"} or \
                X_type == "Nothing" and Y_type != "Nothing":
            flip_xy = True
            X_type, Y_type = Y_type, X_type
            X_value, Y_value = Y_value, X_value
        else:
            flip_xy = False

        # Flip X and Y
        if XY_flip == "True":
            X_type, Y_type = Y_type, X_type
            X_value, Y_value = Y_value, X_value

        # Define Ksampler output image behavior
        xyplot_as_output_image = ksampler_output_image == "Plot"

        return ((X_type, X_value, Y_type, Y_value, grid_spacing, Y_label_orientation, cache_models,
                 xyplot_as_output_image, flip_xy, dependencies),)


# TSC XY Plot: Seeds Values
class TSC_XYplot_SeedsBatch:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "batch_count": ("INT", {"default": 1, "min": 0, "max": 50}),},
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, batch_count):
        if batch_count == 0:
            return (None,)
        xy_type = "Seeds++ Batch"
        xy_value = [batch_count]
        return ((xy_type, xy_value),)

# TSC XY Plot: Step Values
class TSC_XYplot_Steps:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "select_count": ("INT", {"default": 0, "min": 0, "max": 5}),
            "steps_1": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "steps_2": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "steps_3": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "steps_4": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "steps_5": ("INT", {"default": 20, "min": 1, "max": 10000}),},
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, select_count, steps_1, steps_2, steps_3, steps_4, steps_5):
        xy_type = "Steps"
        xy_value = [step for idx, step in enumerate([steps_1, steps_2, steps_3, steps_4, steps_5], start=1) if
                 idx <= select_count]
        if not xy_value:  # Check if the list is empty
            return (None,)
        return ((xy_type, xy_value),)


# TSC XY Plot: CFG Values
class TSC_XYplot_CFG:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "select_count": ("INT", {"default": 0, "min": 0, "max": 5}),
            "cfg_1": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
            "cfg_2": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
            "cfg_3": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
            "cfg_4": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
            "cfg_5": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),},
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, select_count, cfg_1, cfg_2, cfg_3, cfg_4, cfg_5):
        xy_type = "CFG Scale"
        xy_value = [cfg for idx, cfg in enumerate([cfg_1, cfg_2, cfg_3, cfg_4, cfg_5], start=1) if idx <= select_count]
        if not xy_value:  # Check if the list is empty
            return (None,)
        return ((xy_type, xy_value),)


# TSC XY Plot: Sampler Values
class TSC_XYplot_Sampler:
    
    samplers = ["None"] + comfy.samplers.KSampler.SAMPLERS
    schedulers = ["None"] + comfy.samplers.KSampler.SCHEDULERS

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                            "sampler_1": (cls.samplers,),
                            "scheduler_1": (cls.schedulers,),
                            "sampler_2": (cls.samplers,),
                            "scheduler_2": (cls.schedulers,),
                            "sampler_3": (cls.samplers,),
                            "scheduler_3": (cls.schedulers,),
                            "sampler_4": (cls.samplers,),
                            "scheduler_4": (cls.schedulers,),
                            "sampler_5": (cls.samplers,),
                            "scheduler_5": (cls.schedulers,),},
                }
    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, sampler_1, scheduler_1, sampler_2, scheduler_2, sampler_3, scheduler_3,
                 sampler_4, scheduler_4, sampler_5, scheduler_5):

        samplers = [sampler_1, sampler_2, sampler_3, sampler_4, sampler_5]
        schedulers = [scheduler_1, scheduler_2, scheduler_3, scheduler_4, scheduler_5]

        pairs = []
        for sampler, scheduler in zip(samplers, schedulers):
            if sampler != "None":
                if scheduler != "None":
                    pairs.append((sampler, scheduler))
                else:
                    pairs.append((sampler,None))

        xy_type = "Sampler"
        xy_value = pairs
        if not xy_value:  # Check if the list is empty
            return (None,)
        return ((xy_type, xy_value),)


# TSC XY Plot: Scheduler Values
class TSC_XYplot_Scheduler:

    schedulers = ["None"] + comfy.samplers.KSampler.SCHEDULERS

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "scheduler_1": (cls.schedulers,),
            "scheduler_2": (cls.schedulers,),
            "scheduler_3": (cls.schedulers,),
            "scheduler_4": (cls.schedulers,),
            "scheduler_5": (cls.schedulers,),},
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, scheduler_1, scheduler_2, scheduler_3, scheduler_4, scheduler_5):
        xy_type = "Scheduler"
        xy_value = [scheduler for scheduler in [scheduler_1, scheduler_2, scheduler_3, scheduler_4, scheduler_5] if
                      scheduler != "None"]
        if not xy_value:  # Check if the list is empty
            return (None,)
        return ((xy_type, xy_value),)


# TSC XY Plot: Denoise Values
class TSC_XYplot_Denoise:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "select_count": ("INT", {"default": 0, "min": 0, "max": 5}),
            "denoise_1": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 1.0, "step": 0.01}),
            "denoise_2": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 1.0, "step": 0.01}),
            "denoise_3": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 1.0, "step": 0.01}),
            "denoise_4": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 1.0, "step": 0.01}),
            "denoise_5": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 1.0, "step": 0.01}),},
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, select_count, denoise_1, denoise_2, denoise_3, denoise_4, denoise_5):
        xy_type = "Denoise"
        xy_value = [denoise for idx, denoise in
                    enumerate([denoise_1, denoise_2, denoise_3, denoise_4, denoise_5], start=1) if idx <= select_count]
        if not xy_value:  # Check if the list is empty
            return (None,)
        return ((xy_type, xy_value),)


# TSC XY Plot: VAE Values
class TSC_XYplot_VAE:

    vaes = ["None"] + folder_paths.get_filename_list("vae")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "vae_name_1": (cls.vaes,),
            "vae_name_2": (cls.vaes,),
            "vae_name_3": (cls.vaes,),
            "vae_name_4": (cls.vaes,),
            "vae_name_5": (cls.vaes,),},
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, vae_name_1, vae_name_2, vae_name_3, vae_name_4, vae_name_5):
        xy_type = "VAE"
        xy_value = [vae for vae in [vae_name_1, vae_name_2, vae_name_3, vae_name_4, vae_name_5] if vae != "None"]
        if not xy_value:  # Check if the list is empty
            return (None,)
        return ((xy_type, xy_value),)


# TSC XY Plot: Prompt S/R Positive
class TSC_XYplot_PromptSR_Positive:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "search_txt": ("STRING", {"default": "", "multiline": False}),
            "replace_count": ("INT", {"default": 0, "min": 0, "max": 4}),
            "replace_1":("STRING", {"default": "", "multiline": False}),
            "replace_2": ("STRING", {"default": "", "multiline": False}),
            "replace_3": ("STRING", {"default": "", "multiline": False}),
            "replace_4": ("STRING", {"default": "", "multiline": False}),},
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, search_txt, replace_count, replace_1, replace_2, replace_3, replace_4):
        # If search_txt is empty, return (None,)
        if search_txt == "":
            return (None,)

        xy_type = "Positive Prompt S/R"

        # Create a list of replacement arguments
        replacements = [replace_1, replace_2, replace_3, replace_4]

        # Create base entry
        xy_values = [(search_txt, None)]

        if replace_count > 0:
            # Append additional entries based on replace_count
            xy_values.extend([(search_txt, replacements[i]) for i in range(replace_count)])

        return ((xy_type, xy_values),)

# TSC XY Plot: Prompt S/R Negative
class TSC_XYplot_PromptSR_Negative:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "search_txt": ("STRING", {"default": "", "multiline": False}),
            "replace_count": ("INT", {"default": 0, "min": 0, "max": 4}),
            "replace_1":("STRING", {"default": "", "multiline": False}),
            "replace_2": ("STRING", {"default": "", "multiline": False}),
            "replace_3": ("STRING", {"default": "", "multiline": False}),
            "replace_4": ("STRING", {"default": "", "multiline": False}),},
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, search_txt, replace_count, replace_1, replace_2, replace_3, replace_4):
        # If search_txt is empty, return (None,)
        if search_txt == "":
            return (None,)

        xy_type = "Negative Prompt S/R"

        # Create a list of replacement arguments
        replacements = [replace_1, replace_2, replace_3, replace_4]

        # Create base entry
        xy_values = [(search_txt, None)]

        if replace_count > 0:
            # Append additional entries based on replace_count
            xy_values.extend([(search_txt, replacements[i]) for i in range(replace_count)])

        return ((xy_type, xy_values),)

# TSC XY Plot: Checkpoint Values
class TSC_XYplot_Checkpoint:

    checkpoints = ["None"] + folder_paths.get_filename_list("checkpoints")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "ckpt_name_1": (cls.checkpoints,),
            "clip_skip1": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
            "ckpt_name_2": (cls.checkpoints,),
            "clip_skip2": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
            "ckpt_name_3": (cls.checkpoints,),
            "clip_skip3": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
            "ckpt_name_4": (cls.checkpoints,),
            "clip_skip4": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
            "ckpt_name_5": (cls.checkpoints,),
            "clip_skip5": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),},
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, ckpt_name_1, clip_skip1, ckpt_name_2, clip_skip2, ckpt_name_3, clip_skip3,
                 ckpt_name_4, clip_skip4, ckpt_name_5, clip_skip5):
        xy_type = "Checkpoint"
        checkpoints = [ckpt_name_1, ckpt_name_2, ckpt_name_3, ckpt_name_4, ckpt_name_5]
        clip_skips = [clip_skip1, clip_skip2, clip_skip3, clip_skip4, clip_skip5]
        xy_value = [(checkpoint, clip_skip) for checkpoint, clip_skip in zip(checkpoints, clip_skips) if
                    checkpoint != "None"]
        if not xy_value:  # Check if the list is empty
            return (None,)
        return ((xy_type, xy_value),)


# TSC XY Plot: Checkpoint Merge
class TSC_XYplot_Checkpoint_Merge:
    checkpoints = ["None"] + folder_paths.get_filename_list("checkpoints")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "ckpt_name_1": (cls.checkpoints,),
            "ckpt_weight1": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
            "ckpt_name_2": (cls.checkpoints,),
            "ckpt_weight2": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
            "ckpt_name_3": (cls.checkpoints,),
            "ckpt_weight3": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
            "ckpt_name_4": (cls.checkpoints,),
            "ckpt_weight4": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
            "ckpt_name_5": (cls.checkpoints,),
            "ckpt_weight5": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
            },
        }
        
    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"
    
    def xy_value(self, ckpt_name_1, ckpt_weight1, ckpt_name_2, ckpt_weight2, ckpt_name_3, ckpt_weight3,
                 ckpt_name_4, ckpt_weight4, ckpt_name_5, ckpt_weight5):
        xy_type = "Checkpoint Merge"
        checkpoints = [ckpt_name_1, ckpt_name_2, ckpt_name_3, ckpt_name_4, ckpt_name_5]
        checkpoints_weights = [ckpt_weight1, ckpt_weight2, ckpt_weight3, ckpt_weight4, ckpt_weight5]
        clip_weights = [ckpt_weight1, ckpt_weight2, ckpt_weight3, ckpt_weight4, ckpt_weight5]
        xy_value = [(checkpoint, ckptweight, clipweight) for checkpoint, ckptweight, clipweight in zip(checkpoints, checkpoints_weights, clip_weights) if
                    checkpoint != "None"]
        if not xy_value:  # Check if the list is empty
            return (None,)
        return ((xy_type, xy_value),)
    
    
# TSC XY Plot: Checkpoint Merge Advanced
# Allows to set different checkpoint and clip weights
class TSC_XYplot_Checkpoint_Merge_Advanced:
    checkpoints = ["None"] + folder_paths.get_filename_list("checkpoints")
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "ckpt_name_1": (cls.checkpoints,),
            "ckpt_weight1": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
            "clip_weight1": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
            "ckpt_name_2": (cls.checkpoints,),
            "ckpt_weight2": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
            "clip_weight2": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
            "ckpt_name_3": (cls.checkpoints,),
            "ckpt_weight3": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
            "clip_weight3": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
            "ckpt_name_4": (cls.checkpoints,),
            "ckpt_weight4": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
            "clip_weight4": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
            "ckpt_name_5": (cls.checkpoints,),
            "ckpt_weight5": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
            "clip_weight5": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
            },
        }
        
    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"
    
    def xy_value(self, ckpt_name_1, ckpt_weight1, clip_weight1, ckpt_name_2, ckpt_weight2, clip_weight2, ckpt_name_3, ckpt_weight3, clip_weight3,
                 ckpt_name_4, ckpt_weight4, clip_weight4, ckpt_name_5, ckpt_weight5, clip_weight5):
        xy_type = "Checkpoint Merge Adv"
        checkpoints = [ckpt_name_1, ckpt_name_2, ckpt_name_3, ckpt_name_4, ckpt_name_5]
        checkpoints_weights = [ckpt_weight1, ckpt_weight2, ckpt_weight3, ckpt_weight4, ckpt_weight5]
        clip_weights = [clip_weight1, clip_weight2, clip_weight3, clip_weight4, clip_weight5]
        xy_value = [(checkpoint, ckptweight, clipweight) for checkpoint, ckptweight, clipweight in zip(checkpoints, checkpoints_weights, clip_weights) if
                    checkpoint != "None"]
        if not xy_value:  # Check if the list is empty
            return (None,)
        return ((xy_type, xy_value),)


# TSC XY Plot: Clip Skip
class TSC_XYplot_ClipSkip:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "select_count": ("INT", {"default": 0, "min": 0, "max": 5}),
            "clip_skip_1": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
            "clip_skip_2": ("INT", {"default": -2, "min": -24, "max": -1, "step": 1}),
            "clip_skip_3": ("INT", {"default": -3, "min": -24, "max": -1, "step": 1}),
            "clip_skip_4": ("INT", {"default": -4, "min": -24, "max": -1, "step": 1}),
            "clip_skip_5": ("INT", {"default": -5, "min": -24, "max": -1, "step": 1}),},
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, select_count, clip_skip_1, clip_skip_2, clip_skip_3, clip_skip_4, clip_skip_5):
        xy_type = "Clip Skip"
        xy_value = [clip_skip for idx, clip_skip in
                    enumerate([clip_skip_1, clip_skip_2, clip_skip_3, clip_skip_4, clip_skip_5], start=1) if idx <= select_count]
        if not xy_value:  # Check if the list is empty
            return (None,)
        return ((xy_type, xy_value),)

# TSC XY Plot: LoRA Values
class TSC_XYplot_LoRA:

    loras = ["None"] + folder_paths.get_filename_list("loras")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                            "model_strengths": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                            "clip_strengths": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                            "lora_name_1": (cls.loras,),
                            "lora_name_2": (cls.loras,),
                            "lora_name_3": (cls.loras,),
                            "lora_name_4": (cls.loras,),
                            "lora_name_5": (cls.loras,)},
                "optional": {"lora_stack": ("LORA_STACK", )}
                }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, model_strengths, clip_strengths, lora_name_1, lora_name_2, lora_name_3, lora_name_4, lora_name_5,
                 lora_stack=None):
        xy_type = "LoRA"
        loras = [lora_name_1, lora_name_2, lora_name_3, lora_name_4, lora_name_5]

        # Extend each sub-array with lora_stack if it's not None
        xy_value = [[(lora, model_strengths, clip_strengths)] + (lora_stack if lora_stack else []) for lora in loras if
                    lora != "None"]

        if not xy_value:  # Check if the list is empty
            return (None,)
        return ((xy_type, xy_value),)


# TSC XY Plot: LoRA Advanced
class TSC_XYplot_LoRA_Adv:

    loras = ["None"] + folder_paths.get_filename_list("loras")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                            "lora_name_1": (cls.loras,),
                            "model_str_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                            "clip_str_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                            "lora_name_2": (cls.loras,),
                            "model_str_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                            "clip_str_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                            "lora_name_3": (cls.loras,),
                            "model_str_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                            "clip_str_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                            "lora_name_4": (cls.loras,),
                            "model_str_4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                            "clip_str_4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                            "lora_name_5": (cls.loras,),
                            "model_str_5": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                            "clip_str_5": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),},
                "optional": {"lora_stack": ("LORA_STACK",)}
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, lora_name_1, model_str_1, clip_str_1, lora_name_2, model_str_2, clip_str_2, lora_name_3,
                 model_str_3,
                 clip_str_3, lora_name_4, model_str_4, clip_str_4, lora_name_5, model_str_5, clip_str_5,
                 lora_stack=None):
        xy_type = "LoRA"
        loras = [lora_name_1, lora_name_2, lora_name_3, lora_name_4, lora_name_5]
        model_strs = [model_str_1, model_str_2, model_str_3, model_str_4, model_str_5]
        clip_strs = [clip_str_1, clip_str_2, clip_str_3, clip_str_4, clip_str_5]

        # Extend each sub-array with lora_stack if it's not None
        xy_value = [[(lora, model_str, clip_str)] + (lora_stack if lora_stack else []) for lora, model_str, clip_str in
                    zip(loras, model_strs, clip_strs) if lora != "None"]

        if not xy_value:  # Check if the list is empty
            return (None,)
        return ((xy_type, xy_value),)


# TSC XY Plot: LoRA Stacks
class TSC_XYplot_LoRA_Stacks:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "node_state": (["Enabled", "Disabled"],)},
                "optional": {
                    "lora_stack_1": ("LORA_STACK",),
                    "lora_stack_2": ("LORA_STACK",),
                    "lora_stack_3": ("LORA_STACK",),
                    "lora_stack_4": ("LORA_STACK",),
                    "lora_stack_5": ("LORA_STACK",),},
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, node_state, lora_stack_1=None, lora_stack_2=None, lora_stack_3=None, lora_stack_4=None, lora_stack_5=None):
        xy_type = "LoRA"
        xy_value = [stack for stack in [lora_stack_1, lora_stack_2, lora_stack_3, lora_stack_4, lora_stack_5] if stack is not None]
        if not xy_value or not any(xy_value) or node_state == "Disabled":
            return (None,)
        else:
            return ((xy_type, xy_value),)

# TSC XY Plot: Manual Entry Notes
class TSC_XYplot_Manual_XY_Entry_Info:

    syntax = "(X/Y_types)     (X/Y_values)\n" \
               "Seeds++ Batch   batch_count\n" \
               "Steps           steps_1;steps_2;...\n" \
               "CFG Scale       cfg_1;cfg_2;...\n" \
               "Sampler(1)      sampler_1;sampler_2;...\n" \
               "Sampler(2)      sampler_1,scheduler_1;...\n" \
               "Sampler(3)      sampler_1;...;,default_scheduler\n" \
               "Scheduler       scheduler_1;scheduler_2;...\n" \
               "Denoise         denoise_1;denoise_2;...\n" \
               "VAE             vae_1;vae_2;vae_3;...\n" \
               "+Prompt S/R     search_txt;replace_1;replace_2;...\n" \
               "-Prompt S/R     search_txt;replace_1;replace_2;...\n" \
               "Checkpoint(1)   ckpt_1;ckpt_2;ckpt_3;...\n" \
               "Checkpoint(2)   ckpt_1,clip_skip_1;...\n" \
               "Checkpoint(3)   ckpt_1;ckpt_2;...;,default_clip_skip\n" \
               "Clip Skip       clip_skip_1;clip_skip_2;...\n" \
               "LoRA(1)         lora_1;lora_2;lora_3;...\n" \
               "LoRA(2)         lora_1;...;,default_model_str,default_clip_str\n" \
               "LoRA(3)         lora_1,model_str_1,clip_str_1;..."

    samplers = ";\n".join(comfy.samplers.KSampler.SAMPLERS)
    schedulers = ";\n".join(comfy.samplers.KSampler.SCHEDULERS)
    vaes = ";\n".join(folder_paths.get_filename_list("vae"))
    ckpts = ";\n".join(folder_paths.get_filename_list("checkpoints"))
    loras = ";\n".join(folder_paths.get_filename_list("loras"))

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "notes": ("STRING", {"default":
                                    f"_____________SYNTAX_____________\n{cls.syntax}\n\n"
                                    f"____________SAMPLERS____________\n{cls.samplers}\n\n"
                                    f"___________SCHEDULERS___________\n{cls.schedulers}\n\n"
                                    f"_____________VAES_______________\n{cls.vaes}\n\n"
                                    f"___________CHECKPOINTS__________\n{cls.ckpts}\n\n"
                                    f"_____________LORAS______________\n{cls.loras}\n","multiline": True}),},}

    RETURN_TYPES = ()
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

# TSC XY Plot: Manual Entry
class TSC_XYplot_Manual_XY_Entry:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "X_type": (["Nothing", "Seeds++ Batch", "Steps", "CFG Scale", "Sampler", "Scheduler", "Denoise", "VAE",
                        "Positive Prompt S/R", "Negative Prompt S/R", "Checkpoint", "Clip Skip", "LoRA"],),
            "X_value": ("STRING", {"default": "", "multiline": True}),
            "Y_type": (["Nothing", "Seeds++ Batch", "Steps", "CFG Scale", "Sampler", "Scheduler", "Denoise", "VAE",
                        "Positive Prompt S/R", "Negative Prompt S/R", "Checkpoint", "Clip Skip", "LoRA"],),
            "Y_value": ("STRING", {"default": "", "multiline": True}),},}

    RETURN_TYPES = ("XY", "XY",)
    RETURN_NAMES = ("X", "Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, X_type, X_value, Y_type, Y_value, prompt=None, my_unique_id=None):

        # Store X values as arrays
        if X_type not in {"Positive Prompt S/R", "Negative Prompt S/R", "VAE", "Checkpoint", "LoRA"}:
            X_value = X_value.replace(" ", "")  # Remove spaces
        X_value = X_value.replace("\n", "")  # Remove newline characters
        X_value = X_value.rstrip(";")  # Remove trailing semicolon
        X_value = X_value.split(";")  # Turn to array

        # Store Y values as arrays
        if Y_type not in {"Positive Prompt S/R", "Negative Prompt S/R", "VAE", "Checkpoint", "LoRA"}:
            Y_value = Y_value.replace(" ", "")  # Remove spaces
        Y_value = Y_value.replace("\n", "")  # Remove newline characters
        Y_value = Y_value.rstrip(";")  # Remove trailing semicolon
        Y_value = Y_value.split(";")  # Turn to array

        # Define the valid bounds for each type
        bounds = {
            "Seeds++ Batch": {"min": 1, "max": 50},
            "Steps": {"min": 1, "max": 10000},
            "CFG Scale": {"min": 0, "max": 100},
            "Sampler": {"options": comfy.samplers.KSampler.SAMPLERS},
            "Scheduler": {"options": comfy.samplers.KSampler.SCHEDULERS},
            "Denoise": {"min": 0, "max": 1},
            "VAE": {"options": folder_paths.get_filename_list("vae")},
            "Checkpoint": {"options": folder_paths.get_filename_list("checkpoints")},
            "Clip Skip": {"min": -24, "max": -1},
            "LoRA": {"options": folder_paths.get_filename_list("loras"),
                     "model_str": {"min": 0, "max": 10},"clip_str": {"min": 0, "max": 10},},
        }

        # Validates a value based on its corresponding value_type and bounds.
        def validate_value(value, value_type, bounds):
            # ________________________________________________________________________
            # Seeds++ Batch
            if value_type == "Seeds++ Batch":
                try:
                    x = int(float(value))
                    if x < bounds["Seeds++ Batch"]["min"]:
                        x = bounds["Seeds++ Batch"]["min"]
                    elif x > bounds["Seeds++ Batch"]["max"]:
                        x = bounds["Seeds++ Batch"]["max"]
                except ValueError:
                    print(f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid batch count.")
                    return None
                if float(value) != x:
                    print(f"\033[31mmXY Plot Error:\033[0m '{value}' is not a valid batch count.")
                    return None
                return x
            # ________________________________________________________________________
            # Steps
            elif value_type == "Steps":
                try:
                    x = int(value)
                    if x < bounds["Steps"]["min"]:
                        x = bounds["Steps"]["min"]
                    elif x > bounds["Steps"]["max"]:
                        x = bounds["Steps"]["max"]
                    return x
                except ValueError:
                    print(
                        f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid Step count.")
                    return None
            # ________________________________________________________________________
            # CFG Scale
            elif value_type == "CFG Scale":
                try:
                    x = float(value)
                    if x < bounds["CFG Scale"]["min"]:
                        x = bounds["CFG Scale"]["min"]
                    elif x > bounds["CFG Scale"]["max"]:
                        x = bounds["CFG Scale"]["max"]
                    return x
                except ValueError:
                    print(
                        f"\033[31mXY Plot Error:\033[0m '{value}' is not a number between {bounds['CFG Scale']['min']}"
                        f" and {bounds['CFG Scale']['max']} for CFG Scale.")
                    return None
            # ________________________________________________________________________
            # Sampler
            elif value_type == "Sampler":
                if isinstance(value, str) and ',' in value:
                    value = tuple(map(str.strip, value.split(',')))
                if isinstance(value, tuple):
                    if len(value) >= 2:
                        value = value[:2]  # Slice the value tuple to keep only the first two elements
                        sampler, scheduler = value
                        scheduler = scheduler.lower()  # Convert the scheduler name to lowercase
                        if sampler not in bounds["Sampler"]["options"]:
                            valid_samplers = '\n'.join(bounds["Sampler"]["options"])
                            print(
                                f"\033[31mXY Plot Error:\033[0m '{sampler}' is not a valid sampler. Valid samplers are:\n{valid_samplers}")
                            sampler = None
                        if scheduler not in bounds["Scheduler"]["options"]:
                            valid_schedulers = '\n'.join(bounds["Scheduler"]["options"])
                            print(
                                f"\033[31mXY Plot Error:\033[0m '{scheduler}' is not a valid scheduler. Valid schedulers are:\n{valid_schedulers}")
                            scheduler = None
                        if sampler is None or scheduler is None:
                            return None
                        else:
                            return sampler, scheduler
                    else:
                        print(
                            f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid sampler.'")
                        return None
                else:
                    if value not in bounds["Sampler"]["options"]:
                        valid_samplers = '\n'.join(bounds["Sampler"]["options"])
                        print(
                            f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid sampler. Valid samplers are:\n{valid_samplers}")
                        return None
                    else:
                        return value, None
            # ________________________________________________________________________
            # Scheduler
            elif value_type == "Scheduler":
                if value not in bounds["Scheduler"]["options"]:
                    valid_schedulers = '\n'.join(bounds["Scheduler"]["options"])
                    print(
                        f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid Scheduler. Valid Schedulers are:\n{valid_schedulers}")
                    return None
                else:
                    return value
            # ________________________________________________________________________
            # Denoise
            elif value_type == "Denoise":
                try:
                    x = float(value)
                    if x < bounds["Denoise"]["min"]:
                        x = bounds["Denoise"]["min"]
                    elif x > bounds["Denoise"]["max"]:
                        x = bounds["Denoise"]["max"]
                    return x
                except ValueError:
                    print(
                        f"\033[31mXY Plot Error:\033[0m '{value}' is not a number between {bounds['Denoise']['min']} "
                        f"and {bounds['Denoise']['max']} for Denoise.")
                    return None
            # ________________________________________________________________________
            # VAE
            elif value_type == "VAE":
                if value not in bounds["VAE"]["options"]:
                    valid_vaes = '\n'.join(bounds["VAE"]["options"])
                    print(f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid VAE. Valid VAEs are:\n{valid_vaes}")
                    return None
                else:
                    return value
            # ________________________________________________________________________
            # Checkpoint
            elif value_type == "Checkpoint":
                if isinstance(value, str) and ',' in value:
                    value = tuple(map(str.strip, value.split(',')))
                if isinstance(value, tuple):
                    if len(value) >= 2:
                        value = value[:2]  # Slice the value tuple to keep only the first two elements
                        checkpoint, clip_skip = value
                        try:
                            clip_skip = int(clip_skip)  # Convert the clip_skip to integer
                        except ValueError:
                            print(f"\033[31mXY Plot Error:\033[0m '{clip_skip}' is not a valid clip_skip. "
                                  f"Valid clip skip values are integers between {bounds['Clip Skip']['min']} and {bounds['Clip Skip']['max']}.")
                            return None
                        if checkpoint not in bounds["Checkpoint"]["options"]:
                            valid_checkpoints = '\n'.join(bounds["Checkpoint"]["options"])
                            print(
                                f"\033[31mXY Plot Error:\033[0m '{checkpoint}' is not a valid checkpoint. Valid checkpoints are:\n{valid_checkpoints}")
                            checkpoint = None
                        if clip_skip < bounds["Clip Skip"]["min"] or clip_skip > bounds["Clip Skip"]["max"]:
                            print(f"\033[31mXY Plot Error:\033[0m '{clip_skip}' is not a valid clip skip. "
                                  f"Valid clip skip values are integers between {bounds['Clip Skip']['min']} and {bounds['Clip Skip']['max']}.")
                            clip_skip = None
                        if checkpoint is None or clip_skip is None:
                            return None
                        else:
                            return checkpoint, clip_skip
                    else:
                        print(
                            f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid checkpoint.'")
                        return None
                else:
                    if value not in bounds["Checkpoint"]["options"]:
                        valid_checkpoints = '\n'.join(bounds["Checkpoint"]["options"])
                        print(
                            f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid checkpoint. Valid checkpoints are:\n{valid_checkpoints}")
                        return None
                    else:
                        return value, None
            # ________________________________________________________________________
            # Clip Skip
            elif value_type == "Clip Skip":
                try:
                    x = int(value)
                    if x < bounds["Clip Skip"]["min"]:
                        x = bounds["Clip Skip"]["min"]
                    elif x > bounds["Clip Skip"]["max"]:
                        x = bounds["Clip Skip"]["max"]
                    return x
                except ValueError:
                    print(f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid Clip Skip.")
                    return None
            # ________________________________________________________________________
            # LoRA
            elif value_type == "LoRA":
                if isinstance(value, str) and ',' in value:
                    value = tuple(map(str.strip, value.split(',')))

                if isinstance(value, tuple):
                    lora_name, model_str, clip_str = (value + (1.0, 1.0))[:3]  # Defaults model_str and clip_str to 1 if not provided

                    if lora_name not in bounds["LoRA"]["options"]:
                        valid_loras = '\n'.join(bounds["LoRA"]["options"])
                        print(f"\033[31mXY Plot Error:\033[0m '{lora_name}' is not a valid LoRA. Valid LoRAs are:\n{valid_loras}")
                        lora_name = None

                    try:
                        model_str = float(model_str)
                        clip_str = float(clip_str)
                    except ValueError:
                        print(f"\033[31mXY Plot Error:\033[0m The LoRA model strength and clip strength values should be numbers"
                              f" between {bounds['LoRA']['model_str']['min']} and {bounds['LoRA']['model_str']['max']}.")
                        return None

                    if model_str < bounds["LoRA"]["model_str"]["min"] or model_str > bounds["LoRA"]["model_str"]["max"]:
                        print(f"\033[31mXY Plot Error:\033[0m '{model_str}' is not a valid LoRA model strength value. "
                              f"Valid lora model strength values are between {bounds['LoRA']['model_str']['min']} and {bounds['LoRA']['model_str']['max']}.")
                        model_str = None

                    if clip_str < bounds["LoRA"]["clip_str"]["min"] or clip_str > bounds["LoRA"]["clip_str"]["max"]:
                        print(f"\033[31mXY Plot Error:\033[0m '{clip_str}' is not a valid LoRA clip strength value. "
                              f"Valid lora clip strength values are between {bounds['LoRA']['clip_str']['min']} and {bounds['LoRA']['clip_str']['max']}.")
                        clip_str = None

                    if lora_name is None or model_str is None or clip_str is None:
                        return None
                    else:
                        return lora_name, model_str, clip_str
                else:
                    if value not in bounds["LoRA"]["options"]:
                        valid_loras = '\n'.join(bounds["LoRA"]["options"])
                        print(
                            f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid LoRA. Valid LoRAs are:\n{valid_loras}")
                        return None
                    else:
                        return value, 1.0, 1.0

            # ________________________________________________________________________
            else:
                return None

        # Validate X_value array length is 1 if doing a "Seeds++ Batch"
        if len(X_value) != 1 and X_type == "Seeds++ Batch":
            print(f"\033[31mXY Plot Error:\033[0m '{';'.join(X_value)}' is not a valid batch count.")
            return (None,None,)

        # Validate Y_value array length is 1 if doing a "Seeds++ Batch"
        if len(Y_value) != 1 and Y_type == "Seeds++ Batch":
            print(f"\033[31mXY Plot Error:\033[0m '{';'.join(Y_value)}' is not a valid batch count.")
            return (None,None,)

        # Apply allowed shortcut syntax to certain input types
        if X_type in ["Sampler", "Checkpoint", "LoRA"]:
            if X_value[-1].startswith(','):
                # Remove the leading comma from the last entry and store it as suffixes
                suffixes = X_value.pop().lstrip(',').split(',')
                # Split all preceding entries into subentries
                X_value = [entry.split(',') for entry in X_value]
                # Make all entries the same length as suffixes by appending missing elements
                for entry in X_value:
                    entry += suffixes[len(entry) - 1:]
                # Join subentries back into strings
                X_value = [','.join(entry) for entry in X_value]

        # Apply allowed shortcut syntax to certain input types
        if Y_type in ["Sampler", "Checkpoint", "LoRA"]:
            if Y_value[-1].startswith(','):
                # Remove the leading comma from the last entry and store it as suffixes
                suffixes = Y_value.pop().lstrip(',').split(',')
                # Split all preceding entries into subentries
                Y_value = [entry.split(',') for entry in Y_value]
                # Make all entries the same length as suffixes by appending missing elements
                for entry in Y_value:
                    entry += suffixes[len(entry) - 1:]
                # Join subentries back into strings
                Y_value = [','.join(entry) for entry in Y_value]

        # Prompt S/R X Cleanup
        if X_type in {"Positive Prompt S/R", "Negative Prompt S/R"}:
            if X_value[0] == '':
                print(f"\033[31mXY Plot Error:\033[0m Prompt S/R value can not be empty.")
                return (None, None,)
            else:
                X_value = [(X_value[0], None) if i == 0 else (X_value[0], x) for i, x in enumerate(X_value)]

        # Prompt S/R X Cleanup
        if Y_type in {"Positive Prompt S/R", "Negative Prompt S/R"}:
            if Y_value[0] == '':
                print(f"\033[31mXY Plot Error:\033[0m Prompt S/R value can not be empty.")
                return (None, None,)
            else:
                Y_value = [(Y_value[0], None) if i == 0 else (Y_value[0], y) for i, y in enumerate(Y_value)]

        # Loop over each entry in X_value and check if it's valid
        if X_type not in {"Nothing", "Positive Prompt S/R", "Negative Prompt S/R"}:
            for i in range(len(X_value)):
                X_value[i] = validate_value(X_value[i], X_type, bounds)
                if X_value[i] == None:
                    return (None,None,)

        # Loop over each entry in Y_value and check if it's valid
        if Y_type not in {"Nothing", "Positive Prompt S/R", "Negative Prompt S/R"}:
            for i in range(len(Y_value)):
                Y_value[i] = validate_value(Y_value[i], Y_type, bounds)
                if Y_value[i] == None:
                    return (None,None,)

        # Nest LoRA value in another array to reflect LoRA stack changes
        if X_type == "LoRA":
            X_value = [[x] for x in X_value]
        if Y_type == "LoRA":
            Y_value = [[y] for y in Y_value]

        # Clean X/Y_values
        if X_type == "Nothing":
            X_value = [""]
        if Y_type == "Nothing":
            Y_value = [""]

        return ((X_type, X_value), (Y_type, Y_value),)

# TSC XY Plot: Seeds Values
class TSC_XYplot_JoinInputs:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "XY_1": ("XY",),
            "XY_2": ("XY",),},
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, XY_1, XY_2):
        xy_type_1, xy_value_1 = XY_1
        xy_type_2, xy_value_2 = XY_2

        if xy_type_1 != xy_type_2:
            print(f"\033[31mJoin XY Inputs Error:\033[0m Input types must match")
            return (None,)
        elif xy_type_1 == "Seeds++ Batch":
            xy_type = xy_type_1
            xy_value = [xy_value_1[0] + xy_value_2[0]]
        elif xy_type_1 == "Positive Prompt S/R" or xy_type_1 == "Negative Prompt S/R":
            xy_type = xy_type_1
            xy_value = xy_value_1 + [(xy_value_1[0][0], t[1]) for t in xy_value_2[1:]]
        else:
            xy_type = xy_type_1
            xy_value = xy_value_1 + xy_value_2
        return ((xy_type, xy_value),)

########################################################################################################################
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
    FUNCTION = "apply_overlay_image"
    CATEGORY = "Efficiency Nodes/Image"

    def apply_overlay_image(self, base_image, overlay_image, overlay_resize, resize_method, rescale_factor,
                            width, height, x_offset, y_offset, rotation, opacity, optional_mask=None):

        # Pack tuples and assign variables
        size = width, height
        location = x_offset, y_offset
        mask = optional_mask

        # Check for different sizing options
        if overlay_resize != "None":
            #Extract overlay_image size and store in Tuple "overlay_image_size" (WxH)
            overlay_image_size = overlay_image.size()
            overlay_image_size = (overlay_image_size[2], overlay_image_size[1])
            if overlay_resize == "Fit":
                overlay_image_size = (base_image.size[0],base_image.size[1])
            elif overlay_resize == "Resize by rescale_factor":
                overlay_image_size = tuple(int(dimension * rescale_factor) for dimension in overlay_image_size)
            elif overlay_resize == "Resize to width & heigth":
                overlay_image_size = (size[0], size[1])

            samples = overlay_image.movedim(-1, 1)
            overlay_image = comfy.utils.common_upscale(samples, overlay_image_size[0], overlay_image_size[1], resize_method, False)
            overlay_image = overlay_image.movedim(1, -1)
            
        overlay_image = tensor2pil(overlay_image)

         # Add Alpha channel to overlay
        overlay_image = overlay_image.convert('RGBA')
        overlay_image.putalpha(Image.new("L", overlay_image.size, 255))

        # If mask connected, check if the overlay_image image has an alpha channel
        if mask is not None:
            # Convert mask to pil and resize
            mask = tensor2pil(mask)
            mask = mask.resize(overlay_image.size)
            # Apply mask as overlay's alpha
            overlay_image.putalpha(ImageOps.invert(mask))

        # Rotate the overlay image
        overlay_image = overlay_image.rotate(rotation, expand=True)

        # Apply opacity on overlay image
        r, g, b, a = overlay_image.split()
        a = a.point(lambda x: max(0, int(x * (1 - opacity / 100))))
        overlay_image.putalpha(a)

        # Split the base_image tensor along the first dimension to get a list of tensors
        base_image_list = torch.unbind(base_image, dim=0)

        # Convert each tensor to a PIL image, apply the overlay, and then convert it back to a tensor
        processed_base_image_list = []
        for tensor in base_image_list:
            # Convert tensor to PIL Image
            image = tensor2pil(tensor)

            # Paste the overlay image onto the base image
            if mask is None:
                image.paste(overlay_image, location)
            else:
                image.paste(overlay_image, location, overlay_image)

            # Convert PIL Image back to tensor
            processed_tensor = pil2tensor(image)

            # Append to list
            processed_base_image_list.append(processed_tensor)

        # Combine the processed images back into a single tensor
        base_image = torch.stack([tensor.squeeze() for tensor in processed_base_image_list])

        # Return the edited base image
        return (base_image,)

########################################################################################################################
# Install simple_eval if missing from packages
def install_simpleeval():
    if 'simpleeval' not in packages():
        print("\033[32mEfficiency Nodes:\033[0m")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'simpleeval'])

def packages(versions=False):
    return [(r.decode().split('==')[0] if not versions else r.decode()) for r in subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).split()]

install_simpleeval()
import simpleeval

# TSC Evaluate Integers (https://github.com/danthedeckie/simpleeval)
class TSC_EvaluateInts:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "python_expression": ("STRING", {"default": "((a + b) - c) / 2", "multiline": False}),
                    "print_to_console": (["False", "True"],),},
                "optional": {
                    "a": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 1}),
                    "b": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 1}),
                    "c": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 1}),},
                }
    RETURN_TYPES = ("INT", "FLOAT", "STRING",)
    OUTPUT_NODE = True
    FUNCTION = "evaluate"
    CATEGORY = "Efficiency Nodes/Simple Eval"

    def evaluate(self, python_expression, print_to_console, a=0, b=0, c=0):
        # simple_eval doesn't require the result to be converted to a string
        result = simpleeval.simple_eval(python_expression, names={'a': a, 'b': b, 'c': c})
        int_result = int(result)
        float_result = float(result)
        string_result = str(result)
        if print_to_console == "True":
            print("\n\033[31mEvaluate Integers:\033[0m")
            print(f"\033[90m{{a = {a} , b = {b} , c = {c}}} \033[0m")
            print(f"{python_expression} = \033[92m INT: " + str(int_result) + " , FLOAT: " + str(
                float_result) + ", STRING: " + string_result + "\033[0m")
        return (int_result, float_result, string_result,)

# TSC Evaluate Floats (https://github.com/danthedeckie/simpleeval)
class TSC_EvaluateFloats:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "python_expression": ("STRING", {"default": "((a + b) - c) / 2", "multiline": False}),
                    "print_to_console": (["False", "True"],),},
                "optional": {
                    "a": ("FLOAT", {"default": 0, "min": -sys.float_info.max, "max": sys.float_info.max, "step": 1}),
                    "b": ("FLOAT", {"default": 0, "min": -sys.float_info.max, "max": sys.float_info.max, "step": 1}),
                    "c": ("FLOAT", {"default": 0, "min": -sys.float_info.max, "max": sys.float_info.max, "step": 1}),},
                }
    RETURN_TYPES = ("INT", "FLOAT", "STRING",)
    OUTPUT_NODE = True
    FUNCTION = "evaluate"
    CATEGORY = "Efficiency Nodes/Simple Eval"

    def evaluate(self, python_expression, print_to_console, a=0, b=0, c=0):
        # simple_eval doesn't require the result to be converted to a string
        result = simpleeval.simple_eval(python_expression, names={'a': a, 'b': b, 'c': c})
        int_result = int(result)
        float_result = float(result)
        string_result = str(result)
        if print_to_console == "True":
            print("\n\033[31mEvaluate Floats:\033[0m")
            print(f"\033[90m{{a = {a} , b = {b} , c = {c}}} \033[0m")
            print(f"{python_expression} = \033[92m INT: " + str(int_result) + " , FLOAT: " + str(
                float_result) + ", STRING: " + string_result + "\033[0m")
        return (int_result, float_result, string_result,)

# TSC Evaluate Strings (https://github.com/danthedeckie/simpleeval)
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
    CATEGORY = "Efficiency Nodes/Simple Eval"

    def evaluate(self, python_expression, print_to_console, a="", b="", c=""):
        variables = {'a': a, 'b': b, 'c': c}  # Define the variables for the expression
        
        functions = simpleeval.DEFAULT_FUNCTIONS.copy()
        functions.update({"len": len}) # Add the functions for the expression
        
        result = simpleeval.simple_eval(python_expression, names=variables, functions=functions)
        if print_to_console == "True":
            print("\n\033[31mEvaluate Strings:\033[0m")
            print(f"\033[90ma = {a} \nb = {b} \nc = {c}\033[0m")
            print(f"{python_expression} = \033[92m" + str(result) + "\033[0m")
        return (str(result),)  # Convert result to a string before returning

# TSC Simple Eval Examples (https://github.com/danthedeckie/simpleeval)
class TSC_EvalExamples:
    filepath = os.path.join(my_dir, 'workflows', 'SimpleEval_Node_Examples.txt')
    with open(filepath, 'r') as file:
        examples = file.read()
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { "models_text": ("STRING", {"default": cls.examples ,"multiline": True}),},}
    RETURN_TYPES = ()
    CATEGORY = "Efficiency Nodes/Simple Eval"

# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "KSampler (Efficient)": TSC_KSampler,
    "Efficient Loader": TSC_EfficientLoader,
    "LoRA Stacker": TSC_LoRA_Stacker,
    "LoRA Stacker Adv.": TSC_LoRA_Stacker_Adv,
    "XY Plot": TSC_XYplot,
    "XY Input: Seeds++ Batch": TSC_XYplot_SeedsBatch,
    "XY Input: Steps": TSC_XYplot_Steps,
    "XY Input: CFG Scale": TSC_XYplot_CFG,
    "XY Input: Sampler": TSC_XYplot_Sampler,
    "XY Input: Scheduler": TSC_XYplot_Scheduler,
    "XY Input: Denoise": TSC_XYplot_Denoise,
    "XY Input: VAE": TSC_XYplot_VAE,
    "XY Input: Positive Prompt S/R": TSC_XYplot_PromptSR_Positive,
    "XY Input: Negative Prompt S/R": TSC_XYplot_PromptSR_Negative,
    "XY Input: Checkpoint": TSC_XYplot_Checkpoint,
    "XY Input: Checkpoint Merge": TSC_XYplot_Checkpoint_Merge,
    "XY Input: Checkpoint Merge Adv.": TSC_XYplot_Checkpoint_Merge_Advanced,
    "XY Input: Clip Skip": TSC_XYplot_ClipSkip,
    "XY Input: LoRA": TSC_XYplot_LoRA,
    "XY Input: LoRA Adv.": TSC_XYplot_LoRA_Adv,
    "XY Input: LoRA Stacks": TSC_XYplot_LoRA_Stacks,
    "XY Input: Manual XY Entry": TSC_XYplot_Manual_XY_Entry,
    "Manual XY Entry Info": TSC_XYplot_Manual_XY_Entry_Info,
    "Join XY Inputs of Same Type": TSC_XYplot_JoinInputs,
    "Image Overlay": TSC_ImageOverlay,
    "Evaluate Integers": TSC_EvaluateInts,
    "Evaluate Floats": TSC_EvaluateFloats,
    "Evaluate Strings": TSC_EvaluateStrs,
    "Simple Eval Examples": TSC_EvalExamples
}
