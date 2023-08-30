# Efficiency Nodes - A collection of my ComfyUI custom nodes to help streamline workflows and reduce total node count.
#  by Luciano Cirino (Discord: TSC#9184) - April 2023

from nodes import KSampler, KSamplerAdvanced, CLIPSetLastLayer, CLIPTextEncode, ControlNetApply

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

# Import ComfyUI files
import comfy.samplers
import comfy.sd
import comfy.utils
import comfy.latent_formats

# Import my utility functions
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
                "optional": {"lora_stack": ("LORA_STACK", ),
                             "cnet_stack": ("CONTROL_NET_STACK",)},
                "hidden": { "prompt": "PROMPT",
                            "my_unique_id": "UNIQUE_ID",},
                }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "DEPENDENCIES",)
    RETURN_NAMES = ("MODEL", "CONDITIONING+", "CONDITIONING-", "LATENT", "VAE", "CLIP", "DEPENDENCIES", )
    FUNCTION = "efficientloader"
    CATEGORY = "Efficiency Nodes/Loaders"

    def efficientloader(self, ckpt_name, vae_name, clip_skip, lora_name, lora_model_strength, lora_clip_strength,
                        positive, negative, empty_latent_width, empty_latent_height, batch_size, lora_stack=None,
                        cnet_stack=None, prompt=None, my_unique_id=None):

        # Create Empty Latent
        latent = torch.zeros([batch_size, 4, empty_latent_height // 8, empty_latent_width // 8]).cpu()

        # Clean globally stored objects
        globals_cleanup(prompt)

        # Retrieve cache numbers
        vae_cache, ckpt_cache, lora_cache = get_cache_numbers("Efficient Loader")

        if lora_name != "None" or lora_stack:
            # Initialize an empty list to store LoRa parameters.
            lora_params = []

            # Check if lora_name is not the string "None" and if so, add its parameters.
            if lora_name != "None":
                lora_params.append((lora_name, lora_model_strength, lora_clip_strength))

            # If lora_stack is not None or an empty list, extend lora_params with its items.
            if lora_stack:
                lora_params.extend(lora_stack)

            # Load LoRa(s)
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
        clip = CLIPSetLastLayer().set_last_layer(clip, clip_skip)[0]

        # Encode Conditionings
        positive_encoded = CLIPTextEncode().encode(clip, positive)[0]
        negative_encoded = CLIPTextEncode().encode(clip, negative)[0]

        # Recursively apply Control Net to the positive encoding for each entry in the stack
        if cnet_stack is not None:
            for control_net_tuple in cnet_stack:
                control_net, image, strength = control_net_tuple
                positive_encoded = ControlNetApply().apply_controlnet(positive_encoded, control_net, image, strength)[0]

        # Data for XY Plot
        dependencies = (vae_name, ckpt_name, clip, clip_skip, positive, negative, lora_params, cnet_stack)

        return (model, positive_encoded, negative_encoded, {"samples":latent}, vae, clip, dependencies, )

########################################################################################################################
# TSC LoRA Stacker
class TSC_LoRA_Stacker:

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {"required": {
            "lora_name_1": (loras,),
            "lora_wt_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_name_2": (loras,),
            "lora_wt_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_name_3": (loras,),
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

#=======================================================================================================================
# TSC LoRA Stacker Advanced
class TSC_LoRA_Stacker_Adv:

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {"required": {
            "lora_name_1": (loras,),
            "model_str_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "clip_str_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_name_2": (loras,),
            "model_str_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "clip_str_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            "lora_name_3": (loras,),
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

#=======================================================================================================================
# TSC Control Net Stacker
class TSC_Control_Net_Stacker:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"control_net": ("CONTROL_NET",),
                         "image": ("IMAGE",),
                         "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01})},
                "optional": {"cnet_stack": ("CONTROL_NET_STACK",)},
                }

    RETURN_TYPES = ("CONTROL_NET_STACK",)
    RETURN_NAMES = ("CNET_STACK",)
    FUNCTION = "control_net_stacker"
    CATEGORY = "Efficiency Nodes/Misc"

    def control_net_stacker(self, control_net, image, strength, cnet_stack=None):
        # If control_net_stack is None, initialize as an empty list
        if cnet_stack is None:
            cnet_stack = []

        # Extend the control_net_stack with the new tuple
        cnet_stack.extend([(control_net, image, strength)])

        return (cnet_stack,)

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
                     "preview_method": (["auto", "latent2rgb", "taesd", "none"],),
                     "vae_decode": (["true", "true (tiled)", "false", "output only", "output only (tiled)"],),
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
               latent_image, preview_method, vae_decode, denoise=1.0, prompt=None, extra_pnginfo=None, my_unique_id=None,
               optional_vae=(None,), script=None, add_noise=None, start_at_step=None, end_at_step=None,
               return_with_leftover_noise=None):

        # Rename the vae variable
        vae = optional_vae

        # Find instance of ksampler type
        if add_noise == None:
            ksampler_adv_flag = False
        else:
            ksampler_adv_flag = True
            
        # If vae is not connected, disable vae decoding
        if vae == (None,) and vae_decode != "false":
            print('\033[33mKSampler(Efficient) Warning:\033[0m No vae input detected, proceeding as if vae_decode was false.\n')
            vae_decode = "false"

        # Functions for previewing images in Ksampler
        def map_filename(filename):
            prefix_len = len(os.path.basename(filename_prefix))
            prefix = filename[:prefix_len + 1]
            try:
                digits = int(filename[prefix_len + 1:].split('_')[0])
            except:
                digits = 0
            return (digits, prefix)

        def compute_vars(images,input):
            input = input.replace("%width%", str(images[0].shape[1]))
            input = input.replace("%height%", str(images[0].shape[0]))
            return input

        def preview_image(images, filename_prefix):

            if images == list():
                return list()

            filename_prefix = compute_vars(images,filename_prefix)

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

        def vae_decode_latent(latent, vae_decode):
            return vae.decode_tiled(latent).cpu() if "tiled" in vae_decode else vae.decode(latent).cpu()

        # Clean globally stored objects of non-existant nodes
        globals_cleanup(prompt)

        # Convert ID string to an integer
        my_unique_id = int(my_unique_id)

        # Init last_preview_images
        if get_value_by_id("preview_images", my_unique_id) is None:
            last_preview_images = list()
        else:
            last_preview_images = get_value_by_id("preview_images", my_unique_id)

        # Init last_latent
        if get_value_by_id("latent", my_unique_id) is None:
            last_latent = latent_image
        else:
            last_latent = {"samples": None}
            last_latent["samples"] = get_value_by_id("latent", my_unique_id)

        # Init last_output_images
        if get_value_by_id("output_images", my_unique_id) == None:
            last_output_images = TSC_KSampler.empty_image
        else:
            last_output_images = get_value_by_id("output_images", my_unique_id)

        # Define filename_prefix
        filename_prefix = "KSeff_{:02d}".format(my_unique_id)

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Check the current sampler state
        if sampler_state == "Sample":

            # Store the global preview method
            previous_preview_method = global_preview_method()

            # Change the global preview method temporarily during sampling
            set_preview_method(preview_method)

            # Define commands arguments to send to front-end via websocket
            if preview_method != "none" and "true" in vae_decode:
                send_command_to_frontend(startListening=True, maxCount=steps-1, sendBlob=False)

            # Sample the latent_image(s) using the Comfy KSampler nodes
            if ksampler_adv_flag == False:
                samples = KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)
            else:
                samples = KSamplerAdvanced().sample(model, add_noise, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                latent_image, start_at_step, end_at_step, return_with_leftover_noise, denoise=1.0)

            # Extract the latent samples from the returned samples dictionary
            latent = samples[0]["samples"]

            # Cache latent samples in the 'last_helds' dictionary "latent"
            update_value_by_id("latent", my_unique_id, latent)

            # Define node output images & next Hold's vae_decode behavior
            output_images = node_images = get_latest_image() ###
            if vae_decode == "false":
                update_value_by_id("vae_decode_flag", my_unique_id, True)
                if preview_method == "none" or output_images == list():
                    output_images = TSC_KSampler.empty_image
            else:
                update_value_by_id("vae_decode_flag", my_unique_id, False)
                decoded_image = vae_decode_latent(latent, vae_decode)
                output_images = node_images = decoded_image

            # Cache output images to global 'last_helds' dictionary "output_images"
            update_value_by_id("output_images", my_unique_id, output_images)

            # Generate preview_images (PIL)
            preview_images = preview_image(node_images, filename_prefix)

            # Cache node preview images to global 'last_helds' dictionary "preview_images"
            update_value_by_id("preview_images", my_unique_id, preview_images)

            # Set xy_plot_flag to 'False' and set the stored (if any) XY Plot image tensor to 'None'
            update_value_by_id("xy_plot_flag", my_unique_id, False)
            update_value_by_id("xy_plot_image", my_unique_id, None)

            if "output only" in vae_decode:
                preview_images = list()

            if preview_method != "none":
                # Send message to front-end to revoke the last blob image from browser's memory (fixes preview duplication bug)
                send_command_to_frontend(startListening=False)

            result = (model, positive, negative, {"samples": latent}, vae, output_images,)
            return result if not preview_images and preview_method != "none" else {"ui": {"images": preview_images}, "result": result}

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # If the sampler state is "Hold"
        elif sampler_state == "Hold":
            output_images = last_output_images
            preview_images = last_preview_images if "true" in vae_decode else list()

            if get_value_by_id("vae_decode_flag", my_unique_id):
                if "true" in vae_decode or "output only" in vae_decode:
                    output_images = node_images = vae_decode_latent(last_latent["samples"], vae_decode)
                    update_value_by_id("vae_decode_flag", my_unique_id, False)
                    update_value_by_id("output_images", my_unique_id, output_images)
                    preview_images = preview_image(node_images, filename_prefix)
                    update_value_by_id("preview_images", my_unique_id, preview_images)
                    if "output only" in vae_decode:
                        preview_images = list()

            # Check if holding an XY Plot image
            elif get_value_by_id("xy_plot_flag", my_unique_id):
                # Extract the name of the node connected to script input
                script_node_name, _ = extract_node_info(prompt, my_unique_id, 'script')
                if script_node_name == "XY Plot":
                    # Extract the 'xyplot_as_output_image' input parameter from the connected xy_plot
                    _, _, _, _, _, _, _, xyplot_as_output_image, _ = script
                    if xyplot_as_output_image == True:
                        output_images = get_value_by_id("xy_plot_image", my_unique_id)
                    else:
                        output_images = get_value_by_id("output_images", my_unique_id)
                    preview_images = last_preview_images
                else:
                    output_images = last_output_images
                    preview_images = last_preview_images if "true" in vae_decode else list()

            return {"ui": {"images": preview_images},
                    "result": (model, positive, negative, {"samples": last_latent["samples"]}, vae, output_images,)}

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        elif sampler_state == "Script":

            # Store name of connected node to script input
            script_node_name, script_node_id = extract_node_info(prompt, my_unique_id, 'script')

            # If no valid script input connected, error out
            if script == None or script == (None,) or script_node_name!="XY Plot":
                if script_node_name!="XY Plot":
                    print('\033[31mKSampler(Efficient) Error:\033[0m No valid script input detected')
                return {"ui": {"images": list()},
                        "result": (model, positive, negative, last_latent, vae, TSC_KSampler.empty_image,)}

            # If no vae connected, throw errors
            if vae == (None,):
                print('\033[31mKSampler(Efficient) Error:\033[0m VAE input must be connected in order to use the XY Plotter.')
                return {"ui": {"images": list()},
                        "result": (model, positive, negative, last_latent, vae, TSC_KSampler.empty_image,)}

            # If vae_decode is not set to true, print message that changing it to true
            if "true" not in vae_decode:
                print('\033[33mKSampler(Efficient) Warning:\033[0m VAE decoding must be set to \'true\''
                    ' for XY Plot script, proceeding as if \'true\'.\n')

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
                clip_skip = None
                positive_prompt = None
                negative_prompt = None
                lora_stack = None
                cnet_stack = None

                # Unpack script Tuple (X_type, X_value, Y_type, Y_value, grid_spacing, Y_label_orientation, dependencies)
                X_type, X_value, Y_type, Y_value, grid_spacing, Y_label_orientation, cache_models, xyplot_as_output_image, dependencies = script

                #_______________________________________________________________________________________________________
                # The below section is used to check wether the XY_type is allowed for the Ksampler instance being used.
                # If not the correct type, this section will abort the xy plot script.
                
                # Define disallowed XY_types for each ksampler type
                disallowed_ksampler_types = ["AddNoise", "ReturnNoise", "StartStep", "EndStep"]
                disallowed_ksamplerAdv_types = ["Denoise"]

                # Check against the relevant disallowed values array based on ksampler_adv_flag
                current_disallowed_values = disallowed_ksamplerAdv_types if ksampler_adv_flag else disallowed_ksampler_types

                # Print error and exit
                if X_type in current_disallowed_values or Y_type in current_disallowed_values:
                    error_prefix = '\033[31mKSampler Adv.(Efficient) Error:\033[0m' \
                        if ksampler_adv_flag else '\033[31mKSampler(Efficient) Error:\033[0m'

                    # Determine which type failed
                    failed_type = None
                    if X_type in current_disallowed_values:
                        failed_type = f"X_type: '{X_type}'"
                    if Y_type in current_disallowed_values:
                        if failed_type:
                            failed_type += f" and Y_type: '{Y_type}'"
                        else:
                            failed_type = f"Y_type: '{Y_type}'"

                    # Suggest alternative KSampler
                    suggested_ksampler = "KSampler(Efficient)" if ksampler_adv_flag else "KSampler Adv.(Efficient)"

                    print(f"{error_prefix} Invalid value for {failed_type}. Use {suggested_ksampler} for this XY Plot type."
                          f"\nDisallowed XY_types for this KSampler are: {', '.join(current_disallowed_values)}.")

                    return {"ui": {"images": list()},
                            "result": (model, positive, negative, last_latent, vae, TSC_KSampler.empty_image,)}

                #_______________________________________________________________________________________________________
                # Unpack Effficient Loader dependencies
                if dependencies is not None:
                    vae_name, ckpt_name, clip, clip_skip, positive_prompt, negative_prompt, lora_stack, cnet_stack = dependencies

                # Helper function to process printout values
                def process_xy_for_print(value, replacement, type_):
                    if isinstance(value, tuple) and type_ == "Scheduler":
                        return value[0]  # Return only the first entry of the tuple
                    elif type_ == "ControlNetStr" and isinstance(value, list):
                        # Extract the first inner array of each entry and then get the third entry of its tuple
                        return [round(inner_list[0][2], 3) for inner_list in value if
                                inner_list and isinstance(inner_list[0], tuple) and len(inner_list[0]) == 3]
                    elif isinstance(value, tuple):
                        return tuple(replacement if v is None else v for v in value)
                    else:
                        return replacement if value is None else value

                # Determine the replacements based on X_type and Y_type
                replacement_X = scheduler if X_type == 'Sampler' else clip_skip if X_type == 'Checkpoint' else None
                replacement_Y = scheduler if Y_type == 'Sampler' else clip_skip if Y_type == 'Checkpoint' else None

                # Process X_value and Y_value
                X_value_processed = process_xy_for_print(X_value, replacement_X, X_type)
                Y_value_processed = process_xy_for_print(Y_value, replacement_Y, Y_type)

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
                    lora_dict = [(lora_stack, ckpt) for ckpt in ckpt_dict for lora_stack in lora_dict]
                # If lora_dict is not empty and ckpt_dict is empty, insert ckpt_name into each tuple in lora_dict
                elif lora_dict:
                    lora_dict = [(lora_stack, ckpt_name) for lora_stack in lora_dict]

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
                def define_variable(var_type, var, add_noise, seed, steps, start_at_step, end_at_step,
                                    return_with_leftover_noise, cfg, sampler_name, scheduler, denoise, vae_name, ckpt_name,
                                    clip_skip, positive_prompt, negative_prompt, lora_stack, cnet_stack, var_label, num_label):

                    # Define default max label size limit
                    max_label_len = 36

                    # If var_type is "AddNoise", update 'add_noise' with 'var', and generate text label
                    if var_type == "AddNoise":
                        add_noise = var
                        text = f"AddNoise: {add_noise}"

                    # If var_type is "Seeds++ Batch", generate text label
                    elif var_type == "Seeds++ Batch":
                        text = f"Seed: {seed}"

                    # If var_type is "Steps", update 'steps' with 'var' and generate text label
                    elif var_type == "Steps":
                        steps = var
                        text = f"Steps: {steps}"

                    # If var_type is "StartStep", update 'start_at_step' with 'var' and generate text label
                    elif var_type == "StartStep":
                        start_at_step = var
                        text = f"StartStep: {start_at_step}"

                    # If var_type is "EndStep", update 'end_at_step' with 'var' and generate text label
                    elif var_type == "EndStep":
                        end_at_step = var
                        text = f"EndStep: {end_at_step}"

                    # If var_type is "ReturnNoise", update 'return_with_leftover_noise' with 'var', and generate text label
                    elif var_type == "ReturnNoise":
                        return_with_leftover_noise = var
                        text = f"ReturnNoise: {return_with_leftover_noise}"

                    # If var_type is "CFG Scale", update cfg with var and generate text label
                    elif var_type == "CFG Scale":
                        cfg = var
                        text = f"CFG: {round(cfg,2)}"

                    # If var_type is "Sampler", update sampler_name and scheduler with var, and generate text label
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
                        text = f"Denoise: {round(denoise, 2)}"

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

                    elif var_type == "Clip Skip":
                        clip_skip = (var, clip_skip[1])
                        text = f"ClipSkip ({clip_skip[0]})"

                    elif var_type == "LoRA":
                        lora_stack = var
                        max_label_len = 30 + (12 * (len(lora_stack)-1))
                        if len(lora_stack) == 1:
                            lora_name, lora_model_wt, lora_clip_wt = lora_stack[0]
                            lora_filename = os.path.splitext(os.path.basename(lora_name))[0]
                            lora_model_wt = format(float(lora_model_wt), ".2f").rstrip('0').rstrip('.')
                            lora_clip_wt = format(float(lora_clip_wt), ".2f").rstrip('0').rstrip('.')
                            lora_filename = lora_filename[:max_label_len - len(f"LoRA: ({lora_model_wt})")]
                            if lora_model_wt == lora_clip_wt:
                                text = f"LoRA: {lora_filename}({lora_model_wt})"
                            else:
                                text = f"LoRA: {lora_filename}({lora_model_wt},{lora_clip_wt})"
                        elif len(lora_stack) > 1:
                            lora_filenames = [os.path.splitext(os.path.basename(lora_name))[0] for lora_name, _, _ in lora_stack]
                            lora_details = [(format(float(lora_model_wt), ".2f").rstrip('0').rstrip('.'),
                                             format(float(lora_clip_wt), ".2f").rstrip('0').rstrip('.')) for _, lora_model_wt, lora_clip_wt in lora_stack]
                            non_name_length = sum(len(f"({lora_details[i][0]},{lora_details[i][1]})") + 2 for i in range(len(lora_stack)))
                            available_space = max_label_len - non_name_length
                            max_name_length = available_space // len(lora_stack)
                            lora_filenames = [filename[:max_name_length] for filename in lora_filenames]
                            text_elements = [f"{lora_filename}({lora_details[i][0]})" if lora_details[i][0] == lora_details[i][1] else f"{lora_filename}({lora_details[i][0]},{lora_details[i][1]})" for i, lora_filename in enumerate(lora_filenames)]
                            text = " ".join(text_elements)
                            
                    elif var_type == "ControlNetStr":
                        cnet_stack = var
                        text = f'ControlNetStr: {round(cnet_stack[0][2], 3)}'

                    elif var_type == "XY_Capsule":
                        text = var.getLabel()

                    else: # No matching type found
                        text=""

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
                    return add_noise, steps, start_at_step, end_at_step, return_with_leftover_noise, cfg,\
                        sampler_name, scheduler, denoise, vae_name, ckpt_name, clip_skip, \
                        positive_prompt, negative_prompt, lora_stack, cnet_stack, var_label

                #_______________________________________________________________________________________________________
                # The function below is used to smartly load Checkpoint/LoRA/VAE models between generations.
                def define_model(model, clip, positive, negative, positive_prompt, negative_prompt, clip_skip, vae,
                                 vae_name, ckpt_name, lora_stack, cnet_stack, index, types, script_node_id, cache):
        
                    # Encode prompt, apply clip_skip, and Control Net (if given). Return new conditioning.
                    def encode_prompt(positive_prompt, negative_prompt, clip, clip_skip, cnet_stack):
                        clip = CLIPSetLastLayer().set_last_layer(clip, clip_skip)[0]
                        positive_encoded = CLIPTextEncode().encode(clip, positive_prompt)[0]
                        negative_encoded = CLIPTextEncode().encode(clip, negative_prompt)[0]

                        # Recursively apply Control Net to the positive encoding for each entry in the stack
                        if cnet_stack is not None:
                            for control_net_tuple in cnet_stack:
                                control_net, image, strength = control_net_tuple
                                positive_encoded = \
                                ControlNetApply().apply_controlnet(positive_encoded, control_net, image, strength)[0]

                        return positive_encoded, negative_encoded

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
                        if lora_stack is None:
                            model, clip, _ = load_checkpoint(ckpt_name, script_node_id, output_vae=False, cache=cache[1])
                        else: # Load Efficient Loader LoRA
                            model, clip = load_lora(lora_stack, ckpt_name, script_node_id,
                                                    cache=None, ckpt_cache=cache[1])
                        encode = True

                    # Load LoRA if required
                    elif (X_type == "LoRA" and index == 0):
                        # Don't cache Checkpoints
                        model, clip = load_lora(lora_stack, ckpt_name, script_node_id, cache=cache[2])
                        encode = True
                        
                    elif Y_type == "LoRA":  # X_type must be Checkpoint, so cache those as defined
                        model, clip = load_lora(lora_stack, ckpt_name, script_node_id,
                                                cache=None, ckpt_cache=cache[1])
                        encode = True

                    # Encode Prompt if required
                    prompt_types = ["Positive Prompt S/R", "Negative Prompt S/R", "Clip Skip", "ControlNetStr"]
                    if (X_type in prompt_types and index == 0) or Y_type in prompt_types:
                        encode = True

                    # Encode prompt if encode == True
                    if encode == True:
                        positive, negative = encode_prompt(positive_prompt[0], negative_prompt[0], clip, clip_skip, cnet_stack)
                        
                    return model, positive, negative, vae

                # ______________________________________________________________________________________________________
                # The below function is used to generate the results based on all the processed variables
                def process_values(model, add_noise, seed, steps, start_at_step, end_at_step, return_with_leftover_noise,
                                   cfg, sampler_name, scheduler, positive, negative, latent_image, denoise, vae, vae_decode,
                                   ksampler_adv_flag, latent_list=[], image_tensor_list=[], image_pil_list=[], xy_capsule=None):

                    capsule_result = None
                    if xy_capsule is not None:
                        capsule_result = xy_capsule.get_result(model, clip, vae)
                        if capsule_result is not None:
                            image, latent = capsule_result
                            latent_list.append(latent['samples'])

                    if capsule_result is None:
                        if preview_method != "none":
                            send_command_to_frontend(startListening=True, maxCount=steps - 1, sendBlob=False)

                        # Sample using the Comfy KSampler nodes
                        if ksampler_adv_flag == False:
                            samples = KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler,
                                                        positive, negative, latent_image, denoise=denoise)
                        else:
                            samples = KSamplerAdvanced().sample(model, add_noise, seed, steps, cfg, sampler_name, scheduler,
                                                                positive, negative, latent_image, start_at_step, end_at_step,
                                                                return_with_leftover_noise, denoise=1.0)

                        # Decode images and store
                        latent = samples[0]["samples"]

                        # Add the latent tensor to the tensors list
                        latent_list.append(latent)

                        # Decode the latent tensor
                        image = vae_decode_latent(latent, vae_decode)

                        if xy_capsule is not None:
                            xy_capsule.set_result(image, latent)

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

                # Store the global preview method
                previous_preview_method = global_preview_method()

                # Change the global preview method temporarily during this node's execution
                set_preview_method(preview_method)

                original_model = model.clone()
                original_clip = clip.clone()

                # Fill Plot Rows (X)
                for X_index, X in enumerate(X_value):
                    model = original_model.clone()
                    clip = original_clip.clone()

                    # Seed control based on loop index during Batch
                    if X_type == "Seeds++ Batch":
                        # Update seed based on the inner loop index
                        seed_updated = seed + X_index

                    # Define X parameters and generate labels
                    add_noise, steps, start_at_step, end_at_step, return_with_leftover_noise, cfg,\
                        sampler_name, scheduler, denoise, vae_name, ckpt_name, clip_skip,\
                        positive_prompt, negative_prompt, lora_stack, cnet_stack, X_label = \
                        define_variable(X_type, X, add_noise, seed_updated, steps, start_at_step, end_at_step,
                                        return_with_leftover_noise, cfg, sampler_name, scheduler, denoise, vae_name,
                                        ckpt_name, clip_skip, positive_prompt, negative_prompt, lora_stack, cnet_stack, X_label, len(X_value))


                    if X_type != "Nothing" and Y_type == "Nothing":
                        if X_type == "XY_Capsule":
                            model, clip, vae = X.pre_define_model(model, clip, vae)

                        # Models & Conditionings
                        model, positive, negative , vae = \
                            define_model(model, clip, positive, negative, positive_prompt, negative_prompt, clip_skip[0], vae,
                                         vae_name, ckpt_name, lora_stack, cnet_stack, 0, types, script_node_id, cache)

                        xy_capsule = None
                        if X_type == "XY_Capsule":
                            xy_capsule = X

                        # Generate Results
                        latent_list, image_tensor_list, image_pil_list = \
                            process_values(model, add_noise, seed_updated, steps, start_at_step, end_at_step,
                                           return_with_leftover_noise, cfg, sampler_name, scheduler[0],
                                           positive, negative, latent_image, denoise, vae, vae_decode, ksampler_adv_flag, xy_capsule=xy_capsule)

                    elif X_type != "Nothing" and Y_type != "Nothing":
                        # Seed control based on loop index during Batch
                        for Y_index, Y in enumerate(Y_value):
                            model = original_model.clone()
                            clip = original_clip.clone()

                            if Y_type == "XY_Capsule" and X_type == "XY_Capsule":
                                Y.set_x_capsule(X)

                            if Y_type == "Seeds++ Batch":
                                # Update seed based on the inner loop index
                                seed_updated = seed + Y_index

                            # Define Y parameters and generate labels
                            add_noise, steps, start_at_step, end_at_step, return_with_leftover_noise, cfg,\
                                sampler_name, scheduler, denoise, vae_name, ckpt_name, clip_skip,\
                                positive_prompt, negative_prompt, lora_stack, cnet_stack, Y_label = \
                                define_variable(Y_type, Y, add_noise, seed_updated, steps, start_at_step, end_at_step,
                                                return_with_leftover_noise, cfg, sampler_name, scheduler, denoise, vae_name, ckpt_name,
                                                clip_skip, positive_prompt, negative_prompt, lora_stack, cnet_stack, Y_label, len(Y_value))

                            if Y_type == "XY_Capsule":
                                model, clip, vae = Y.pre_define_model(model, clip, vae)
                            elif X_type == "XY_Capsule":
                                model, clip, vae = X.pre_define_model(model, clip, vae)

                            # Models & Conditionings
                            model, positive, negative, vae = \
                                define_model(model, clip, positive, negative, positive_prompt, negative_prompt, clip_skip[0], vae,
                                         vae_name, ckpt_name, lora_stack, cnet_stack, Y_index, types, script_node_id, cache)

                            # Generate Results
                            xy_capsule = None
                            if Y_type == "XY_Capsule":
                                xy_capsule = Y

                            latent_list, image_tensor_list, image_pil_list = \
                                process_values(model, add_noise, seed_updated, steps, start_at_step, end_at_step,
                                               return_with_leftover_noise, cfg, sampler_name, scheduler[0],
                                               positive, negative, latent_image, denoise, vae, vae_decode, ksampler_adv_flag, xy_capsule=xy_capsule)

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
                def print_plot_variables(X_type, Y_type, X_value, Y_value, add_noise, seed, steps, start_at_step, end_at_step,
                                         return_with_leftover_noise, cfg, sampler_name, scheduler, denoise, vae_name, ckpt_name,
                                         clip_skip, lora_stack, ksampler_adv_flag, num_rows, num_cols, latent_height, latent_width):

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

                    def get_lora_name(X_type, Y_type, X_value, Y_value, lora_stack=None):
                        if X_type != "LoRA" and Y_type != "LoRA":
                            if lora_stack:
                                return f"[{', '.join([f'{os.path.splitext(os.path.basename(name))[0]}({round(model_wt, 3)},{round(clip_wt, 3)})' for name, model_wt, clip_wt in lora_stack])}]"
                            else:
                                return None
                        else:
                            return get_lora_sublist_name(X_type,
                                                         X_value) if X_type == "LoRA" else get_lora_sublist_name(Y_type, Y_value) if Y_type == "LoRA" else None

                    def get_lora_sublist_name(lora_type, lora_value):
                        return ", ".join([f"[{', '.join([f'{os.path.splitext(os.path.basename(str(x[0])))[0]}({round(x[1], 3)},{round(x[2], 3)})' for x in sublist])}]"
                                             for sublist in lora_value])

                    # VAE, Checkpoint, Clip Skip, LoRA
                    ckpt_type, clip_skip_type = (X_type, Y_type) if X_type in ["Checkpoint", "Clip Skip"] else (Y_type, X_type)
                    ckpt_values, clip_skip_values = (X_value, Y_value) if X_type in ["Checkpoint", "Clip Skip"] else (Y_value, X_value)

                    clip_skip = get_clip_skip(X_type, Y_type, X_value, Y_value, clip_skip)
                    ckpt_name, clip_skip = get_checkpoint_name(ckpt_type, ckpt_values, clip_skip_type, clip_skip_values, ckpt_name, clip_skip)
                    vae_name = get_vae_name(X_type, Y_type, X_value, Y_value, vae_name)
                    lora_name = get_lora_name(X_type, Y_type, X_value, Y_value, lora_stack)

                    # AddNoise
                    add_noise = ", ".join(map(str, X_value)) if X_type == "AddNoise" else ", ".join(
                        map(str, Y_value)) if Y_type == "AddNoise" else add_noise

                    # Seeds++ Batch
                    seed_list = [seed + x for x in X_value] if X_type == "Seeds++ Batch" else\
                        [seed + y for y in Y_value] if Y_type == "Seeds++ Batch" else [seed]
                    seed = ", ".join(map(str, seed_list))

                    # Steps
                    steps = ", ".join(map(str, X_value)) if X_type == "Steps" else ", ".join(
                        map(str, Y_value)) if Y_type == "Steps" else steps

                    # StartStep
                    start_at_step = ", ".join(map(str, X_value)) if X_type == "StartStep" else ", ".join(
                        map(str, Y_value)) if Y_type == "StartStep" else start_at_step

                    # EndStep
                    end_at_step = ", ".join(map(str, X_value)) if X_type == "EndStep" else ", ".join(
                        map(str, Y_value)) if Y_type == "EndStep" else end_at_step

                    # ReturnNoise
                    return_with_leftover_noise = ", ".join(map(str, X_value)) if X_type == "ReturnNoise" else ", ".join(
                        map(str, Y_value)) if Y_type == "ReturnNoise" else return_with_leftover_noise

                    # CFG
                    cfg = ", ".join(map(str, X_value)) if X_type == "CFG Scale" else ", ".join(
                        map(str, Y_value)) if Y_type == "CFG Scale" else cfg

                    # Sampler/Scheduler
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
                    if ksampler_adv_flag == True:
                        print(f"add_noise: {add_noise}")
                    print(f"seed: {seed}")
                    print(f"steps: {steps}")
                    if ksampler_adv_flag == True:
                        print(f"start_at_step: {start_at_step}")
                        print(f"end_at_step: {end_at_step}")
                        print(f"return_noise: {return_with_leftover_noise}")
                    print(f"cfg: {cfg}")
                    if scheduler == "_":
                        print(f"sampler(schr): {sampler_name}")
                    else:
                        print(f"sampler: {sampler_name}")
                        print(f"scheduler: {scheduler}")
                    if ksampler_adv_flag == False:
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

                    if X_type == "ControlNetStr":
                        control_net_str = [str(round(inner_list[0][2], 3)) for inner_list in X_value if
                                           isinstance(inner_list, list) and
                                           inner_list and isinstance(inner_list[0], tuple) and len(inner_list[0]) >= 3]
                        print(f"control_net_str: {', '.join(control_net_str)}")
                    elif Y_type == "ControlNetStr":
                        control_net_str = [str(round(inner_list[0][2], 3)) for inner_list in Y_value if
                                           isinstance(inner_list, list) and
                                           inner_list and isinstance(inner_list[0], tuple) and len(inner_list[0]) >= 3]
                        print(f"control_net_str: {', '.join(control_net_str)}")

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
                update_value_by_id("vae_decode_flag", my_unique_id, False)

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
                print_plot_variables(X_type, Y_type, X_value, Y_value, add_noise, seed,  steps, start_at_step, end_at_step,
                                     return_with_leftover_noise, cfg, sampler_name, scheduler, denoise, vae_name, ckpt_name,
                                     clip_skip, lora_stack, ksampler_adv_flag, num_rows, num_cols, latent_height, latent_width)

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

                xy_plot_image = pil2tensor(background)

            # Set xy_plot_flag to 'True' and cache the xy_plot_image tensor
            update_value_by_id("xy_plot_image", my_unique_id, xy_plot_image)
            update_value_by_id("xy_plot_flag", my_unique_id, True)

             # Generate the preview_images and cache results
            preview_images = preview_image(xy_plot_image, filename_prefix)
            update_value_by_id("preview_images", my_unique_id, preview_images)

            # Generate output_images and cache results
            output_images = torch.stack([tensor.squeeze() for tensor in image_tensor_list])
            update_value_by_id("output_images", my_unique_id, output_images)

            # Set the output_image the same as plot image defined by 'xyplot_as_output_image'
            if xyplot_as_output_image == True:
                output_images = xy_plot_image

            # Print cache if set to true
            if cache_models == "True":
                print_loaded_objects_entries(script_node_id, prompt)

            print("-" * 40)  # Print an empty line followed by a separator line

            # Set the preview method back to its original state
            set_preview_method(previous_preview_method)

            if preview_method != "none":
                # Send message to front-end to revoke the last blob image from browser's memory (fixes preview duplication bug)
                send_command_to_frontend(startListening=False)

            return {"ui": {"images": preview_images},
                    "result": (model, positive, negative, {"samples": latent_list}, vae, output_images,)}

#=======================================================================================================================
# TSC KSampler Adv (Efficient)
class TSC_KSamplerAdvanced(TSC_KSampler):

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"sampler_state": (["Sample", "Hold", "Script"],),
                     "model": ("MODEL",),
                     "add_noise": (["enable", "disable"],),
                     "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     "latent_image": ("LATENT",),
                     "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                     "return_with_leftover_noise": (["disable", "enable"],),
                     "preview_method": (["auto", "latent2rgb", "taesd", "none"],),
                     "vae_decode": (["true", "true (tiled)", "false", "output only", "output only (tiled)"],),
                     },
                "optional": {"optional_vae": ("VAE",),
                             "script": ("SCRIPT",), },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID", },
                }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "IMAGE",)
    RETURN_NAMES = ("MODEL", "CONDITIONING+", "CONDITIONING-", "LATENT", "VAE", "IMAGE",)
    OUTPUT_NODE = True
    FUNCTION = "sampleadv"
    CATEGORY = "Efficiency Nodes/Sampling"

    def sampleadv(self, sampler_state, model, add_noise, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative,
               latent_image, start_at_step, end_at_step, return_with_leftover_noise, preview_method, vae_decode,
               prompt=None, extra_pnginfo=None, my_unique_id=None, optional_vae=(None,), script=None):

        return super().sample(sampler_state, model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative,
               latent_image, preview_method, vae_decode, denoise=1.0, prompt=prompt, extra_pnginfo=extra_pnginfo, my_unique_id=my_unique_id,
               optional_vae=optional_vae, script=script, add_noise=add_noise, start_at_step=start_at_step,end_at_step=end_at_step,
                       return_with_leftover_noise=return_with_leftover_noise)

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
        if X_type != "XY_Capsule" and (X_type == Y_type):
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

        # Flip X and Y
        if XY_flip == "True":
            X_type, Y_type = Y_type, X_type
            X_value, Y_value = Y_value, X_value
            
        # Define Ksampler output image behavior
        xyplot_as_output_image = ksampler_output_image == "Plot"

        return ((X_type, X_value, Y_type, Y_value, grid_spacing, Y_label_orientation, cache_models,
                 xyplot_as_output_image, dependencies),)

#=======================================================================================================================
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

#=======================================================================================================================
# TSC XY Plot: Add/Return Noise
class TSC_XYplot_AddReturnNoise:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "XY_type": (["add_noise", "return_with_leftover_noise"],)}
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, XY_type):
        type_mapping = {
            "add_noise": "AddNoise",
            "return_with_leftover_noise": "ReturnNoise"
        }
        xy_type = type_mapping[XY_type]
        xy_value = ["enable", "disable"]
        return ((xy_type, xy_value),)

#=======================================================================================================================
# TSC XY Plot: Step Values
class TSC_XYplot_Steps:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_count": ("INT", {"default": 0, "min": 0, "max": 50}),
                "first_step": ("INT", {"default": 10, "min": 1, "max": 10000}),
                "last_step": ("INT", {"default": 20, "min": 1, "max": 10000}),
            }
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, batch_count, first_step, last_step):
        xy_type = "Steps"
        if batch_count > 1:
            interval = (last_step - first_step) / (batch_count - 1)
            xy_value = [int(first_step + i * interval) for i in range(batch_count)]
        else:
            xy_value = [first_step] if batch_count == 1 else []
        xy_value = list(set(xy_value)) # Remove duplicates
        xy_value.sort()  # Sort in ascending order
        if not xy_value:
            return (None,)
        return ((xy_type, xy_value),)

#=======================================================================================================================
# TSC XY Plot: Start at Step Values
class TSC_XYplot_StartStep:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_count": ("INT", {"default": 0, "min": 0, "max": 50}),
                "first_start_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "last_start_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
            }
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, batch_count, first_start_step, last_start_step):
        xy_type = "StartStep"
        if batch_count > 1:
            step_increment = (last_start_step - first_start_step) / (batch_count - 1)
            xy_value = [int(first_start_step + i * step_increment) for i in range(batch_count)]
        else:
            xy_value = [first_start_step] if batch_count == 1 else []
        xy_value = list(set(xy_value))  # Remove duplicates
        xy_value.sort()  # Sort in ascending order
        if not xy_value:  # Check if the list is empty
            return (None,)
        return ((xy_type, xy_value),)

#=======================================================================================================================
# TSC XY Plot: End at Step Values
class TSC_XYplot_EndStep:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_count": ("INT", {"default": 0, "min": 0, "max": 50}),
                "first_end_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "last_end_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
            }
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, batch_count, first_end_step, last_end_step):
        xy_type = "EndStep"
        if batch_count > 1:
            step_increment = (last_end_step - first_end_step) / (batch_count - 1)
            xy_value = [int(first_end_step + i * step_increment) for i in range(batch_count)]
        else:
            xy_value = [first_end_step] if batch_count == 1 else []
        xy_value = list(set(xy_value))  # Remove duplicates
        xy_value.sort()  # Sort in ascending order
        if not xy_value:  # Check if the list is empty
            return (None,)
        return ((xy_type, xy_value),)

#=======================================================================================================================
# TSC XY Plot: CFG Values
class TSC_XYplot_CFG:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_count": ("INT", {"default": 0, "min": 0, "max": 50}),
                "first_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
                "last_cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0}),
            }
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, batch_count, first_cfg, last_cfg):
        xy_type = "CFG Scale"
        if batch_count > 1:
            interval = (last_cfg - first_cfg) / (batch_count - 1)
            xy_value = [round(first_cfg + i * interval, 2) for i in range(batch_count)]
        else:
            xy_value = [first_cfg] if batch_count == 1 else []
        if not xy_value:
            return (None,)
        return ((xy_type, xy_value),)

#=======================================================================================================================
# TSC XY Plot: Sampler Values
class TSC_XYplot_Sampler:

    @classmethod
    def INPUT_TYPES(cls):
        samplers = ["None"] + comfy.samplers.KSampler.SAMPLERS
        schedulers = ["None"] + comfy.samplers.KSampler.SCHEDULERS
        return {"required": {
                            "sampler_1": (samplers,),
                            "scheduler_1": (schedulers,),
                            "sampler_2": (samplers,),
                            "scheduler_2": (schedulers,),
                            "sampler_3": (samplers,),
                            "scheduler_3": (schedulers,),
                            "sampler_4": (samplers,),
                            "scheduler_4": (schedulers,),
                            "sampler_5": (samplers,),
                            "scheduler_5": (schedulers,),},
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

#=======================================================================================================================
# TSC XY Plot: Scheduler Values
class TSC_XYplot_Scheduler:

    @classmethod
    def INPUT_TYPES(cls):
        schedulers = ["None"] + comfy.samplers.KSampler.SCHEDULERS
        return {"required": {
            "scheduler_1": (schedulers,),
            "scheduler_2": (schedulers,),
            "scheduler_3": (schedulers,),
            "scheduler_4": (schedulers,),
            "scheduler_5": (schedulers,),},
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

#=======================================================================================================================
# TSC XY Plot: Denoise Values
class TSC_XYplot_Denoise:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_count": ("INT", {"default": 0, "min": 0, "max": 50}),
                "first_denoise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "last_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, batch_count, first_denoise, last_denoise):
        xy_type = "Denoise"
        if batch_count > 1:
            interval = (last_denoise - first_denoise) / (batch_count - 1)
            xy_value = [round(first_denoise + i * interval, 2) for i in range(batch_count)]
        else:
            xy_value = [first_denoise] if batch_count == 1 else []
        if not xy_value:
            return (None,)
        return ((xy_type, xy_value),)

#=======================================================================================================================
# TSC XY Plot: VAE Values
class TSC_XYplot_VAE:

    @classmethod
    def INPUT_TYPES(cls):
        vaes = ["None"] + folder_paths.get_filename_list("vae")
        return {"required": {
            "vae_name_1": (vaes,),
            "vae_name_2": (vaes,),
            "vae_name_3": (vaes,),
            "vae_name_4": (vaes,),
            "vae_name_5": (vaes,),},
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

#=======================================================================================================================
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

#=======================================================================================================================
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

#=======================================================================================================================
# TSC XY Plot: Checkpoint Values
class TSC_XYplot_Checkpoint:

    @classmethod
    def INPUT_TYPES(cls):
        checkpoints = ["None"] + folder_paths.get_filename_list("checkpoints")
        return {"required": {
            "ckpt_name_1": (checkpoints,),
            "clip_skip1": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
            "ckpt_name_2": (checkpoints,),
            "clip_skip2": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
            "ckpt_name_3": (checkpoints,),
            "clip_skip3": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
            "ckpt_name_4": (checkpoints,),
            "clip_skip4": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
            "ckpt_name_5": (checkpoints,),
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

#=======================================================================================================================
# TSC XY Plot: Clip Skip
class TSC_XYplot_ClipSkip:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch_count": ("INT", {"default": 0, "min": 0, "max": 50}),
                "first_clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),
                "last_clip_skip": ("INT", {"default": -2, "min": -24, "max": -1, "step": 1}),
            },
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, batch_count, first_clip_skip, last_clip_skip):
        xy_type = "Clip Skip"
        if batch_count > 1:
            clip_skip_increment = (last_clip_skip - first_clip_skip) / (batch_count - 1)
            xy_value = [int(first_clip_skip + i * clip_skip_increment) for i in range(batch_count)]
        else:
            xy_value = [first_clip_skip] if batch_count == 1 else []
        xy_value = list(set(xy_value))  # Remove duplicates
        xy_value.sort()  # Sort in ascending order
        if not xy_value:  # Check if the list is empty
            return (None,)
        return ((xy_type, xy_value),)

#=======================================================================================================================
# TSC XY Plot: LoRA Values
class TSC_XYplot_LoRA:

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {"required": {
                            "model_strengths": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                            "clip_strengths": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                            "lora_name_1": (loras,),
                            "lora_name_2": (loras,),
                            "lora_name_3": (loras,),
                            "lora_name_4": (loras,),
                            "lora_name_5": (loras,)},
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

#=======================================================================================================================
# TSC XY Plot: LoRA Advanced
class TSC_XYplot_LoRA_Adv:

    @classmethod
    def INPUT_TYPES(cls):
        loras = ["None"] + folder_paths.get_filename_list("loras")
        return {"required": {
                            "lora_name_1": (loras,),
                            "model_str_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                            "clip_str_1": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                            "lora_name_2": (loras,),
                            "model_str_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                            "clip_str_2": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                            "lora_name_3": (loras,),
                            "model_str_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                            "clip_str_3": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                            "lora_name_4": (loras,),
                            "model_str_4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                            "clip_str_4": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                            "lora_name_5": (loras,),
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

#=======================================================================================================================
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

#=======================================================================================================================
# TSC XY Plot: Control_Net_Strengths
class TSC_XYplot_Control_Net_Strengths:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "control_net": ("CONTROL_NET",),
                "image": ("IMAGE",),
                "batch_count": ("INT", {"default": 0, "min": 0, "max": 50}),
                "first_strength": ("FLOAT", {"default": 0.0, "min": 0.00, "max": 1.0, "step": 0.01}),
                "last_strength": ("FLOAT", {"default": 1.0, "min": 0.00, "max": 1.0, "step": 0.01}),
            },
            "optional": {"cnet_stack": ("CONTROL_NET_STACK",)},
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, control_net, image, batch_count, first_strength, last_strength, cnet_stack=None):
        xy_type = "ControlNetStr"
        strength_increment = (last_strength - first_strength) / (batch_count - 1) if batch_count > 1 else 0
        xy_value = [[(control_net, image, first_strength + i * strength_increment)] for i in range(batch_count)]

        # If cnet_stack is provided, extend each inner array with its content
        if cnet_stack:
            for inner_list in xy_value:
                inner_list.extend(cnet_stack)

        return ((xy_type, xy_value),)

#=======================================================================================================================
# TSC XY Plot: Manual Entry Notes
class TSC_XYplot_Manual_XY_Entry_Info:

    syntax = "(X/Y_types)     (X/Y_values)\n" \
               "Seeds++ Batch   batch_count\n" \
               "Steps           steps_1;steps_2;...\n" \
               "StartStep       start_step_1;start_step_2;...\n" \
               "EndStep         end_step_1;end_step_2;...\n" \
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

    @classmethod
    def INPUT_TYPES(cls):
        samplers = ";\n".join(comfy.samplers.KSampler.SAMPLERS)
        schedulers = ";\n".join(comfy.samplers.KSampler.SCHEDULERS)
        vaes = ";\n".join(folder_paths.get_filename_list("vae"))
        ckpts = ";\n".join(folder_paths.get_filename_list("checkpoints"))
        loras = ";\n".join(folder_paths.get_filename_list("loras"))
        return {"required": {
            "notes": ("STRING", {"default":
                                    f"_____________SYNTAX_____________\n{cls.syntax}\n\n"
                                    f"____________SAMPLERS____________\n{samplers}\n\n"
                                    f"___________SCHEDULERS___________\n{schedulers}\n\n"
                                    f"_____________VAES_______________\n{vaes}\n\n"
                                    f"___________CHECKPOINTS__________\n{ckpts}\n\n"
                                    f"_____________LORAS______________\n{loras}\n","multiline": True}),},}

    RETURN_TYPES = ()
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

#=======================================================================================================================
# TSC XY Plot: Manual Entry
class TSC_XYplot_Manual_XY_Entry:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "X_type": (["Nothing", "Seeds++ Batch", "Steps", "StartStep", "EndStep", "CFG Scale",
                        "Sampler", "Scheduler", "Denoise", "VAE",
                        "Positive Prompt S/R", "Negative Prompt S/R", "Checkpoint", "Clip Skip", "LoRA"],),
            "X_value": ("STRING", {"default": "", "multiline": True}),
            "Y_type": (["Nothing", "Seeds++ Batch", "Steps", "StartStep", "EndStep", "CFG Scale",
                        "Sampler", "Scheduler", "Denoise", "VAE",
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
            "StartStep": {"min": 0, "max": 10000},
            "EndStep": {"min": 0, "max": 10000},
            "CFG Scale": {"min": 0, "max": 100},
            "Sampler": {"options": comfy.samplers.KSampler.SAMPLERS},
            "Scheduler": {"options": comfy.samplers.KSampler.SCHEDULERS},
            "Denoise": {"min": 0, "max": 1},
            "VAE": {"options": folder_paths.get_filename_list("vae")},
            "Checkpoint": {"options": folder_paths.get_filename_list("checkpoints")},
            "Clip Skip": {"min": -24, "max": -1},
            "LoRA": {"options": folder_paths.get_filename_list("loras"),
                     "model_str": {"min": -10, "max": 10},"clip_str": {"min": -10, "max": 10},},
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
            # __________________________________________________________________________________________________________
            # Start at Step
            elif value_type == "StartStep":
                try:
                    x = int(value)
                    if x < bounds["StartStep"]["min"]:
                        x = bounds["StartStep"]["min"]
                    elif x > bounds["StartStep"]["max"]:
                        x = bounds["StartStep"]["max"]
                    return x
                except ValueError:
                    print(
                        f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid Start Step.")
                    return None
            # __________________________________________________________________________________________________________
            # End at Step
            elif value_type == "EndStep":
                try:
                    x = int(value)
                    if x < bounds["EndStep"]["min"]:
                        x = bounds["EndStep"]["min"]
                    elif x > bounds["EndStep"]["max"]:
                        x = bounds["EndStep"]["max"]
                    return x
                except ValueError:
                    print(
                        f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid End Step.")
                    return None
            # __________________________________________________________________________________________________________
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
            # __________________________________________________________________________________________________________
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
            # __________________________________________________________________________________________________________
            # Scheduler
            elif value_type == "Scheduler":
                if value not in bounds["Scheduler"]["options"]:
                    valid_schedulers = '\n'.join(bounds["Scheduler"]["options"])
                    print(
                        f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid Scheduler. Valid Schedulers are:\n{valid_schedulers}")
                    return None
                else:
                    return value
            # __________________________________________________________________________________________________________
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
            # __________________________________________________________________________________________________________
            # VAE
            elif value_type == "VAE":
                if value not in bounds["VAE"]["options"]:
                    valid_vaes = '\n'.join(bounds["VAE"]["options"])
                    print(f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid VAE. Valid VAEs are:\n{valid_vaes}")
                    return None
                else:
                    return value
            # __________________________________________________________________________________________________________
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
            # __________________________________________________________________________________________________________
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
            # __________________________________________________________________________________________________________
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

            # __________________________________________________________________________________________________________
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

#=======================================================================================================================
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
# NODE MAPPING
NODE_CLASS_MAPPINGS = {
    "KSampler (Efficient)": TSC_KSampler,
    "KSampler Adv. (Efficient)":TSC_KSamplerAdvanced,
    "Efficient Loader": TSC_EfficientLoader,
    "LoRA Stacker": TSC_LoRA_Stacker,
    "LoRA Stacker Adv.": TSC_LoRA_Stacker_Adv,
    "Control Net Stacker": TSC_Control_Net_Stacker,
    "XY Plot": TSC_XYplot,
    "XY Input: Seeds++ Batch": TSC_XYplot_SeedsBatch,
    "XY Input: Add/Return Noise": TSC_XYplot_AddReturnNoise,
    "XY Input: Steps": TSC_XYplot_Steps,
    "XY Input: Start at Step": TSC_XYplot_StartStep,
    "XY Input: End at Step": TSC_XYplot_EndStep,
    "XY Input: CFG Scale": TSC_XYplot_CFG,
    "XY Input: Sampler": TSC_XYplot_Sampler,
    "XY Input: Scheduler": TSC_XYplot_Scheduler,
    "XY Input: Denoise": TSC_XYplot_Denoise,
    "XY Input: VAE": TSC_XYplot_VAE,
    "XY Input: Positive Prompt S/R": TSC_XYplot_PromptSR_Positive,
    "XY Input: Negative Prompt S/R": TSC_XYplot_PromptSR_Negative,
    "XY Input: Checkpoint": TSC_XYplot_Checkpoint,
    "XY Input: Clip Skip": TSC_XYplot_ClipSkip,
    "XY Input: LoRA": TSC_XYplot_LoRA,
    "XY Input: LoRA Adv.": TSC_XYplot_LoRA_Adv,
    "XY Input: LoRA Stacks": TSC_XYplot_LoRA_Stacks,
    "XY Input: Control Net Strengths": TSC_XYplot_Control_Net_Strengths,
    "XY Input: Manual XY Entry": TSC_XYplot_Manual_XY_Entry,
    "Manual XY Entry Info": TSC_XYplot_Manual_XY_Entry_Info,
    "Join XY Inputs of Same Type": TSC_XYplot_JoinInputs,
    "Image Overlay": TSC_ImageOverlay
}
########################################################################################################################
# Simpleeval Nodes
try:
    import simpleeval

    # TSC Evaluate Integers (https://github.com/danthedeckie/simpleeval)
    class TSC_EvaluateInts:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "python_expression": ("STRING", {"default": "((a + b) - c) / 2", "multiline": False}),
                "print_to_console": (["False", "True"],), },
                "optional": {
                    "a": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 1}),
                    "b": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 1}),
                    "c": ("INT", {"default": 0, "min": -48000, "max": 48000, "step": 1}), },
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


    # ==================================================================================================================
    # TSC Evaluate Floats (https://github.com/danthedeckie/simpleeval)
    class TSC_EvaluateFloats:
        @classmethod
        def INPUT_TYPES(cls):
            return {"required": {
                "python_expression": ("STRING", {"default": "((a + b) - c) / 2", "multiline": False}),
                "print_to_console": (["False", "True"],), },
                "optional": {
                    "a": ("FLOAT", {"default": 0, "min": -sys.float_info.max, "max": sys.float_info.max, "step": 1}),
                    "b": ("FLOAT", {"default": 0, "min": -sys.float_info.max, "max": sys.float_info.max, "step": 1}),
                    "c": ("FLOAT", {"default": 0, "min": -sys.float_info.max, "max": sys.float_info.max, "step": 1}), },
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


    # ==================================================================================================================
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
                    "c": ("STRING", {"default": "!", "multiline": False}), }
            }

        RETURN_TYPES = ("STRING",)
        OUTPUT_NODE = True
        FUNCTION = "evaluate"
        CATEGORY = "Efficiency Nodes/Simple Eval"

        def evaluate(self, python_expression, print_to_console, a="", b="", c=""):
            variables = {'a': a, 'b': b, 'c': c}  # Define the variables for the expression

            functions = simpleeval.DEFAULT_FUNCTIONS.copy()
            functions.update({"len": len})  # Add the functions for the expression

            result = simpleeval.simple_eval(python_expression, names=variables, functions=functions)
            if print_to_console == "True":
                print("\n\033[31mEvaluate Strings:\033[0m")
                print(f"\033[90ma = {a} \nb = {b} \nc = {c}\033[0m")
                print(f"{python_expression} = \033[92m" + str(result) + "\033[0m")
            return (str(result),)  # Convert result to a string before returning


    # ==================================================================================================================
    # TSC Simple Eval Examples (https://github.com/danthedeckie/simpleeval)
    class TSC_EvalExamples:
        @classmethod
        def INPUT_TYPES(cls):
            filepath = os.path.join(my_dir, 'workflows', 'SimpleEval_Node_Examples.txt')
            with open(filepath, 'r') as file:
                examples = file.read()
            return {"required": {"models_text": ("STRING", {"default": examples, "multiline": True}), }, }

        RETURN_TYPES = ()
        CATEGORY = "Efficiency Nodes/Simple Eval"

    # ==================================================================================================================
    NODE_CLASS_MAPPINGS.update({"Evaluate Integers": TSC_EvaluateInts})
    NODE_CLASS_MAPPINGS.update({"Evaluate Floats": TSC_EvaluateFloats})
    NODE_CLASS_MAPPINGS.update({"Evaluate Strings": TSC_EvaluateStrs})
    NODE_CLASS_MAPPINGS.update({"Simple Eval Examples": TSC_EvalExamples})

except ImportError:
    print(f"\r\033[33mEfficiency Nodes Warning:\033[0m Failed to import python package 'simpleeval'; related nodes disabled.\n")
