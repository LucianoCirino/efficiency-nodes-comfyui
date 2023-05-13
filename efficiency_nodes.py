# Efficiency Nodes - A collection of my ComfyUI custom nodes to help streamline workflows and reduce total node count.
#  by Luciano Cirino (Discord: TSC#9184) - April 2023

from comfy.sd import ModelPatcher, CLIP, VAE
from nodes import common_ksampler
from torch import Tensor
from PIL import Image, ImageOps, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo
import numpy as np
import torch

from pathlib import Path
import os
import sys
import json
import folder_paths

# Get the absolute path of the parent directory of the current script
my_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the ComfyUI directory
comfy_dir = os.path.abspath(os.path.join(my_dir, '..', '..'))

# Add the ComfyUI directory path to the sys.path list
sys.path.append(comfy_dir)

# Construct the path to the font file
font_path = os.path.join(my_dir, 'arial.ttf')

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

def resolve_input_links(prompt, input_value):
    if isinstance(input_value, list) and len(input_value) == 2:
        input_id, input_index = input_value
        return prompt[input_id]['inputs'][list(prompt[input_id]['inputs'].keys())[input_index]]
    return input_value

# Cache models in RAM
loaded_objects = {
    "ckpt": [], # (ckpt_name, model)
    "clip": [], # (ckpt_name, clip)
    "bvae": [], # (ckpt_name, vae)
    "vae": [],   # (vae_name, vae)
    "lora": [] # (lora_name, model_name, model_lora, clip_lora, strength_model, strength_clip)
}

def print_loaded_objects_entries():
    print("\n" + "-" * 40)  # Print an empty line followed by a separator line

    for key in ["ckpt", "clip", "bvae", "vae", "lora"]:
        print(f"{key.capitalize()} entries:")
        for entry in loaded_objects[key]:
            truncated_name = entry[0][:20]
            print(f"  Name: {truncated_name}\n  Location: {entry[1]}")
            if len(entry) == 3:
                print(f"  Entry[2]: {entry[2]}")
        print("-" * 40)  # Print a separator line

    print("\n")  # Print an empty line


def update_loaded_objects(prompt):
    global loaded_objects

    # Extract all Efficient Loader class type entries
    efficient_loader_entries = [entry for entry in prompt.values() if entry["class_type"] == "Efficient Loader"]

    # Collect all desired model, vae, and lora names
    desired_ckpt_names = set()
    desired_vae_names = set()
    desired_lora_names = set()
    for entry in efficient_loader_entries:
        desired_ckpt_names.add(entry["inputs"]["ckpt_name"])
        desired_vae_names.add(entry["inputs"]["vae_name"])
        desired_lora_names.add(entry["inputs"]["lora_name"])

    # Check and clear unused ckpt, clip, and bvae entries
    for list_key in ["ckpt", "clip", "bvae"]:
        unused_indices = [i for i, entry in enumerate(loaded_objects[list_key]) if entry[0] not in desired_ckpt_names]
        for index in sorted(unused_indices, reverse=True):
            loaded_objects[list_key].pop(index)

    # Check and clear unused vae entries
    unused_vae_indices = [i for i, entry in enumerate(loaded_objects["vae"]) if entry[0] not in desired_vae_names]
    for index in sorted(unused_vae_indices, reverse=True):
        loaded_objects["vae"].pop(index)

    # Check and clear unused lora entries
    unused_lora_indices = [i for i, entry in enumerate(loaded_objects["lora"]) if entry[0] not in desired_lora_names]
    for index in sorted(unused_lora_indices, reverse=True):
        loaded_objects["lora"].pop(index)


def load_checkpoint(ckpt_name,output_vae=True, output_clip=True):
    """
    Searches for tuple index that contains ckpt_name in "ckpt" array of loaded_objects.
    If found, extracts the model, clip, and vae from the loaded_objects.
    If not found, loads the checkpoint, extracts the model, clip, and vae, and adds them to the loaded_objects.
    Returns the model, clip, and vae.
    """
    global loaded_objects

    # Search for tuple index that contains ckpt_name in "ckpt" array of loaded_objects
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
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True,
                                                    embedding_directory=folder_paths.get_folder_paths("embeddings"))
        model = out[0]
        clip = out[1]
        vae = out[2]

        # Update loaded_objects[] array
        loaded_objects["ckpt"].append((ckpt_name, out[0]))
        loaded_objects["clip"].append((ckpt_name, out[1]))
        loaded_objects["bvae"].append((ckpt_name, out[2]))

    return model, clip, vae


def load_vae(vae_name):
    """
    Extracts the vae with a given name from the "vae" array in loaded_objects.
    If the vae is not found, creates a new VAE object with the given name and adds it to the "vae" array.
    """
    global loaded_objects

    # Check if vae_name exists in "vae" array
    if any(entry[0] == vae_name for entry in loaded_objects["vae"]):
        # Extract the second tuple entry of the checkpoint
        vae = [entry[1] for entry in loaded_objects["vae"] if entry[0] == vae_name][0]
    else:
        vae_path = folder_paths.get_full_path("vae", vae_name)
        vae = comfy.sd.VAE(ckpt_path=vae_path)
        # Update loaded_objects[] array
        loaded_objects["vae"].append((vae_name, vae))
    return vae


def load_lora(lora_name, model, clip, strength_model, strength_clip):
    """
    Extracts the Lora model with a given name from the "lora" array in loaded_objects.
    If the Lora model is not found or the strength values change or the original model has changed, creates a new Lora object with the given name and adds it to the "lora" array.
    """
    global loaded_objects

    # Get the model_name (ckpt_name) from the first entry in loaded_objects
    model_name = loaded_objects["ckpt"][0][0] if loaded_objects["ckpt"] else None

    # Check if lora_name exists in "lora" array
    existing_lora = [entry for entry in loaded_objects["lora"] if entry[0] == lora_name]

    if existing_lora:
        lora_name, stored_model_name, model_lora, clip_lora, stored_strength_model, stored_strength_clip = existing_lora[0]

        # Check if the model_name, strength_model, and strength_clip are the same
        if model_name == stored_model_name and strength_model == stored_strength_model and strength_clip == stored_strength_clip:
            # Check if the model has not changed in the loaded_objects
            existing_model = [entry for entry in loaded_objects["ckpt"] if entry[0] == model_name]
            if existing_model and existing_model[0][1] == model:
                return model_lora, clip_lora

    # If Lora model not found or strength values changed or model changed, generate new Lora models
    lora_path = folder_paths.get_full_path("loras", lora_name)
    model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora_path, strength_model, strength_clip)

    # Remove existing Lora model if it exists
    if existing_lora:
        loaded_objects["lora"].remove(existing_lora[0])

    # Update loaded_objects[] array
    loaded_objects["lora"].append((lora_name, model_name, model_lora, clip_lora, strength_model, strength_clip))

    return model_lora, clip_lora


# TSC Efficient Loader
class TSC_EfficientLoader:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
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
                "hidden": {"prompt": "PROMPT"}}

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP" ,)
    RETURN_NAMES = ("MODEL", "CONDITIONING+", "CONDITIONING-", "LATENT", "VAE", "CLIP", )
    FUNCTION = "efficientloader"
    CATEGORY = "Efficiency Nodes/Loaders"

    def efficientloader(self, ckpt_name, vae_name, clip_skip, lora_name, lora_model_strength, lora_clip_strength,
                        positive, negative, empty_latent_width, empty_latent_height, batch_size, prompt=None):

        model: ModelPatcher | None = None
        clip: CLIP | None = None
        vae: VAE | None = None

        # Create Empty Latent
        latent = torch.zeros([batch_size, 4, empty_latent_height // 8, empty_latent_width // 8]).cpu()

        # Clean models from loaded_objects
        update_loaded_objects(prompt)

        # Load models
        model, clip, vae = load_checkpoint(ckpt_name)

        if lora_name != "None":
            model, clip = load_lora(lora_name, model, clip, lora_model_strength, lora_clip_strength)
            # note:  load_lora only works properly (as of now) when ckpt dictionary is only 1 entry long!
        
        # Check for custom VAE
        if vae_name != "Baked VAE":
            vae = load_vae(vae_name)

        #print_loaded_objects_entries()

        # CLIP skip
        if not clip:
            raise Exception("No CLIP found")
        clip = clip.clone()
        clip.clip_layer(clip_skip)

        return (model, [[clip.encode(positive), {}]], [[clip.encode(negative), {}]], {"samples":latent}, vae, clip, )

# KSampler Efficient ID finder
last_returned_ids = {}
def find_k_sampler_id(prompt, sampler_state=None, seed=None, steps=None, cfg=None,
                      sampler_name=None, scheduler=None, denoise=None, preview_image=None):
    global last_returned_ids

    input_params = [
        ('sampler_state', sampler_state),
        ('seed', seed),
        ('steps', steps),
        ('cfg', cfg),
        ('sampler_name', sampler_name),
        ('scheduler', scheduler),
        ('denoise', denoise),
        ('preview_image', preview_image),
    ]

    matching_ids = []

    for key, value in prompt.items():
        if value.get('class_type') == 'KSampler (Efficient)':
            inputs = value['inputs']
            match = all(inputs[param_name] == param_value for param_name, param_value in input_params if param_value is not None)

            if match:
                matching_ids.append(key)

    if matching_ids:
        input_key = tuple(param_value for param_name, param_value in input_params)

        if input_key in last_returned_ids:
            last_id = last_returned_ids[input_key]
            next_id = None
            for id in matching_ids:
                if id > last_id:
                    if next_id is None or id < next_id:
                        next_id = id

            if next_id is None:
                # All IDs have been used; start again from the first one
                next_id = min(matching_ids)

        else:
            next_id = min(matching_ids)

        last_returned_ids[input_key] = next_id
        return next_id
    else:
        last_returned_ids.clear()
        return None

# TSC KSampler (Efficient)
last_helds: dict[str, list] = {
    "results": [],
    "latent": [],
    "images": [],
    "vae_decode": []
}
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
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     "latent_image": ("LATENT",),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "preview_image": (["Disabled", "Enabled"],),
                     },
                "optional": { "optional_vae": ("VAE",),
                              "script": ("SCRIPT",),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "IMAGE", )
    RETURN_NAMES = ("MODEL", "CONDITIONING+", "CONDITIONING-", "LATENT", "VAE", "IMAGE", )
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "Efficiency Nodes/Sampling"
    
    def sample(self, sampler_state, model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
               latent_image, preview_image, denoise=1.0, prompt=None, extra_pnginfo=None, optional_vae=(None,), script=None):

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

        my_unique_id = int(find_k_sampler_id(prompt, sampler_state, seed, steps, cfg, sampler_name,scheduler, denoise, preview_image))

        # Vae input check
        vae = optional_vae
        if vae == (None,):
            print('\033[32mKSampler(Efficient)[{}] Warning:\033[0m No vae input detected, preview and output image disabled.\n'.format(my_unique_id))
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
                images = vae.decode(latent).cpu()
                update_value_by_id("images", my_unique_id, images)

                # Disable vae decode on next Hold
                update_value_by_id("vae_decode", my_unique_id, False)

                # Generate image results and store
                results = preview_images(images, filename_prefix)
                update_value_by_id("results", my_unique_id, results)

                # Output image results to ui and node outputs
                return {"ui": {"images": results},
                        "result": (model, positive, negative, {"samples": latent}, vae, images,)}

        # If the sampler state is "Hold"
        elif sampler_state == "Hold":
            # Print a message indicating that the KSampler is in "Hold" state with the unique ID
            print('\033[32mKSampler(Efficient)[{}]:\033[0mHeld'.format(my_unique_id))

            # If not in preview mode, return the results in the specified format
            if preview_image == "Disabled":
                return {"ui": {"images": list()},
                        "result": (model, positive, negative, last_latent, vae, TSC_KSampler.empty_image,)}

            # if preview_image == "Enabled":
            else:
                latent = last_latent["samples"]

                if get_value_by_id("vae_decode", my_unique_id) == True:

                    # Decode images and store
                    images = vae.decode(latent).cpu()
                    update_value_by_id("images", my_unique_id, images)

                    # Disable vae decode on next Hold
                    update_value_by_id("vae_decode", my_unique_id, False)

                    # Generate image results and store
                    results = preview_images(images, filename_prefix)
                    update_value_by_id("results", my_unique_id, results)

                else:
                    images = last_images
                    results = last_results

                # Output image results to ui and node outputs
                return {"ui": {"images": results},
                        "result": (model, positive, negative, {"samples": latent}, vae, images,)}

        elif sampler_state == "Script":

            # If no script input connected, set X_type and Y_type to "Nothing"
            if script is None:
                X_type = "Nothing"
                Y_type = "Nothing"
            else:
                # Unpack script Tuple (X_type, X_value, Y_type, Y_value, grid_spacing, latent_id)
                X_type, X_value, Y_type, Y_value, grid_spacing, latent_id = script

            if (X_type == "Nothing" and Y_type == "Nothing"):
                print('\033[31mKSampler(Efficient)[{}] Error:\033[0m No valid script entry detected'.format(my_unique_id))
                return {"ui": {"images": list()},
                        "result": (model, positive, negative, last_latent, vae, last_images,)}

            if vae == (None,):
                print('\033[31mKSampler(Efficient)[{}] Error:\033[0m VAE must be connected to use Script mode.'.format(my_unique_id))
                return {"ui": {"images": list()},
                        "result": (model, positive, negative, last_latent, vae, last_images,)}

            # Extract the 'samples' tensor from the dictionary
            latent_image_tensor = latent_image['samples']

            # Split the tensor into individual image tensors
            image_tensors = torch.split(latent_image_tensor, 1, dim=0)

            # Create a list of dictionaries containing the individual image tensors
            latent_list = [{'samples': image} for image in image_tensors]

            # Set latent only to the first latent of batch
            if latent_id >= len(latent_list):
                print(
                    f'\033[31mKSampler(Efficient)[{my_unique_id}] Warning:\033[0m '
                    f'The selected latent_id ({latent_id}) is out of range.\n'
                    f'Automatically setting the latent_id to the last image in the list (index: {len(latent_list) - 1}).')
                latent_id = len(latent_list) - 1

            latent_image = latent_list[latent_id]

            # Define X/Y_values for "Seeds++ Batch"
            if X_type == "Seeds++ Batch":
                X_value = [latent_image for _ in range(X_value[0])]
            if Y_type == "Seeds++ Batch":
                Y_value = [latent_image for _ in range(Y_value[0])]

            # Define X/Y_values for "Latent Batch"
            if X_type == "Latent Batch":
                X_value = latent_list
            if Y_type == "Latent Batch":
                Y_value = latent_list

            # Embedd information into "Scheduler" X/Y_values for text label
            if X_type == "Scheduler" and Y_type != "Sampler":
                # X_value second list value of each array entry = None
                for i in range(len(X_value)):
                    if len(X_value[i]) == 2:
                        X_value[i][1] = None
                    else:
                        X_value[i] = [X_value[i], None]
            if Y_type == "Scheduler" and X_type != "Sampler":
                # Y_value second list value of each array entry = None
                for i in range(len(Y_value)):
                    if len(Y_value[i]) == 2:
                        Y_value[i][1] = None
                    else:
                        Y_value[i] = [Y_value[i], None]

            def define_variable(var_type, var, seed, steps, cfg,sampler_name, scheduler, latent_image, denoise,
                                vae_name, var_label, num_label):

                # If var_type is "Seeds++ Batch", update var and seed, and generate labels
                if var_type == "Latent Batch":
                    latent_image = var
                    text = f"{len(var_label)}"
                # If var_type is "Seeds++ Batch", update var and seed, and generate labels
                elif var_type == "Seeds++ Batch":
                    text = f"seed: {seed}"
                # If var_type is "Steps", update steps and generate labels
                elif var_type == "Steps":
                    steps = var
                    text = f"Steps: {steps}"
                # If var_type is "CFG Scale", update cfg and generate labels
                elif var_type == "CFG Scale":
                    cfg = var
                    text = f"CFG Scale: {cfg}"
                # If var_type is "Sampler", update sampler_name, scheduler, and generate labels
                elif var_type == "Sampler":
                    sampler_name = var[0]
                    if var[1] == "":
                        text = f"{sampler_name}"
                    else:
                        if var[1] != None:
                            scheduler[0] = var[1]
                        else:
                            scheduler[0] = scheduler[1]
                        text = f"{sampler_name} ({scheduler[0]})"
                    text = text.replace("ancestral", "a").replace("uniform", "u")
                # If var_type is "Scheduler", update scheduler and generate labels
                elif var_type == "Scheduler":
                    scheduler[0] = var[0]
                    if len(var) == 2:
                        text = f"{sampler_name} ({var[0]})"
                    else:
                        text = f"{var}"
                    text = text.replace("ancestral", "a").replace("uniform", "u")
                # If var_type is "Denoise", update denoise and generate labels
                elif var_type == "Denoise":
                    denoise = var
                    text = f"Denoise: {denoise}"
                # For any other var_type, set text to "?"
                elif var_type == "VAE":
                    vae_name = var
                    text = f"VAE: {vae_name}"
                # For any other var_type, set text to ""
                else:
                    text = ""

                def truncate_texts(texts, num_label):
                    min_length = min([len(text) for text in texts])
                    truncate_length = min(min_length, 24)

                    if truncate_length < 16:
                        truncate_length = 16

                    truncated_texts = []
                    for text in texts:
                        if len(text) > truncate_length:
                            text = text[:truncate_length] + "..."
                        truncated_texts.append(text)

                    return truncated_texts

                # Add the generated text to var_label if it's not full
                if len(var_label) < num_label:
                    var_label.append(text)

                # If var_type VAE , truncate entries in the var_label list when it's full
                if len(var_label) == num_label and var_type == "VAE":
                    var_label = truncate_texts(var_label, num_label)

                # Return the modified variables
                return steps, cfg,sampler_name, scheduler, latent_image, denoise, vae_name, var_label

            # Define a helper function to help process X and Y values
            def process_values(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise,
                               vae,vae_name, latent_new=[], max_width=0, max_height=0, image_list=[], size_list=[]):

                # Sample
                samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                          latent_image, denoise=denoise)

                # Decode images and store
                latent = samples[0]["samples"]

                # Add the latent tensor to the tensors list
                latent_new.append(latent)

                # Load custom vae if available
                if vae_name is not None:
                    vae = load_vae(vae_name)

                # Decode the image
                image = vae.decode(latent).cpu()

                # Convert the image from tensor to PIL Image and add it to the list
                pil_image = tensor2pil(image)
                image_list.append(pil_image)
                size_list.append(pil_image.size)

                # Update max dimensions
                max_width = max(max_width, pil_image.width)
                max_height = max(max_height, pil_image.height)

                # Return the touched variables
                return image_list, size_list, max_width, max_height, latent_new

             # Initiate Plot label text variables X/Y_label
            X_label = []
            Y_label = []

            # Seed_updated for "Seeds++ Batch" incremental seeds
            seed_updated = seed

            # Store the KSamplers original scheduler inside the same scheduler variable
            scheduler = [scheduler, scheduler]

            # By default set vae_name to None
            vae_name = None

            # Fill Plot Rows (X)
            for X_index, X in enumerate(X_value):
                # Seed control based on loop index during Batch
                if X_type == "Seeds++ Batch":
                    # Update seed based on the inner loop index
                    seed_updated = seed + X_index

                # Define X parameters and generate labels
                steps, cfg, sampler_name, scheduler, latent_image, denoise, vae_name, X_label = \
                    define_variable(X_type, X, seed_updated, steps, cfg, sampler_name, scheduler, latent_image,
                                    denoise, vae_name, X_label, len(X_value))

                if Y_type != "Nothing":
                    # Seed control based on loop index during Batch
                    for Y_index, Y in enumerate(Y_value):
                        if Y_type == "Seeds++ Batch":
                            # Update seed based on the inner loop index
                            seed_updated = seed + Y_index

                        # Define Y parameters and generate labels
                        steps, cfg, sampler_name, scheduler, latent_image, denoise, vae_name, Y_label = \
                            define_variable(Y_type, Y, seed_updated, steps, cfg, sampler_name, scheduler, latent_image,
                                            denoise, vae_name, Y_label, len(Y_value))

                        # Generate images
                        image_list, size_list, max_width, max_height, latent_new = \
                            process_values(model, seed_updated, steps, cfg, sampler_name, scheduler[0],
                                           positive, negative, latent_image, denoise, vae, vae_name)
                else:
                    # Generate images
                    image_list, size_list, max_width, max_height, latent_new = \
                        process_values(model, seed_updated, steps, cfg, sampler_name, scheduler[0],
                                       positive, negative, latent_image, denoise, vae, vae_name)


            def adjusted_font_size(text, initial_font_size, max_width):
                font = ImageFont.truetype(str(Path(font_path)), initial_font_size)
                text_width, _ = font.getsize(text)

                if text_width > (max_width * 0.9):
                    scaling_factor = 0.9  # A value less than 1 to shrink the font size more aggressively
                    new_font_size = int(initial_font_size * (max_width / text_width) * scaling_factor)
                else:
                    new_font_size = initial_font_size

                return new_font_size

            # Disable vae decode on next Hold
            update_value_by_id("vae_decode", my_unique_id, False)

            # Extract plot dimensions
            num_rows = max(len(Y_value) if Y_value is not None else 0, 1)
            num_cols = max(len(X_value) if X_value is not None else 0, 1)

            def rearrange_tensors(latent, num_cols, num_rows):
                new_latent = []
                for i in range(num_rows):
                    for j in range(num_cols):
                        index = j * num_rows + i
                        new_latent.append(latent[index])
                return new_latent

            # Rearrange latent array to match preview image grid
            latent_new = rearrange_tensors(latent_new, num_cols, num_rows)

            # Concatenate the tensors along the first dimension (dim=0)
            latent_new = torch.cat(latent_new, dim=0)

            # Store latent_new as last latent
            update_value_by_id("latent", my_unique_id, latent_new)

            # Calculate the dimensions of the white background image
            border_size = max_width // 15

            # Modify the background width and x_offset initialization based on Y_type
            if Y_type == "Nothing":
                bg_width = num_cols * max_width + (num_cols - 1) * grid_spacing
                x_offset_initial = 0
            else:
                bg_width = num_cols * max_width + (num_cols - 1) * grid_spacing + 3 * border_size
                x_offset_initial = border_size * 3

            # Modify the background height based on X_type
            if X_type == "Nothing":
                bg_height = num_rows * max_height + (num_rows - 1) * grid_spacing
                y_offset = 0
            else:
                bg_height = num_rows * max_height + (num_rows - 1) * grid_spacing + 3 * border_size
                y_offset = border_size * 3

            # Create the white background image
            background = Image.new('RGBA', (int(bg_width), int(bg_height)), color=(255, 255, 255, 255))

            for row in range(num_rows):

                # Initialize the X_offset
                x_offset = x_offset_initial

                for col in range(num_cols):
                    # Calculate the index for image_list
                    index = col * num_rows + row
                    img = image_list[index]

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
                        text_width, text_height = d.textsize(text, font=font)
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
                        initial_font_size = int(48 * img.height / 512)
                        font_size = adjusted_font_size(text, initial_font_size, img.height)

                        # Create a white background label image
                        label_bg = Image.new('RGBA', (img.height, font_size), color=(255, 255, 255, 0))
                        d = ImageDraw.Draw(label_bg)

                        # Create the font object
                        font = ImageFont.truetype(str(Path(font_path)), font_size)

                        # Calculate the text size and the starting position
                        text_width, text_height = d.textsize(text, font=font)
                        text_x = (img.height - text_width) // 2
                        text_y = (font_size - text_height) // 2

                        # Add the text to the label image
                        d.text((text_x, text_y), text, fill='black', font=font)

                        # Rotate the label_bg 90 degrees counter-clockwise
                        if Y_type != "Latent Batch":
                            label_bg = label_bg.rotate(90, expand=True)

                        # Calculate the available space between the left of the background and the left of the image
                        available_space = x_offset - label_bg.width

                        # Calculate the new X position for the label image
                        label_x = available_space // 2

                        # Calculate the Y position for the label image
                        label_y = y_offset + (img.height - label_bg.height) // 2

                        # Paste the label image to the left of the image on the background using alpha_composite()
                        background.alpha_composite(label_bg, (label_x, label_y))

                    # Update the x_offset
                    x_offset += img.width + grid_spacing

                # Update the y_offset
                y_offset += img.height + grid_spacing

            images = pil2tensor(background)
            update_value_by_id("images", my_unique_id, images)

            # Generate image results and store
            results = preview_images(images, filename_prefix)
            update_value_by_id("results", my_unique_id, results)

            # Clean loaded_objects
            update_loaded_objects(prompt)

            # Output image results to ui and node outputs
            return {"ui": {"images": results}, "result": (model, positive, negative, {"samples": latent_new}, vae, images,)}


# TSC XY Plot
class TSC_XYplot:
    examples = "(X/Y_types)     (X/Y_values)\n" \
               "Latent Batch    n/a\n" \
               "Seeds++ Batch   3\n" \
               "Steps           15;20;25\n" \
               "CFG Scale       5;10;15;20\n" \
               "Sampler(1)      dpmpp_2s_ancestral;euler;ddim\n" \
               "Sampler(2)      dpmpp_2m,karras;heun,normal\n" \
               "Scheduler       normal;simple;karras\n" \
               "Denoise         .3;.4;.5;.6;.7\n" \
               "VAE             vae_1; vae_2; vae_3"

    samplers = ";\n".join(comfy.samplers.KSampler.SAMPLERS)
    schedulers = ";\n".join(comfy.samplers.KSampler.SCHEDULERS)
    vaes = ";\n".join(folder_paths.get_filename_list("vae"))
    notes = "- During a 'Latent Batch', the corresponding X/Y_value is ignored.\n" \
            "- During a 'Latent Batch', the latent_id is ignored.\n" \
            "- For a 'Seeds++ Batch', starting seed is defined by the KSampler.\n" \
            "- Trailing semicolons are ignored in the X/Y_values.\n" \
            "- Parameter types not set by this node are defined in the KSampler."

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "X_type": (["Nothing", "Latent Batch", "Seeds++ Batch", "Steps", "CFG Scale",
                                "Sampler", "Scheduler", "Denoise", "VAE"],),
                    "X_value": ("STRING", {"default": "", "multiline": False}),
                    "Y_type": (["Nothing", "Latent Batch", "Seeds++ Batch", "Steps", "CFG Scale",
                                "Sampler", "Scheduler", "Denoise", "VAE"],),
                    "Y_value": ("STRING", {"default": "", "multiline": False}),
                    "grid_spacing": ("INT", {"default": 0, "min": 0, "max": 500, "step": 5}),
                    "XY_flip": (["False","True"],),
                    "latent_id": ("INT", {"default": 0, "min": 0, "max": 100}),
                    "help": ("STRING", {"default":
                                        f"____________EXAMPLES____________\n{cls.examples}\n\n"
                                        f"____________SAMPLERS____________\n{cls.samplers}\n\n"
                                        f"___________SCHEDULERS___________\n{cls.schedulers}\n\n"
                                        f"______________VAE_______________\n{cls.vaes}\n\n"
                                        f"_____________NOTES______________\n{cls.notes}",
                                        "multiline": True}),},
                }
    RETURN_TYPES = ("SCRIPT",)
    RETURN_NAMES = ("script",)
    FUNCTION = "XYplot"
    CATEGORY = "Efficiency Nodes/Scripts"

    def XYplot(self, X_type, X_value, Y_type, Y_value, grid_spacing, XY_flip, latent_id, help):

        # Store values as arrays
        X_value = X_value.replace(" ", "").replace("\n", "")  # Remove spaces and newline characters
        X_value = X_value.rstrip(";")  # Remove trailing semicolon
        X_value = X_value.split(";")  # Turn to array

        Y_value = Y_value.replace(" ", "").replace("\n", "")  # Remove spaces and newline characters
        Y_value = Y_value.rstrip(";")  # Remove trailing semicolon
        Y_value = Y_value.split(";")  # Turn to array

        # Define the valid bounds for each type
        bounds = {
            "Seeds++ Batch": {"min": 0, "max": 50},
            "Steps": {"min": 0},
            "CFG Scale": {"min": 0, "max": 100},
            "Sampler": {"options": comfy.samplers.KSampler.SAMPLERS},
            "Scheduler": {"options": comfy.samplers.KSampler.SCHEDULERS},
            "Denoise": {"min": 0, "max": 1},
            "VAE": {"options": folder_paths.get_filename_list("vae")}
        }

        def validate_value(value, value_type, bounds):
            """
            Validates a value based on its corresponding value_type and bounds.

            Parameters:
                value (str or int or float): The value to validate.
                value_type (str): The type of the value, which determines the valid bounds.
                bounds (dict): A dictionary that contains the valid bounds for each value_type.

            Returns:
                The validated value.
                None if no validation was done or failed.
            """

            # Seeds++ Batch
            if value_type == "Seeds++ Batch":
                try:
                    x = float(value)
                except ValueError:
                    print(
                        f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid batch count.")
                    return None

                if not x.is_integer():
                    print(
                        f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid batch count.")
                    return None
                else:
                    x = int(x)

                if x < bounds["Seeds++ Batch"]["min"]:
                    x = bounds["Seeds++ Batch"]["min"]
                elif x > bounds["Seeds++ Batch"]["max"]:
                    x = bounds["Seeds++ Batch"]["max"]

                return x
            # Steps
            elif value_type == "Steps":
                try:
                    x = int(value)
                    if x < bounds["Steps"]["min"]:
                        x = bounds["Steps"]["min"]
                    return x
                except ValueError:
                    print(
                        f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid Step count.")
                    return None
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
            # Sampler
            elif value_type == "Sampler":
                if isinstance(value, str) and ',' in value:
                    value = tuple(map(str.strip, value.split(',')))

                if isinstance(value, tuple):
                    if len(value) == 2:
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
            # Scheduler
            elif value_type == "Scheduler":
                if value not in bounds["Scheduler"]["options"]:
                    valid_schedulers = '\n'.join(bounds["Scheduler"]["options"])
                    print(
                        f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid Scheduler. Valid Schedulers are:\n{valid_schedulers}")
                    return None
                else:
                    return value
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
            # VAE
            elif value_type == "VAE":
                if value not in bounds["VAE"]["options"]:
                    valid_vaes = '\n'.join(bounds["VAE"]["options"])
                    print(
                        f"\033[31mXY Plot Error:\033[0m '{value}' is not a valid VAE. Valid VAEs are:\n{valid_vaes}")
                    return None
                else:
                    return value
            else:
                return None

        def reset_variables():
            X_type = "Nothing"
            X_value = [None]
            Y_type = "Nothing"
            Y_value = [None]
            latent_id = None
            grid_spacing = None
            return X_type, X_value, Y_type, Y_value, grid_spacing, latent_id

        if X_type == Y_type == "Nothing":
            return (reset_variables(),)

        # If types are the same, error and return
        if (X_type == Y_type) and (X_type != "Nothing"):
            print(f"\033[31mXY Plot Error:\033[0m X_type and Y_type must be different.")
            # Reset variables to default values and return
            return (reset_variables(),)

        # Validate X_value array length is 1 if doing a "Seeds++ Batch"
        if len(X_value) != 1 and X_type == "Seeds++ Batch":
            print(f"\033[31mXY Plot Error:\033[0m '{';'.join(X_value)}' is not a valid batch count.")
            return (reset_variables(),)

        # Validate Y_value array length is 1 if doing a "Seeds++ Batch"
        if len(Y_value) != 1 and Y_type == "Seeds++ Batch":
            print(f"\033[31mXY Plot Error:\033[0m '{';'.join(Y_value)}' is not a valid batch count.")
            return (reset_variables(),)

        # Loop over each entry in X_value and check if it's valid
        # Validate X_value based on X_type
        if X_type != "Nothing" and X_type != "Latent Batch":
            for i in range(len(X_value)):
                X_value[i] = validate_value(X_value[i], X_type, bounds)
                if X_value[i] == None:
                    # Reset variables to default values and return
                    return (reset_variables(),)

        # Loop over each entry in Y_value and check if it's valid
        # Validate Y_value based on Y_type
        if Y_type != "Nothing" and Y_type != "Latent Batch":
            for i in range(len(Y_value)):
                Y_value[i] = validate_value(Y_value[i], Y_type, bounds)
                if Y_value[i] == None:
                    # Reset variables to default values and return
                    return (reset_variables(),)

        # Clean Schedulers from Sampler data (if other type is Scheduler)
        if X_type == "Sampler" and Y_type == "Scheduler":
            # Clear X_value Scheduler's
            X_value = [[x[0], ""] for x in X_value]
        elif Y_type == "Sampler" and X_type == "Scheduler":
            # Clear Y_value Scheduler's
            Y_value = [[y[0], ""] for y in Y_value]

        # Clean X/Y_values
        if X_type == "Nothing" or X_type == "Latent Batch":
            X_value = [None]
        if Y_type == "Nothing" or Y_type == "Latent Batch":
            Y_value = [None]

        # Flip X and Y
        if XY_flip == "True":
            X_type, Y_type = Y_type, X_type
            X_value, Y_value = Y_value, X_value

        # Print the validated values
        if X_type != "Nothing" and X_type != "Latent Batch":
            print("\033[90m" + f"XY Plot validated values for X_type '{X_type}': {', '.join(map(str, X_value))}\033[0m")
        if Y_type != "Nothing" and Y_type != "Latent Batch":
            print("\033[90m" + f"XY Plot validated values for Y_type '{Y_type}': {', '.join(map(str, Y_value))}\033[0m")

        return ((X_type, X_value, Y_type, Y_value, grid_spacing, latent_id),)


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
    "XY Plot": TSC_XYplot,
    "Image Overlay": TSC_ImageOverlay,
    "Evaluate Integers": TSC_EvaluateInts,
    "Evaluate Strings": TSC_EvaluateStrs,
}