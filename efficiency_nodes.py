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

def extract_node_info(prompt, id, indirect_key=None):
    # Convert ID to string
    id = str(id)
    node_id = None

    # If an indirect_key (like 'script') is provided, perform a two-step lookup
    if indirect_key:
        # Ensure the id exists in the prompt and has an 'inputs' entry with the indirect_key
        if id in prompt and 'inputs' in prompt[id] and indirect_key in prompt[id]['inputs']:
            # Extract the indirect_id
            indirect_id = prompt[id]['inputs'][indirect_key][0]

            # Ensure the indirect_id exists in the prompt
            if indirect_id in prompt:
                node_id = indirect_id
                return prompt[indirect_id].get('class_type', None), node_id

        # If indirect_key is not found within the prompt
        return None, None

    # If no indirect_key is provided, perform a direct lookup
    return prompt.get(id, {}).get('class_type', None), node_id

def extract_node_value(prompt, id, key):
    # If ID is in data, return its 'inputs' value for a given key. Otherwise, return None.
    return prompt.get(str(id), {}).get('inputs', {}).get(key, None)

# Cache models in RAM
loaded_objects = {
    "ckpt": [], # (ckpt_name, ckpt_model, clip, bvae, [id])
    "vae": [],  # (vae_name, vae, [id])
    "lora": []  # (lora_name, ckpt_name, lora_model, clip_lora, strength_model, strength_clip, [id])
}

def print_loaded_objects_entries(id=None, prompt=None, show_id=False):
    print("\n" + "-" * 40)  # Print an empty line followed by a separator line
    if id is not None:
        id = str(id)  # Convert ID to string
    if prompt is not None and id is not None:
        node_name, _ = extract_node_info(prompt, id)
        if show_id:
            print(f"\033[36m{node_name} Models Cache: (node_id:{int(id)})\033[0m")
        else:
            print(f"\033[36m{node_name} Models Cache:\033[0m")
    elif id is None:
        print(f"\033[36mGlobal Models Cache:\033[0m")
    else:
        print(f"\033[36mModels Cache: \nnode_id:{int(id)}\033[0m")
    print("- " * 20)  # Print an empty line followed by a separator line
    for key in ["ckpt", "vae", "lora"]:
        entries_with_id = loaded_objects[key] if id is None else [entry for entry in loaded_objects[key] if id in entry[-1]]
        if not entries_with_id:  # If no entries with the chosen ID, print None and skip this key
            continue
        print(f"{key.capitalize()}:")
        for i, entry in enumerate(entries_with_id, 1):  # Start numbering from 1
            truncated_name = entry[0][:50]  # Truncate at 50 characters
            if key == "lora":
                lora_weight_rounded = round(entry[4], 3)  # Round lora_weight to 3 decimal places
                if id is None:
                    associated_ids = ', '.join(map(str, entry[-1]))  # Gather all associated ids
                    print(f"  [{i}] {truncated_name} (ids: {associated_ids}, lora_weight: {lora_weight_rounded}, ckpt_name: {entry[1]})")
                else:
                    print(f"  [{i}] {truncated_name} (lora_weight: {lora_weight_rounded}, base_ckpt: {entry[1]})")
            else:
                if id is None:
                    associated_ids = ', '.join(map(str, entry[-1]))  # Gather all associated ids
                    print(f"  [{i}] {truncated_name} (ids: {associated_ids})")
                else:
                    print(f"  [{i}] {truncated_name}")
    #print("-" * 40)  # Print a separator line
    #print("\n")  # Print an empty line

# This function cleans global variables associated with nodes that are no longer detected on UI
def globals_cleanup(prompt):
    global loaded_objects
    global last_helds

    # Step 1: Clean up last_helds
    for key in list(last_helds.keys()):
        original_length = len(last_helds[key])
        last_helds[key] = [(value, id) for value, id in last_helds[key] if str(id) in prompt.keys()]
        ###if original_length != len(last_helds[key]):
            ###print(f'Updated {key} in last_helds: {last_helds[key]}')

    # Step 2: Clean up loaded_objects
    for key in list(loaded_objects.keys()):
        for i, tup in enumerate(list(loaded_objects[key])):
            # Remove ids from id array in each tuple that don't exist in prompt
            id_array = [id for id in tup[-1] if str(id) in prompt.keys()]
            if len(id_array) != len(tup[-1]):
                if id_array:
                    loaded_objects[key][i] = tup[:-1] + (id_array,)
                    ###print(f'Updated tuple at index {i} in {key} in loaded_objects: {loaded_objects[key][i]}')
                else:
                    # If id array becomes empty, delete the corresponding tuple
                    loaded_objects[key].remove(tup)
                    ###print(f'Deleted tuple at index {i} in {key} in loaded_objects because its id array became empty.')

def load_checkpoint(ckpt_name, id, output_vae=True, cache=None):
    """
    Searches for tuple index that contains ckpt_name in "ckpt" array of loaded_objects.
    If found, extracts the model, clip, and vae from the loaded_objects.
    If not found, loads the checkpoint, extracts the model, clip, and vae.
    The id parameter represents the node ID and is used for caching models for the XY Plot node.
    If the cache limit is reached for a specific id, clears the cache and returns the loaded model, clip, and vae without adding a new entry.
    If there is cache space, adds the id to the ids list if it's not already there.
    If there is cache space and the checkpoint was not found in loaded_objects, adds a new entry to loaded_objects.

    Parameters:
    - ckpt_name: name of the checkpoint to load.
    - id: an identifier for caching models for specific nodes.
    - output_vae: boolean, if True loads the VAE too.
    - cache (optional): an integer that specifies how many checkpoint entries with a given id can exist in loaded_objects. Defaults to None.
    """
    global loaded_objects

    for entry in loaded_objects["ckpt"]:
        if entry[0] == ckpt_name:
            _, model, clip, vae, ids = entry
            cache_full = cache and len([entry for entry in loaded_objects["ckpt"] if id in entry[-1]]) >= cache

            if cache_full:
                clear_cache(id, cache, "ckpt")
            elif id not in ids:
                ids.append(id)

            return model, clip, vae

    ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
    out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae, output_clip=True,
                                                embedding_directory=folder_paths.get_folder_paths("embeddings"))
    model = out[0]
    clip = out[1]
    vae = out[2]  # bvae

    if cache:
        if len([entry for entry in loaded_objects["ckpt"] if id in entry[-1]]) < cache:
            loaded_objects["ckpt"].append((ckpt_name, model, clip, vae, [id]))
        else:
            clear_cache(id, cache, "ckpt")

    return model, clip, vae

def load_vae(vae_name, id, cache=None):
    """
    Extracts the vae with a given name from the "vae" array in loaded_objects.
    If the vae is not found, creates a new VAE object with the given name and adds it to the "vae" array.
    Also stores the id parameter, which is used for caching models specifically for nodes with the given ID.
    If the cache limit is reached for a specific id, returns the loaded vae without adding id or making a new entry in loaded_objects.
    If there is cache space, and the id is not in the ids list, adds the id to the ids list.
    If there is cache space, and the vae was not found in loaded_objects, adds a new entry to the loaded_objects.

    Parameters:
    - vae_name: name of the VAE to load.
    - id (optional): an identifier for caching models for specific nodes. Defaults to None.
    - cache (optional): an integer that specifies how many vae entries with a given id can exist in loaded_objects. Defaults to None.
    """
    global loaded_objects

    for i, entry in enumerate(loaded_objects["vae"]):
        if entry[0] == vae_name:
            vae, ids = entry[1], entry[2]
            if id not in ids:
                if cache and len([entry for entry in loaded_objects["vae"] if id in entry[-1]]) >= cache:
                    return vae
                ids.append(id)
            if cache:
                clear_cache(id, cache, "vae")
            return vae

    vae_path = folder_paths.get_full_path("vae", vae_name)
    vae = comfy.sd.VAE(ckpt_path=vae_path)

    if cache:
        if len([entry for entry in loaded_objects["vae"] if id in entry[-1]]) < cache:
            loaded_objects["vae"].append((vae_name, vae, [id]))
        else:
            clear_cache(id, cache, "vae")

    return vae

def load_lora(lora_name, ckpt_name, strength_model, strength_clip, id, cache=None):
    """
    Extracts the Lora model with a given name from the "lora" array in loaded_objects.
    If the Lora model is not found or strength values changed or model changed, creates a new Lora object with the given name and adds it to the "lora" array.
    Also stores the id parameter, which is used for caching models specifically for nodes with the given ID.
    If the cache limit is reached for a specific id, clears the cache and returns the loaded Lora model and clip without adding a new entry.
    If there is cache space, adds the id to the ids list if it's not already there.
    If there is cache space and the Lora model was not found in loaded_objects, adds a new entry to loaded_objects.

    Parameters:
    - lora_name: name of the Lora model to load.
    - ckpt_name: name of the checkpoint from which the Lora model is created.
    - strength_model: strength of the Lora model.
    - strength_clip: strength of the clip in the Lora model.
    - id: an identifier for caching models for specific nodes.
    - cache (optional): an integer that specifies how many Lora entries with a given id can exist in loaded_objects. Defaults to None.
    """
    global loaded_objects

    for entry in loaded_objects["lora"]:
        if entry[0] == lora_name and entry[1] == ckpt_name and entry[4] == strength_model and entry[5] == strength_clip:
            _, _, lora_model, lora_clip, _, _, ids = entry
            cache_full = cache and len([entry for entry in loaded_objects["lora"] if id in entry[-1]]) >= cache

            if cache_full:
                clear_cache(id, cache, "lora")
            elif id not in ids:
                ids.append(id)

            return lora_model, lora_clip

    ckpt, clip, _ = load_checkpoint(ckpt_name, id, output_vae=False, cache=None)
    lora_path = folder_paths.get_full_path("loras", lora_name)
    lora_model, lora_clip = comfy.sd.load_lora_for_models(ckpt, clip, lora_path, strength_model, strength_clip)

    if cache:
        if len([entry for entry in loaded_objects["lora"] if id in entry[-1]]) < cache:
            loaded_objects["lora"].append((lora_name, ckpt_name, lora_model, lora_clip, strength_model, strength_clip, [id]))
        else:
            clear_cache(id, cache, "lora")

    return lora_model, lora_clip


def clear_cache(id, cache, dict_name):
    """
    Clear the cache for a specific id in a specific dictionary (either "ckpt" or "vae").
    If the cache limit is reached for a specific id, deletes the id from the oldest entry.
    If the id array of the entry becomes empty, deletes the entry.
    """
    # Get all entries associated with the id_element
    id_associated_entries = [entry for entry in loaded_objects[dict_name] if id in entry[-1]]
    while len(id_associated_entries) > cache:
        # Identify an older entry (but not necessarily the oldest) containing id
        older_entry = id_associated_entries[0]
        # Remove the id_element from the older entry
        older_entry[-1].remove(id)
        # If the id array of the older entry becomes empty after this, delete the entry
        if not older_entry[-1]:
            loaded_objects[dict_name].remove(older_entry)
        # Update the id_associated_entries
        id_associated_entries = [entry for entry in loaded_objects[dict_name] if id in entry[-1]]

def clear_cache_by_exception(node_id, vae_dict=None, ckpt_dict=None, lora_dict=None):
    """
    This function deletes a specific ID from tuples in one or more specified dictionaries in the global 'loaded_objects' variable.
    The function requires the 'node_id' to delete and takes optional arguments for each dictionary ('vae_dict', 'ckpt_dict', 'lora_dict').
    If an argument is None, the function does nothing for that dictionary.
    If an argument is an empty list, the function deletes the 'node_id' from all tuples in that dictionary.
    For 'lora_dict', exceptions to deletion can be passed as a list of tuples.

    node_id : The ID to delete.
    vae_dict : The 'vae' dictionary exceptions. If empty list, delete 'node_id' from all 'vae' tuples. If None, do nothing.
    ckpt_dict : The 'ckpt' dictionary exceptions. If empty list, delete 'node_id' from all 'ckpt' tuples. If None, do nothing.
    lora_dict : The 'lora' dictionary exceptions. Each exception is a tuple of ('lora_name', 'ckpt_name', 'strength_model', 'strength_clip').
                If empty list, delete 'node_id' from all 'lora' tuples. If None, do nothing.
    """
    global loaded_objects  # reference the global variable 'loaded_objects'

    # Create a dictionary to map argument names to 'loaded_objects' dictionary names
    dict_mapping = {
        "vae_dict": "vae",
        "ckpt_dict": "ckpt",
        "lora_dict": "lora"
    }

    # Loop over the input arguments
    for arg_name, arg_val in {"vae_dict": vae_dict, "ckpt_dict": ckpt_dict, "lora_dict": lora_dict}.items():
        # Skip if argument is None
        if arg_val is None:
            continue

        dict_name = dict_mapping[arg_name]  # get the corresponding dictionary name in 'loaded_objects'

        # Iterate over a copy of the list to allow modification during iteration
        for tuple_idx, tuple_item in enumerate(loaded_objects[dict_name].copy()):
            # Handle 'lora_dict' exceptions differently, checking if the tuple matches one in exceptions
            if arg_name == "lora_dict" and (tuple_item[0], tuple_item[1], tuple_item[4], tuple_item[5]) in arg_val:
                continue
            # For 'ckpt_dict' and 'vae_dict', check if the name is in exceptions
            elif tuple_item[0] in arg_val:
                continue

            # Check if the 'node_id' is in the id array of the tuple
            if node_id in tuple_item[-1]:
                # Remove the 'node_id' from the id array
                tuple_item[-1].remove(node_id)

                # If the id array becomes empty, remove the entire tuple
                if not tuple_item[-1]:
                    loaded_objects[dict_name].remove(tuple_item)

# Retrieve the cache number from 'cache_settings' json file
def get_cache_numbers(node_name):
    # Get the directory path of the current file
    my_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the file path for cache_settings.json
    settings_file = os.path.join(my_dir, 'cache_settings.json')
    # Load the settings from the JSON file
    with open(settings_file, 'r') as file:
        cache_settings = json.load(file)
    # Retrieve the cache numbers for the given node
    cache_numbers = cache_settings.get(node_name, {})
    vae_cache = int(cache_numbers.get('vae', 1))
    ckpt_cache = int(cache_numbers.get('ckpt', 1))
    lora_cache = int(cache_numbers.get('lora', 1))
    return vae_cache, ckpt_cache, lora_cache

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
                "hidden": { "prompt": "PROMPT",
                            "my_unique_id": "UNIQUE_ID",},
                }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "DEPENDENCIES",)
    RETURN_NAMES = ("MODEL", "CONDITIONING+", "CONDITIONING-", "LATENT", "VAE", "DEPENDENCIES", )
    FUNCTION = "efficientloader"
    CATEGORY = "Efficiency Nodes/Loaders"

    def efficientloader(self, ckpt_name, vae_name, clip_skip, lora_name, lora_model_strength, lora_clip_strength,
                        positive, negative, empty_latent_width, empty_latent_height, batch_size,
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

        # Load models
        model, clip, vae = load_checkpoint(ckpt_name, my_unique_id, cache=ckpt_cache)

        if lora_name != "None":
            model, clip = load_lora(lora_name, ckpt_name, lora_model_strength, lora_clip_strength,my_unique_id, cache=lora_cache)

        # Check for custom VAE
        if vae_name != "Baked VAE":
            vae = load_vae(vae_name, my_unique_id, cache=vae_cache)

        # CLIP skip
        if not clip:
            raise Exception("No CLIP found")
        clip = clip.clone()
        clip.clip_layer(clip_skip)

        # Data for XY Plot
        dependencies = (vae_name, ckpt_name, clip, clip_skip, positive, negative)

        return (model, [[clip.encode(positive), {}]], [[clip.encode(negative), {}]], {"samples":latent}, vae, dependencies, )

########################################################################################################################
# TSC KSampler (Efficient)
last_helds: dict[str, list] = {
    "results": [],      # (results, id) # Preview Images, stored as a pil image list
    "latent": [],       # (latent, id)  # Latent outputs, stored as a latent tensor list
    "images": [],       # (images, id)  # Image outputs, stored as an image tensor list
    "vae_decode": [],   # (vae_decode, id) # Used to track wether to vae-decode or not
}

def print_last_helds(id=None):
    print("\n" + "-" * 40)  # Print an empty line followed by a separator line
    if id is not None:
        id = str(id)  # Convert ID to string
        print(f"Node-specific Last Helds (node_id:{int(id)})")
    else:
        print(f"Global Last Helds:")
    print("- " * 20)  # Print an empty line followed by a separator line
    for key in ["results", "latent", "images", "vae_decode"]:
        entries_with_id = last_helds[key] if id is None else [entry for entry in last_helds[key] if id == entry[-1]]
        if not entries_with_id:  # If no entries with the chosen ID, print None and skip this key
            continue
        print(f"{key.capitalize()}:")
        for i, entry in enumerate(entries_with_id, 1):  # Start numbering from 1
            if isinstance(entry[0], bool):  # Special handling for boolean types
                output = entry[0]
            else:
                output = len(entry[0])
            if id is None:
                print(f"  [{i}] Output: {output} (id: {entry[-1]})")
            else:
                print(f"  [{i}] Output: {output}")
    print("-" * 40)  # Print a separator line
    print("\n")  # Print an empty line

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
                     "preview_image": (["Disabled", "Enabled"],),
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

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # If the sampler state is "Hold"
        elif sampler_state == "Hold":

            #Debug
            ###print_last_helds()
            ###print_loaded_objects_entries()
            ###print("\n" + "-" * 40)  # Print an empty line followed by a separator line

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

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        elif sampler_state == "Script":

            # Store name of connected node to script input
            script_node_name, script_node_id = extract_node_info(prompt, my_unique_id, 'script')

            # If no valid script input connected, error out
            if script == None or script == (None,) or (script_node_name!="XY Plot"):
                print('\033[31mKSampler(Efficient)[{}] Error:\033[0m No valid script input detected'.format(my_unique_id))
                return {"ui": {"images": list()},
                        "result": (model, positive, negative, last_latent, vae, last_images,)}

            # If no vae connected, throw errors
            if vae == (None,):
                print('\033[31mKSampler(Efficient)[{}] Error:\033[0m VAE must be connected to use Script mode.'.format(my_unique_id))
                return {"ui": {"images": list()},
                        "result": (model, positive, negative, last_latent, vae, last_images,)}

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
                lora_name = None
                lora_model_wt = None
                lora_clip_wt = None
                positive_prompt = None
                negative_prompt = None
                clip_skip = None

                # Unpack script Tuple (X_type, X_value, Y_type, Y_value, grid_spacing, Y_label_orientation, dependencies)
                X_type, X_value, Y_type, Y_value, grid_spacing, Y_label_orientation, cache_models, dependencies = script

                # Unpack Effficient Loader dependencies
                if dependencies is not None:
                    vae_name, ckpt_name, clip, clip_skip, positive_prompt, negative_prompt = dependencies

                # If not caching models, set to 1.
                if cache_models == "False":
                    vae_cache = ckpt_cache = lora_cache = 1
                else:
                    # Retrieve cache numbers
                    vae_cache, ckpt_cache, lora_cache = get_cache_numbers("XY Plot")
                # Pack cache numbers in a tuple
                cache = (vae_cache, ckpt_cache, lora_cache)

                # Define X/Y_values for "Seeds++ Batch"
                if X_type == "Seeds++ Batch":
                    X_value = [i for i in range(X_value[0])]
                if Y_type == "Seeds++ Batch":
                    Y_value = [i for i in range(Y_value[0])]

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

                # Optimize image generation by prioritizing Checkpoint>LoRA>VAE as X in For Loop. Flip back when done.
                if Y_type == "Checkpoint" or ( Y_type == "LoRA" and X_type != "Checkpoint") or  \
                        (Y_type == "VAE" and (X_type != "Checkpoint" and X_type != "LoRA")) or \
                        (X_type == "Nothing" and Y_type != "Nothing"):
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
                        dict_map[t] = v

                ckpt_dict = [t[0] for t in dict_map.get("Checkpoint", [])] if dict_map.get("Checkpoint", []) else []

                lora_dict = [t for t in dict_map.get("LoRA", [])] if dict_map.get("LoRA", []) else []

                # If both ckpt_dict and lora_dict are not empty, manipulate lora_dict as described
                if ckpt_dict and lora_dict:
                    lora_dict = [(lora_name, ckpt, lora_model_wt, lora_clip_wt) for ckpt in ckpt_dict for
                                 lora_name, lora_model_wt, lora_clip_wt in lora_dict]

                # If lora_dict is not empty and ckpt_dict is empty, insert ckpt_name into each tuple in lora_dict
                elif lora_dict:
                    lora_dict = [(lora_name, ckpt_name, lora_model_wt, lora_clip_wt) for
                                 lora_name, lora_model_wt, lora_clip_wt in
                                 lora_dict]

                vae_dict = dict_map.get("VAE", [])

                # prioritize Caching Checkpoints over LoRAs but not both.
                if X_type == "LoRA":
                    ckpt_dict = []
                if Y_type == "LoRA":  # This implies X_type == "Checkpoint"
                    lora_dict = []

                # Print dict_arrays for debugging
                ###print(f"vae_dict={vae_dict}\nckpt_dict={ckpt_dict}\nlora_dict={lora_dict}")

                # Clean values that won't be reused
                clear_cache_by_exception(script_node_id, vae_dict=vae_dict, ckpt_dict=ckpt_dict, lora_dict=lora_dict)
            
                #_______________________________________________________________________________________________________
                # Function that changes appropiate variables for next processed generations (also generates XY_labels)
                def define_variable(var_type, var, seed, steps, cfg, sampler_name, scheduler, denoise, vae_name,
                                    ckpt_name, clip_skip, lora_name, lora_model_wt, lora_clip_wt, var_label, num_label):

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
                        text = f"denoise: {round(denoise, 2)}"

                    # If var_type is "VAE", update vae_name and generate labels
                    elif var_type == "VAE":
                        vae_name = var
                        vae_filename = os.path.basename(vae_name)
                        text = f"VAE: {vae_filename}"

                    # If var_type is "Checkpoint", update model and clip (if needed) and generate labels
                    elif var_type == "Checkpoint":
                        ckpt_name = var[0]
                        clip_skip = var[1]
                        ckpt_filename = os.path.basename(ckpt_name)
                        text = f"{ckpt_filename}"
                        #text = f"{ckpt_filename[:16]}... ({clip_skip})" if len(
                            #ckpt_filename) > 16 else f"{ckpt_filename} ({clip_skip})"

                    # If var_type is "LoRA", update lora_model and lora_clip (if needed) and generate labels
                    elif var_type == "LoRA":
                        lora_name = var[0]
                        lora_model_wt = var[1]
                        lora_clip_wt = var[2]
                        lora_filename = os.path.basename(lora_name)
                        text = f"<LoRA:{round(lora_model_wt, 2)}> {lora_filename}"

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
                    if len(var_label) == num_label and (var_type == "VAE" or var_type == "Checkpoint" or var_type == "LoRA"):
                        var_label = truncate_texts(var_label, num_label)

                    # Return the modified variables
                    return steps, cfg, sampler_name, scheduler, denoise, vae_name, ckpt_name, clip_skip,\
                        lora_name, lora_model_wt, lora_clip_wt, var_label

                # _______________________________________________________________________________________________________
                # The function below is used to smartly load Checkpoint/LoRA/VAE models between generations.
                def define_model(model, clip, positive, negative, positive_prompt, negative_prompt, clip_skip, vae,
                                 vae_name, ckpt_name, lora_name, lora_model_wt, lora_clip_wt, index, types, script_node_id, cache):
        
                    # Encode prompt and apply clip_skip. Return new conditioning.
                    def encode_prompt(positive_prompt, negative_prompt, clip, clip_skip):
                        clip = CLIPSetLastLayer().set_last_layer(clip, clip_skip)[0]
                        return [[clip.encode(positive_prompt), {}]], [[clip.encode(negative_prompt), {}]]

                    # Variable to track wether to encode prompt or not
                    encode = False

                    # Unpack types tuple
                    X_type, Y_type = types

                    # Load VAE if required
                    if (X_type == "VAE" and index == 0) or Y_type == "VAE":
                        vae = load_vae(vae_name, script_node_id, cache=cache[0])

                    # Load Checkpoint if required. If Y_type is LoRA, required models will be loaded by load_lora func.
                    if (X_type == "Checkpoint" and index == 0 and Y_type != "LoRA"):
                        model, clip, _ = load_checkpoint(ckpt_name, script_node_id, False, cache=cache[1])
                        encode = True

                    # Load LoRA if required
                    if (X_type == "LoRA" and index == 0) or Y_type == "LoRA":
                        model, clip = load_lora(lora_name, ckpt_name, lora_model_wt, lora_clip_wt, script_node_id, cache=cache[2])
                        encode = True

                    # Encode prompt if needed
                    if encode == True:
                        positive, negative = encode_prompt(positive_prompt, negative_prompt, clip, clip_skip)
                        
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
                    image = vae.decode(latent).cpu()

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
                scheduler = [scheduler, scheduler]

                # Store types in a Tuple for easy function passing
                types = (X_type, Y_type)

                # Fill Plot Rows (X)
                for X_index, X in enumerate(X_value):

                    # Seed control based on loop index during Batch
                    if X_type == "Seeds++ Batch":
                        # Update seed based on the inner loop index
                        seed_updated = seed + X_index

                    # Define X parameters and generate labels
                    steps, cfg, sampler_name, scheduler, denoise, vae_name, ckpt_name, clip_skip,\
                        lora_name, lora_model_wt, lora_clip_wt, X_label = \
                        define_variable(X_type, X, seed_updated, steps, cfg, sampler_name, scheduler, denoise, vae_name,
                                        ckpt_name, clip_skip, lora_name, lora_model_wt, lora_clip_wt, X_label, len(X_value))

                    if X_type != "Nothing" and Y_type == "Nothing":

                        # Models & Conditionings
                        model, positive, negative , vae = \
                            define_model(model, clip, positive, negative, positive_prompt, negative_prompt, clip_skip, vae,
                                         vae_name, ckpt_name, lora_name, lora_model_wt, lora_clip_wt, 0, types, script_node_id, cache)

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
                            steps, cfg, sampler_name, scheduler, denoise, vae_name, ckpt_name, clip_skip, \
                                lora_name, lora_model_wt, lora_clip_wt, Y_label = \
                                define_variable(Y_type, Y, seed_updated, steps, cfg, sampler_name, scheduler, denoise, vae_name,
                                                ckpt_name, clip_skip, lora_name, lora_model_wt, lora_clip_wt, Y_label, len(Y_value))

                            # Models & Conditionings
                            model, positive, negative, vae = \
                            define_model(model, clip, positive, negative, positive_prompt, negative_prompt, clip_skip, vae,
                                         vae_name, ckpt_name, lora_name, lora_model_wt, lora_clip_wt, Y_index, types, script_node_id, cache)

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
                    if Y_type == "LoRA":  # This implies X_type == "Checkpoint"
                        clear_cache_by_exception(script_node_id, lora_dict=[])

                # ______________________________________________________________________________________________________
                def print_plot_variables(X_type, Y_type, X_value, Y_value, seed, ckpt_name, lora_name, vae_name,
                                         clip_skip, steps, cfg, sampler_name, scheduler, denoise, latent_height, latent_width):
                    #print("\n" + "-" * 40)  # Print an empty line followed by a separator line
                    print("-" * 40)  # Print an empty line followed by a separator line
                    print("\033[32mXY Plot Settings:\033[0m")
                    print("- " * 20)  # Print an empty line followed by a separator line

                    if X_type == "Checkpoint" or Y_type == "Checkpoint":
                        ckpt_name = ", ".join([str(x[0]) for x in X_value]) if X_type == "Checkpoint" else ckpt_name
                        clip_skip = ", ".join([str(x[1]) for x in X_value]) if X_type == "Checkpoint" else clip_skip

                    lora_name = ", ".join([str(x[0]) for x in X_value]) if X_type == "LoRA" else ", ".join(
                        [str(y[0]) for y in Y_value]) if Y_type == "LoRA" else lora_name

                    vae_name = ", ".join(X_value) if X_type == "VAE" else vae_name
                    ckpt_name = ", ".join(Y_value) if Y_type == "VAE" else ckpt_name

                    seed_list = [seed + x for x in X_value] if X_type == "Seeds++ Batch" else [seed + y for y in
                                                                                               Y_value] if Y_type == "Seeds++ Batch" else [
                        seed]
                    seed = ", ".join(map(str, seed_list))

                    steps = ", ".join(map(str, X_value)) if X_type == "Steps" else ", ".join(
                        map(str, Y_value)) if Y_type == "Steps" else steps

                    cfg = ", ".join(map(str, X_value)) if X_type == "CFG Scale" else ", ".join(
                        map(str, Y_value)) if Y_type == "CFG Scale" else cfg

                    if X_type == "Sampler" or Y_type == "Sampler":
                        sampler_name = ", ".join([str(x[0]) for x in X_value]) if X_type == "Sampler" else sampler_name
                        scheduler = ", ".join([str(x[1]) for x in X_value]) if X_type == "Sampler" else scheduler

                    scheduler = ", ".join([str(x[0]) for x in X_value]) if X_type == "Scheduler" else ", ".join(
                        [str(y[0]) for y in Y_value]) if Y_type == "Scheduler" else scheduler

                    denoise = ", ".join(map(str, X_value)) if X_type == "Denoise" else ", ".join(
                        map(str, Y_value)) if Y_type == "Denoise" else denoise

                    print(f"img_count: {len(X_value)*len(Y_value)}")
                    print(f"dim: {latent_height} x {latent_width}")
                    print(f"ckpt_name: {ckpt_name if ckpt_name is not None else '?'}")
                    print(f"lora_name: {lora_name}")
                    print(f"vae_name: {vae_name if vae_name is not None else '?'}")
                    print(f"clip_skip: {clip_skip if clip_skip is not None else '?'}")
                    print(f"seed: {seed}")
                    print(f"steps: {steps}")
                    print(f"cfg: {cfg}")
                    print(f"sampler_name: {sampler_name}")
                    print(f"scheduler: {scheduler}")
                    print(f"denoise: {denoise}")

                print_plot_variables(X_type, Y_type, X_value, Y_value, seed, ckpt_name, lora_name, vae_name, clip_skip,
                                     steps, cfg, sampler_name, scheduler[0], denoise, latent_height, latent_width)

                # ______________________________________________________________________________________________________
                def adjusted_font_size(text, initial_font_size, latent_width):
                    font = ImageFont.truetype(str(Path(font_path)), initial_font_size)
                    text_width, _ = font.getsize(text)

                    if text_width > (latent_width * 0.9):
                        scaling_factor = 0.9  # A value less than 1 to shrink the font size more aggressively
                        new_font_size = int(initial_font_size * (latent_width / text_width) * scaling_factor)
                    else:
                        new_font_size = initial_font_size

                    return new_font_size

                # ______________________________________________________________________________________________________
                
                # Disable vae decode on next Hold
                update_value_by_id("vae_decode", my_unique_id, False)

                # Flip X & Y results back if flipped earlier (for Checkpoint/LoRA For loop optimizations)
                if flip_xy == True:
                    X_type, Y_type = Y_type, X_type
                    X_value, Y_value = Y_value, X_value
                    X_label, Y_label = Y_label, X_label

                # Extract plot dimensions
                num_rows = max(len(Y_value) if Y_value is not None else 0, 1)
                num_cols = max(len(X_value) if X_value is not None else 0, 1)

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

                # Rearrange lists for proper display
                if flip_xy == False:
                    latent_list = rearrange_list_A(latent_list, num_cols, num_rows)
                else:
                    latent_list = rearrange_list_B(latent_list, num_cols, num_rows)
                    image_pil_list = rearrange_list_A(image_pil_list, num_rows, num_cols)

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
                            if Y_label_orientation == "Vertical":
                                initial_font_size = int(48 * latent_width / 512)  # Adjusting this to be same as X_label size
                                font_size = adjusted_font_size(text, initial_font_size, latent_width)
                            else:  # Assuming Y_label_orientation is "Horizontal"
                                initial_font_size = int(48 *  (border_size_left/Y_label_scale) / 512)  # Adjusting this to be same as X_label size
                                font_size = adjusted_font_size(text, initial_font_size,  int(border_size_left/Y_label_scale))

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
            image_tensor_list = torch.stack([tensor.squeeze() for tensor in image_tensor_list])
            update_value_by_id("images", my_unique_id, image_tensor_list)

            # Print cache if set to true
            if cache_models == "True":
                print_loaded_objects_entries(script_node_id, prompt)

            print("\n" + "-" * 40)  # Print an empty line followed by a separator line

            # Output image results to ui and node outputs
            return {"ui": {"images": results}, "result": (model, positive, negative, {"samples": latent_list}, vae, image_tensor_list,)}

########################################################################################################################
# TSC XY Plot
class TSC_XYplot:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "grid_spacing": ("INT", {"default": 0, "min": 0, "max": 500, "step": 5}),
                    "XY_flip": (["False","True"],),
                    "Y_label_orientation": (["Horizontal", "Vertical"],),
                    "cache_models": (["True", "False"],),},
                "optional": {"dependencies": ("DEPENDENCIES", ),
                             "X": ("XY", ),
                             "Y": ("XY", ),},
            "hidden": {"my_unique_id": "UNIQUE_ID",},}
    RETURN_TYPES = ("SCRIPT",)
    RETURN_NAMES = ("SCRIPT",)
    FUNCTION = "XYplot"
    CATEGORY = "Efficiency Nodes/XY Plot"

    def XYplot(self, grid_spacing, XY_flip, Y_label_orientation,
               cache_models, dependencies=None, X=None, Y=None, my_unique_id=None):

        # Unpack X & Y Tuples if connected
        if X is not None:
            X_type, X_value  = X
        else:
            X_type = "Nothing"
            X_value = [""]
        if Y is not None:
            Y_type, Y_value = Y
        else:
            Y_type = "Nothing"
            Y_value = [""]

        # Nothing is connected or Error
        if X == Y == None or X_type == "Error" or Y_type == "Error":
            return (None,)

        # If types are the same, error and return
        if (X_type == Y_type) and (X_type != "Nothing"):
            print(f"\033[31mXY Plot Error:\033[0m X and Y must be different.")
            # Return None
            return (None,)

        # Check that dependencies is connected for Checkpoint and LoRA plots
        if X_type == "Checkpoint" or Y_type == "Checkpoint" or  X_type == "LoRA" or Y_type == "LoRA":
            if dependencies == None: # Not connected
                print(f"\033[31mXY Plot Error:\033[0m The dependencies input must be connected for Checkpoint/LoRA plots.")
                # Return None
                return (None,)

        # Clean Schedulers from Sampler data (if other type is Scheduler)
        if X_type == "Sampler" and Y_type == "Scheduler":
            # Clear X_value Scheduler's
            X_value = [[x[0], ""] for x in X_value]
        elif Y_type == "Sampler" and X_type == "Scheduler":
            # Clear Y_value Scheduler's
            Y_value = [[y[0], ""] for y in Y_value]

        # Flip X and Y
        if XY_flip == "True":
            X_type, Y_type = Y_type, X_type
            X_value, Y_value = Y_value, X_value


        return ((X_type, X_value, Y_type, Y_value, grid_spacing, Y_label_orientation, cache_models, dependencies),)


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
            "selection_count": ("INT", {"default": 0, "min": 0, "max": 5}),
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

    def xy_value(self, selection_count, steps_1, steps_2, steps_3, steps_4, steps_5):
        xy_type = "Steps"
        xy_value = [step for idx, step in enumerate([steps_1, steps_2, steps_3, steps_4, steps_5], start=1) if
                 idx <= selection_count]
        if not xy_value:  # Check if the list is empty
            return (None,)
        return ((xy_type, xy_value),)


# TSC XY Plot: CFG Values
class TSC_XYplot_CFG:

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "selection_count": ("INT", {"default": 0, "min": 0, "max": 5}),
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

    def xy_value(self, selection_count, cfg_1, cfg_2, cfg_3, cfg_4, cfg_5):
        xy_type = "CFG Scale"
        xy_value = [cfg for idx, cfg in enumerate([cfg_1, cfg_2, cfg_3, cfg_4, cfg_5], start=1) if idx <= selection_count]
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
            "lora_name_5": (cls.loras,),},
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, model_strengths, clip_strengths, lora_name_1, lora_name_2, lora_name_3, lora_name_4, lora_name_5):
        xy_type = "LoRA"
        loras = [lora_name_1, lora_name_2, lora_name_3, lora_name_4, lora_name_5]
        xy_value = [(lora, model_strengths, clip_strengths) for lora in loras if lora != "None"]
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
        }

    RETURN_TYPES = ("XY",)
    RETURN_NAMES = ("X or Y",)
    FUNCTION = "xy_value"
    CATEGORY = "Efficiency Nodes/XY Plot/XY Inputs"

    def xy_value(self, lora_name_1, model_str_1, clip_str_1, lora_name_2, model_str_2, clip_str_2,
                 lora_name_3, model_str_3, clip_str_3, lora_name_4, model_str_4, clip_str_4, lora_name_5, model_str_5, clip_str_5):
        xy_type = "LoRA"
        loras = [lora_name_1, lora_name_2, lora_name_3, lora_name_4, lora_name_5]
        model_strs = [model_str_1, model_str_2, model_str_3, model_str_4, model_str_5]
        clip_strs = [clip_str_1, clip_str_2, clip_str_3, clip_str_4, clip_str_5]
        xy_value = [(lora, model_str, clip_str) for lora, model_str, clip_str in zip(loras, model_strs, clip_strs) if lora != "None"]
        if not xy_value:  # Check if the list is empty
            return (None,)
        return ((xy_type, xy_value),)


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
            xy_type = "Error"
            xy_value = ""
        elif xy_type_1 == "Seeds++ Batch":
            xy_type = xy_type_1
            xy_value = [xy_value_1[0] + xy_value_2[0]]
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

########################################################################################################################
# Install simple_eval if missing from packages
def install_simpleeval():
    if 'simpleeval' not in packages():
        print("\033[32mEfficiency Nodes:\033[0m")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'simpleeval'])

def packages(versions=False):
    return [(r.decode().split('==')[0] if not versions else r.decode()) for r in subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).split()]

install_simpleeval()
from simpleeval import simple_eval

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
        result = simple_eval(python_expression, names={'a': a, 'b': b, 'c': c})
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
        result = simple_eval(python_expression, names={'a': a, 'b': b, 'c': c})
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
        functions = {"len": len}  # Define the functions for the expression
        result = simple_eval(python_expression, names=variables, functions=functions)
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
    "XY Plot": TSC_XYplot,
    "XY Input: Seeds++ Batch": TSC_XYplot_SeedsBatch,
    "XY Input: Steps": TSC_XYplot_Steps,
    "XY Input: CFG Scale": TSC_XYplot_CFG,
    "XY Input: Sampler": TSC_XYplot_Sampler,
    "XY Input: Scheduler": TSC_XYplot_Scheduler,
    "XY Input: Denoise": TSC_XYplot_Denoise,
    "XY Input: VAE": TSC_XYplot_VAE,
    "XY Input: Checkpoint": TSC_XYplot_Checkpoint,
    "XY Input: LoRA": TSC_XYplot_LoRA,
    "XY Input: LoRA (Advanced)": TSC_XYplot_LoRA_Adv,
    "Join XY Inputs": TSC_XYplot_JoinInputs,
    "Image Overlay": TSC_ImageOverlay,
    "Evaluate Integers": TSC_EvaluateInts,
    "Evaluate Floats": TSC_EvaluateFloats,
    "Evaluate Strings": TSC_EvaluateStrs,
    "Simple Eval Examples": TSC_EvalExamples
}