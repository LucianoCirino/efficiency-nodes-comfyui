import os
import sys

# Get the absolute path of the parent directory of the current script
my_dir = os.path.dirname(os.path.abspath(__file__))

# Add the My directory path to the sys.path list
sys.path.append(my_dir)

# Construct the absolute path to the ComfyUI directory
comfy_dir = os.path.abspath(os.path.join(my_dir, '..', '..'))

# Add the ComfyUI directory path to the sys.path list
sys.path.append(comfy_dir)

# Import functions from ComfyUI
from comfy.sd import *
from comfy import utils

LORA_CLIP_MAP = {
    "mlp.fc1": "mlp_fc1",
    "mlp.fc2": "mlp_fc2",
    "self_attn.k_proj": "self_attn_k_proj",
    "self_attn.q_proj": "self_attn_q_proj",
    "self_attn.v_proj": "self_attn_v_proj",
    "self_attn.out_proj": "self_attn_out_proj",
}

LORA_UNET_MAP_ATTENTIONS = {
    "proj_in": "proj_in",
    "proj_out": "proj_out",
    "transformer_blocks.0.attn1.to_q": "transformer_blocks_0_attn1_to_q",
    "transformer_blocks.0.attn1.to_k": "transformer_blocks_0_attn1_to_k",
    "transformer_blocks.0.attn1.to_v": "transformer_blocks_0_attn1_to_v",
    "transformer_blocks.0.attn1.to_out.0": "transformer_blocks_0_attn1_to_out_0",
    "transformer_blocks.0.attn2.to_q": "transformer_blocks_0_attn2_to_q",
    "transformer_blocks.0.attn2.to_k": "transformer_blocks_0_attn2_to_k",
    "transformer_blocks.0.attn2.to_v": "transformer_blocks_0_attn2_to_v",
    "transformer_blocks.0.attn2.to_out.0": "transformer_blocks_0_attn2_to_out_0",
    "transformer_blocks.0.ff.net.0.proj": "transformer_blocks_0_ff_net_0_proj",
    "transformer_blocks.0.ff.net.2": "transformer_blocks_0_ff_net_2",
}

LORA_UNET_MAP_RESNET = {
    "in_layers.2": "resnets_{}_conv1",
    "emb_layers.1": "resnets_{}_time_emb_proj",
    "out_layers.3": "resnets_{}_conv2",
    "skip_connection": "resnets_{}_conv_shortcut"
}

def load_lora_tsc(path, to_load):
    lora = utils.load_torch_file(path)
    patch_dict = {}
    loaded_keys = set()
    for x in to_load:
        alpha_name = "{}.alpha".format(x)
        alpha = None
        if alpha_name in lora.keys():
            alpha = lora[alpha_name].item()
            loaded_keys.add(alpha_name)

        A_name = "{}.lora_up.weight".format(x)
        B_name = "{}.lora_down.weight".format(x)
        mid_name = "{}.lora_mid.weight".format(x)

        if A_name in lora.keys():
            mid = None
            if mid_name in lora.keys():
                mid = lora[mid_name]
                loaded_keys.add(mid_name)
            patch_dict[to_load[x]] = (lora[A_name], lora[B_name], alpha, mid)
            loaded_keys.add(A_name)
            loaded_keys.add(B_name)


        ######## loha
        hada_w1_a_name = "{}.hada_w1_a".format(x)
        hada_w1_b_name = "{}.hada_w1_b".format(x)
        hada_w2_a_name = "{}.hada_w2_a".format(x)
        hada_w2_b_name = "{}.hada_w2_b".format(x)
        hada_t1_name = "{}.hada_t1".format(x)
        hada_t2_name = "{}.hada_t2".format(x)
        if hada_w1_a_name in lora.keys():
            hada_t1 = None
            hada_t2 = None
            if hada_t1_name in lora.keys():
                hada_t1 = lora[hada_t1_name]
                hada_t2 = lora[hada_t2_name]
                loaded_keys.add(hada_t1_name)
                loaded_keys.add(hada_t2_name)

            patch_dict[to_load[x]] = (lora[hada_w1_a_name], lora[hada_w1_b_name], alpha, lora[hada_w2_a_name], lora[hada_w2_b_name], hada_t1, hada_t2)
            loaded_keys.add(hada_w1_a_name)
            loaded_keys.add(hada_w1_b_name)
            loaded_keys.add(hada_w2_a_name)
            loaded_keys.add(hada_w2_b_name)


        ######## lokr
        lokr_w1_name = "{}.lokr_w1".format(x)
        lokr_w2_name = "{}.lokr_w2".format(x)
        lokr_w1_a_name = "{}.lokr_w1_a".format(x)
        lokr_w1_b_name = "{}.lokr_w1_b".format(x)
        lokr_t2_name = "{}.lokr_t2".format(x)
        lokr_w2_a_name = "{}.lokr_w2_a".format(x)
        lokr_w2_b_name = "{}.lokr_w2_b".format(x)

        lokr_w1 = None
        if lokr_w1_name in lora.keys():
            lokr_w1 = lora[lokr_w1_name]
            loaded_keys.add(lokr_w1_name)

        lokr_w2 = None
        if lokr_w2_name in lora.keys():
            lokr_w2 = lora[lokr_w2_name]
            loaded_keys.add(lokr_w2_name)

        lokr_w1_a = None
        if lokr_w1_a_name in lora.keys():
            lokr_w1_a = lora[lokr_w1_a_name]
            loaded_keys.add(lokr_w1_a_name)

        lokr_w1_b = None
        if lokr_w1_b_name in lora.keys():
            lokr_w1_b = lora[lokr_w1_b_name]
            loaded_keys.add(lokr_w1_b_name)

        lokr_w2_a = None
        if lokr_w2_a_name in lora.keys():
            lokr_w2_a = lora[lokr_w2_a_name]
            loaded_keys.add(lokr_w2_a_name)

        lokr_w2_b = None
        if lokr_w2_b_name in lora.keys():
            lokr_w2_b = lora[lokr_w2_b_name]
            loaded_keys.add(lokr_w2_b_name)

        lokr_t2 = None
        if lokr_t2_name in lora.keys():
            lokr_t2 = lora[lokr_t2_name]
            loaded_keys.add(lokr_t2_name)

        if (lokr_w1 is not None) or (lokr_w2 is not None) or (lokr_w1_a is not None) or (lokr_w2_a is not None):
            patch_dict[to_load[x]] = (lokr_w1, lokr_w2, alpha, lokr_w1_a, lokr_w1_b, lokr_w2_a, lokr_w2_b, lokr_t2)

    for x in lora.keys():
        if x not in loaded_keys:
            print("lora key not loaded", x)
    return patch_dict

def model_lora_keys(model, key_map={}):
    sdk = model.state_dict().keys()

    counter = 0
    for b in range(12):
        tk = "diffusion_model.input_blocks.{}.1".format(b)
        up_counter = 0
        for c in LORA_UNET_MAP_ATTENTIONS:
            k = "{}.{}.weight".format(tk, c)
            if k in sdk:
                lora_key = "lora_unet_down_blocks_{}_attentions_{}_{}".format(counter // 2, counter % 2, LORA_UNET_MAP_ATTENTIONS[c])
                key_map[lora_key] = k
                up_counter += 1
        if up_counter >= 4:
            counter += 1
    for c in LORA_UNET_MAP_ATTENTIONS:
        k = "diffusion_model.middle_block.1.{}.weight".format(c)
        if k in sdk:
            lora_key = "lora_unet_mid_block_attentions_0_{}".format(LORA_UNET_MAP_ATTENTIONS[c])
            key_map[lora_key] = k
    counter = 3
    for b in range(12):
        tk = "diffusion_model.output_blocks.{}.1".format(b)
        up_counter = 0
        for c in LORA_UNET_MAP_ATTENTIONS:
            k = "{}.{}.weight".format(tk, c)
            if k in sdk:
                lora_key = "lora_unet_up_blocks_{}_attentions_{}_{}".format(counter // 3, counter % 3, LORA_UNET_MAP_ATTENTIONS[c])
                key_map[lora_key] = k
                up_counter += 1
        if up_counter >= 4:
            counter += 1
    counter = 0
    text_model_lora_key = "lora_te_text_model_encoder_layers_{}_{}"
    for b in range(24):
        for c in LORA_CLIP_MAP:
            k = "transformer.text_model.encoder.layers.{}.{}.weight".format(b, c)
            if k in sdk:
                lora_key = text_model_lora_key.format(b, LORA_CLIP_MAP[c])
                key_map[lora_key] = k


    #Locon stuff
    ds_counter = 0
    counter = 0
    for b in range(12):
        tk = "diffusion_model.input_blocks.{}.0".format(b)
        key_in = False
        for c in LORA_UNET_MAP_RESNET:
            k = "{}.{}.weight".format(tk, c)
            if k in sdk:
                lora_key = "lora_unet_down_blocks_{}_{}".format(counter // 2, LORA_UNET_MAP_RESNET[c].format(counter % 2))
                key_map[lora_key] = k
                key_in = True
        for bb in range(3):
            k = "{}.{}.op.weight".format(tk[:-2], bb)
            if k in sdk:
                lora_key = "lora_unet_down_blocks_{}_downsamplers_0_conv".format(ds_counter)
                key_map[lora_key] = k
                ds_counter += 1
        if key_in:
            counter += 1

    counter = 0
    for b in range(3):
        tk = "diffusion_model.middle_block.{}".format(b)
        key_in = False
        for c in LORA_UNET_MAP_RESNET:
            k = "{}.{}.weight".format(tk, c)
            if k in sdk:
                lora_key = "lora_unet_mid_block_{}".format(LORA_UNET_MAP_RESNET[c].format(counter))
                key_map[lora_key] = k
                key_in = True
        if key_in:
            counter += 1

    counter = 0
    us_counter = 0
    for b in range(12):
        tk = "diffusion_model.output_blocks.{}.0".format(b)
        key_in = False
        for c in LORA_UNET_MAP_RESNET:
            k = "{}.{}.weight".format(tk, c)
            if k in sdk:
                lora_key = "lora_unet_up_blocks_{}_{}".format(counter // 3, LORA_UNET_MAP_RESNET[c].format(counter % 3))
                key_map[lora_key] = k
                key_in = True
        for bb in range(3):
            k = "{}.{}.conv.weight".format(tk[:-2], bb)
            if k in sdk:
                lora_key = "lora_unet_up_blocks_{}_upsamplers_0_conv".format(us_counter)
                key_map[lora_key] = k
                us_counter += 1
        if key_in:
            counter += 1

    return key_map

def load_lora_for_models_tsc(model, clip, lora_path, strength_model, strength_clip):
    key_map = model_lora_keys(model.model)
    key_map = model_lora_keys(clip.cond_stage_model, key_map)
    loaded = load_lora_tsc(lora_path, key_map)
    new_modelpatcher = model.clone()
    k = new_modelpatcher.add_patches(loaded, strength_model)
    new_clip = clip.clone()
    k1 = new_clip.add_patches(loaded, strength_clip)
    k = set(k)
    k1 = set(k1)
    for x in loaded:
        if (x not in k) and (x not in k1):
            print("NOT LOADED", x)

    return (new_modelpatcher, new_clip)

def load_checkpoint_guess_config_tsc(ckpt_path, output_vae=True, output_clip=True, output_clipvision=False, embedding_directory=None):
    sd = utils.load_torch_file(ckpt_path)
    sd_keys = sd.keys()
    clip = None
    clipvision = None
    vae = None
    model = None
    clip_target = None

    parameters = calculate_parameters(sd, "model.diffusion_model.")
    fp16 = model_management.should_use_fp16(model_params=parameters)

    class WeightsLoader(torch.nn.Module):
        pass

    model_config = model_detection.model_config_from_unet(sd, "model.diffusion_model.", fp16)
    if model_config is None:
        raise RuntimeError("ERROR: Could not detect model type of: {}".format(ckpt_path))

    if model_config.clip_vision_prefix is not None:
        if output_clipvision:
            clipvision = clip_vision.load_clipvision_from_sd(sd, model_config.clip_vision_prefix, True)

    offload_device = model_management.unet_offload_device()
    model = model_config.get_model(sd, "model.diffusion_model.")
    model = model.to(offload_device)
    model.load_model_weights(sd, "model.diffusion_model.")

    if output_vae:
        vae = VAE()
        w = WeightsLoader()
        w.first_stage_model = vae.first_stage_model
        load_model_weights(w, sd)

    if output_clip:
        w = WeightsLoader()
        clip_target = model_config.clip_target()
        clip = CLIP(clip_target, embedding_directory=embedding_directory)
        w.cond_stage_model = clip.cond_stage_model
        sd = model_config.process_clip_state_dict(sd)
        load_model_weights(w, sd)

    return (ModelPatcher(model, load_device=model_management.get_torch_device(), offload_device=offload_device), clip, vae, clipvision)

