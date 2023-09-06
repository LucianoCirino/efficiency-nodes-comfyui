Efficiency Nodes for ComfyUI
=======
### A collection of <a href="https://github.com/comfyanonymous/ComfyUI" >ComfyUI</a> custom nodes to help streamline workflows and reduce total node count.
## [Direct Download Link](https://github.com/LucianoCirino/efficiency-nodes-comfyui/releases/download/v1.92/efficiency-nodes-comfyui-v192.7z)
<!-------------------------------------------------------------------------------------------------------------------------------------------------------->
<details>
    <summary>Efficient Loader</summary>
  
- A combination of common initialization nodes.
- Able to load LoRA and Control Net stacks via its 'lora_stack' and 'cnet_stack' inputs.
- Can cache multiple Checkpoint, VAE, and LoRA models.   <i>(cache settings found in config file 'node_settings.json')</i>
- Used by the XY Plot node for many of its plot type dependencies.

<p align="center">
  <img src="https://github.com/LucianoCirino/efficiency-nodes-media/blob/main/images/nodes/NODE%20-%20Efficient%20Loader.png" width="320">
</p>

</details>
<!-------------------------------------------------------------------------------------------------------------------------------------------------------->
<details>
    <summary>Ksampler & Ksampler Adv. (Efficient)</summary>

- Modded KSamplers with the ability to live preview generations and/or vae decode images.
- Used for running the XY Plot script.   <i>('sampler_state' = "Script")</i>
- Can be set to re-output their last outputs by force.   <i>('sampler_state' = "Hold")</i>

<p align="center">
  <img src="https://github.com/LucianoCirino/efficiency-nodes-media/blob/main/images/nodes/NODE%20-%20KSampler%20(Efficient).png" width="320">
  &nbsp; &nbsp; &nbsp; &nbsp;
  <img src="https://github.com/LucianoCirino/efficiency-nodes-media/blob/main/images/nodes/NODE%20-%20KSampler%20Adv.%20(Efficient).png" width="320">
</p>

</details>
<!-------------------------------------------------------------------------------------------------------------------------------------------------------->
<details>
    <summary>XY Plot</summary>
  
- Node that allows users to specify parameters for the Efficient KSampler's to plot on a grid.

<p align="center">
  <img src="https://github.com/LucianoCirino/efficiency-nodes-media/blob/main/images/nodes/NODE%20-%20XY%20Plot.png" width="320">
</p>

</details>
<!-------------------------------------------------------------------------------------------------------------------------------------------------------->
<details>
    <summary>Image Overlay</summary>
  
- Node that allows for flexible image overlaying. Works also with image batches.

<p align="center">
  <img src="https://github.com/LucianoCirino/efficiency-nodes-media/blob/main/images/nodes/NODE%20-%20Image%20Overlay.png" width="320">
</p>
 
</details>
<!-------------------------------------------------------------------------------------------------------------------------------------------------------->
<details>
    <summary>SimpleEval Nodes</summary>

- A collection of nodes that allows users to write simple Python expressions for a variety of data types using the "<a href="https://github.com/danthedeckie/simpleeval" >simpleeval</a>" library.

<p align="center">
  <img src="https://github.com/LucianoCirino/efficiency-nodes-media/blob/main/images/nodes/NODE%20-%20Evaluate%20Integers.png" width="320">
  &nbsp; &nbsp;
  <img src="https://github.com/LucianoCirino/efficiency-nodes-media/blob/main/images/nodes/NODE%20-%20Evaluate%20Floats.png" width="320">
  &nbsp; &nbsp;
  <img src="https://github.com/LucianoCirino/efficiency-nodes-media/blob/main/images/nodes/NODE%20-%20Evaluate%20Strings.png" width="320">
</p>

</details>

## **Examples:**
  
- HiResFix using the **Efficient Loader**, **Ksampler (Efficient)**, and **HiResFix Script** nodes

[<img src="https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/main/workflows/HiRes Fix (overview).png" width="720">](https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/main/workflows/HiRes%20Fix.png)

- SDXL Refining using the **Eff. SDXL Loader**, and **Ksampler SDXL (Eff.)**

[<img src="https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/main/workflows/SDXL Base+Refine (Overview).png" width="640">](https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/main/workflows/SDXL%20Base%2BRefine.png)

- 2D Plotting using the **XY Plot** & **Ksampler (Efficient)** nodes 

[<img src="https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/main/workflows/XYplot/X-Seeds Y-Checkpoints (overview).png" width="720">](https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/main/workflows/XYplot/X-Seeds%20Y-Checkpoints.png)

[<img src="https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/main/workflows/XYplot/LoRA Plot X-ModelStr Y-ClipStr (Overview).png" width="720">](https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/main/workflows/XYplot/LoRA%20Plot%20X-ModelStr%20Y-ClipStr.png)

- Photobashing using the **Image Overlay** node

[<img src="https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/main/workflows/Image Overlay (overview).png" width="720">](https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/main/workflows/Image%20Overlay.png)

### Dependencies
Dependencies are automatically installed during ComfyUI boot up.

## **Install:**
To install, drop the "_**efficiency-nodes-comfyui**_" folder into the "_**...\ComfyUI\ComfyUI\custom_nodes**_" directory and restart UI.
