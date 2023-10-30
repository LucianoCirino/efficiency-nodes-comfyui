⚠️ **IMPORTANT:** Due to shifts in priorities and a decreased interest in this project from my end, this repository will no longer receive updates or maintenance. Use at your own risk...

Efficiency Nodes for ComfyUI
=======
### A collection of <a href="https://github.com/comfyanonymous/ComfyUI" >ComfyUI</a> custom nodes to help streamline workflows and reduce total node count.
## [Direct Download Link](https://github.com/LucianoCirino/efficiency-nodes-comfyui/releases/download/v2.0/efficiency-nodes-comfyui.7z)
### Nodes:
<!-------------------------------------------------------------------------------------------------------------------------------------------------------->
<details>
    <summary><b>Efficient Loader</b> & <b>Eff. Loader SDXL</b></summary>
<ul>
    <li>Nodes that can load & cache Checkpoint, VAE, & LoRA type models. <i>(cache settings found in config file 'node_settings.json')</i></li>
    <li>Able to apply LoRA & Control Net stacks via their <code>lora_stack</code> and <code>cnet_stack</code> inputs.</li>
    <li>Come with positive and negative prompt text boxes. You can also set the way you want the prompt to be <a href="https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb">encoded</a> via the <code>token_normalization</code> and <code>weight_interpretation</code> widgets.</li>
    <li>These node's also feature a variety of custom menu options as shown below.
        <p></p><img src="https://github.com/LucianoCirino/efficiency-nodes-media/blob/main/images/nodes//NodeMenu%20-%20Efficient%20Loaders.png" width="240" style="display: inline-block;"></p>
         <p><i>note: "🔍 View model info..." requires <a href="https://github.com/pythongosssss/ComfyUI-Custom-Scripts">ComfyUI-Custom-Scripts</a> to be installed to function.</i></p></li>
    <li>These loaders are used by the <b>XY Plot</b> node for many of its plot type dependencies.</li>
</ul>

<p align="center">
  <img src="https://github.com/LucianoCirino/efficiency-nodes-media/blob/main/images/nodes/NODE%20-%20Efficient%20Loader.png" width="240" style="display: inline-block;">
  <img src="https://github.com/LucianoCirino/efficiency-nodes-media/blob/main/images/nodes/NODE%20-%20Eff.%20Loader%20SDXL.png" width="240" style="display: inline-block;">
</p>
</details>
<!-------------------------------------------------------------------------------------------------------------------------------------------------------->
<details>
    <summary><b>KSampler (Efficient)</b>, <b>KSampler Adv. (Efficient)</b>, <b>KSampler SDXL (Eff.)</b></summary>

- Modded KSamplers with the ability to live preview generations and/or vae decode images.
- Feature a special seed box that allows for a clearer management of seeds. <i>(-1 seed to apply the selected seed behavior)</i>
- Can execute a variety of scripts, such as the <b>XY Plot</b> script. To activate the <code>script</code>, simply connect the input connection.

<p align="center">
  <img src="https://github.com/LucianoCirino/efficiency-nodes-media/blob/main/images/nodes/NODE%20-%20KSampler%20(Efficient).png" width="240">
  &nbsp; &nbsp; &nbsp;
  <img src="https://github.com/LucianoCirino/efficiency-nodes-media/blob/main/images/nodes/NODE%20-%20KSampler%20Adv.%20(Efficient).png" width="240">
  &nbsp; &nbsp; &nbsp;
  <img src="https://github.com/LucianoCirino/efficiency-nodes-media/blob/main/images/nodes/NODE%20-%20KSampler%20SDXL%20(Eff.).png" width="240">
</p>

</details>
<!-------------------------------------------------------------------------------------------------------------------------------------------------------->
<details>
    <summary><b>Script Nodes</b></summary>
    
- A group of node's that are used in conjuction with the Efficient KSamplers to execute a variety of 'pre-wired' set of actions.
- Script nodes can be chained if their input/outputs allow it. Multiple instances of the same Script Node in a chain does nothing.
    <p align="center">
      <img src="https://github.com/LucianoCirino/efficiency-nodes-media/blob/main/images/nodes/ScriptChain.png" width="1080">
    </p>
    <!-------------------------------------------------------------------------------------------------------------------------------------------------------->
    <details>
        <summary><b>XY Plot</b></summary>
    <ul>
        <li>Node that allows users to specify parameters for the Efficiency KSamplers to plot on a grid.</li>
    </ul>
    <p align="center">
      <img src="https://github.com/LucianoCirino/efficiency-nodes-media/blob/main/images/nodes/XY%20Plot%20-%20Node%20Example.png" width="1080">
    </p>
    
    </details>
    <!-------------------------------------------------------------------------------------------------------------------------------------------------------->
    <details>
        <summary><b>HighRes-Fix</b></summary>
    <ul>
        <li>Node that the gives user the ability to upscale KSampler results through variety of different methods.</li>
        <li>Comes out of the box with popular Neural Network Latent Upscalers such as Ttl's <a href="https://github.com/Ttl/ComfyUi_NNLatentUpscale">ComfyUi_NNLatentUpscale</a> and City96's <a href="https://github.com/city96/SD-Latent-Upscaler">SD-Latent-Upscaler</a>.</li>
        <li>Supports ControlNet guided latent upscaling. <i> (You must have Fannovel's <a href="https://github.com/Fannovel16/comfyui_controlnet_aux">comfyui_controlnet_aux</a> installed to unlock this feature)</i></li>
    </ul>
    <p align="center">
      <img src="https://github.com/LucianoCirino/efficiency-nodes-media/blob/main/images/nodes/HighResFix%20-%20Node%20Example.gif" width="1080">
    </p>
    
    </details>
    <!-------------------------------------------------------------------------------------------------------------------------------------------------------->
    <details>
        <summary><b>Noise Control</b></summary>
    <ul>
        <li>This node gives the user the ability to manipulate noise sources in a variety of ways, such as the sampling's RNG source.</li>
        <li>The <a href="https://github.com/shiimizu/ComfyUI_smZNodes">CFG Denoiser</a> noise hijack was developed by smZ, it allows you to get closer recreating Automatic1111 results.</li>
            <p></p><i>Note: The CFG Denoiser does not work with a variety of conditioning types such as ControlNet & GLIGEN</i></p>
        <li>This node also allows you to add noise <a href="https://github.com/chrisgoringe/cg-noise">Seed Variations</a> to your generations.</li>
        <li>For trying to replicate Automatic1111 images, this node will help you achieve it. Encode your prompt using "length+mean" <code>token_normalization</code> with "A1111" <code>weight_interpretation</code>, set the Noise Control Script node's <code>rng_source</code> to "gpu", and turn the <code>cfg_denoiser</code> to true.</li>
    </ul>
    <p align="center">
      <img src="https://github.com/LucianoCirino/efficiency-nodes-media/blob/main/images/nodes/NODE%20-%20Noise%20Control%20Script.png" width="320">
    </p>
    
    </details>
    <!-------------------------------------------------------------------------------------------------------------------------------------------------------->
    <details>
        <summary><b>Tiled Upscaler</b></summary>
    <ul>
        <li>The Tiled Upscaler script attempts to encompas BlenderNeko's <a href="https://github.com/BlenderNeko/ComfyUI_TiledKSampler">ComfyUI_TiledKSampler</a> workflow into 1 node.</li>
        <li>Script supports Tiled ControlNet help via the options.</li>
        <li>Strongly recommend the <code>preview_method</code> be "vae_decoded_only" when running the script.</li>
    </ul>
    <p align="center">
      <img src="https://github.com/LucianoCirino/efficiency-nodes-media/blob/main/images/nodes/Tiled%20Upscaler%20-%20Node%20Example.gif" width="1080">
    </p>
    
    </details>
        <!-------------------------------------------------------------------------------------------------------------------------------------------------------->
    <details>
        <summary><b>AnimateDiff</b></summary>
    <ul>
        <li>To unlock the AnimateDiff script it is required you have installed Kosinkadink's <a href="https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved">ComfyUI-AnimateDiff-Evolved</a>.</li>
        <li>The latent <code>batch_size</code> when running this script becomes your frame count.</li>
    </ul>
    <p align="center">
      <img src="https://github.com/LucianoCirino/efficiency-nodes-media/blob/main/images/nodes/AnimateDiff%20-%20Node%20Example.gif" width="1080">
    </p>
    
    </details>
</details>

<!-------------------------------------------------------------------------------------------------------------------------------------------------------->
<details>
    <summary><b>Image Overlay</b></summary>
<ul>
    <li>Node that allows for flexible image overlaying. Works also with image batches.</li>
</ul>
<p align="center">
  <img src="https://github.com/LucianoCirino/efficiency-nodes-media/blob/main/images/nodes/Image%20Overlay%20-%20Node%20Example.png" width="1080">
</p>
 
</details>
<!-------------------------------------------------------------------------------------------------------------------------------------------------------->
<details>
    <summary><b>SimpleEval Nodes</b></summary>
<ul>
    <li>A collection of nodes that allows users to write simple Python expressions for a variety of data types using the <i><a href="https://github.com/danthedeckie/simpleeval" >simpleeval</a></i> library.</li>
    <li>To activate you must have installed the simpleeval library in your Python workspace.</li>
    <pre>pip install simpleeval</pre>
</ul>
<p align="center">
  <img src="https://github.com/LucianoCirino/efficiency-nodes-media/blob/main/images/nodes/NODE%20-%20Evaluate%20Integers.png" width="320">
  &nbsp; &nbsp;
  <img src="https://github.com/LucianoCirino/efficiency-nodes-media/blob/main/images/nodes/NODE%20-%20Evaluate%20Floats.png" width="320">
  &nbsp; &nbsp;
  <img src="https://github.com/LucianoCirino/efficiency-nodes-media/blob/main/images/nodes/NODE%20-%20Evaluate%20Strings.png" width="320">
</p>

</details>

## Workflow Examples:
1. HiRes-Fixing<br>
   [<img src="https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/v2.0/workflows/HiResFix%20Script.png" width="800">](https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/v2.0/workflows/HiResFix%20Script.png)<br>

2. SDXL Refining & **Noise Control Script**<br>
   [<img src="https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/v2.0/workflows/SDXL%20Refining%20%26%20Noise%20Control%20Script.png" width="800">](https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/v2.0/workflows/SDXL%20Refining%20%26%20Noise%20Control%20Script.png)<br>

3. **XY Plot**: LoRA <code>model_strength</code> vs <code>clip_strength</code><br>
   [<img src="https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/v2.0/workflows/XYPlot%20-%20LoRA%20Model%20vs%20Clip%20Strengths.png" width="800">](https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/v2.0/workflows/XYPlot%20-%20LoRA%20Model%20vs%20Clip%20Strengths.png)<br>

4. Stacking Scripts: **XY Plot** + **Noise Control** + **HiRes-Fix**<br>
   [<img src="https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/v2.0/workflows/XYPlot%20-%20Seeds%20vs%20Checkpoints%20%26%20Stacked%20Scripts.png" width="800">](https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/v2.0/workflows/XYPlot%20-%20Seeds%20vs%20Checkpoints%20%26%20Stacked%20Scripts.png)<br>

5. Stacking Scripts: **AnimateDiff** + **HiRes-Fix** (with ControlNet)<br>
   [<img src="https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/v2.0/workflows/AnimateDiff%20%26%20HiResFix%20Scripts.gif" width="800">](https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/v2.0/workflows/AnimateDiff%20%26%20HiResFix%20Scripts.gif)<br>


### Dependencies
The python library <i><a href="https://github.com/danthedeckie/simpleeval" >simpleeval</a></i> is required to be installed if you wish to use the **Simpleeval Nodes**.
<pre>pip install simpleeval</pre>

## **Install:**
To install, drop the "_**efficiency-nodes-comfyui**_" folder into the "_**...\ComfyUI\ComfyUI\custom_nodes**_" directory and restart UI.
