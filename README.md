Efficiency Nodes for ComfyUI
=======
### A collection of ComfyUI custom nodes to help streamline workflows and reduce node count for <a href="https://github.com/comfyanonymous/ComfyUI" >ComfyUI</a>.
## [Direct Download Link](https://github.com/LucianoCirino/efficiency-nodes-comfyui/releases/download/v1.0/efficiency-nodes-comfyui.v1.0.zip)

## **Currently Available Nodes:**
<details><summary>Ksampler (Efficient)</summary>

 * Modded KSampler that has the ability to preview and output images<br>
 * Re-outputs key inputs for a cleaner ComfyUI workflow look<br>
 * Can force hold all of its outputs without regenerating, including the output image
<blockquote>note: When using multiple instances of this node, each node must have a unique id for the "Hold" function to work properly</blockquote>
</details>
<details><summary>Efficient Loader</summary>

* A combination of common initialization nodes
</details>

<details><summary>Image Overlay</summary>

* Node that allows for flexible image overlaying
</details>

<details><summary>Evaluate Integers</summary>

* 3 integer input node that gives the user ability to write their own python expression for a INT/FLOAT type output.
</details>

<details><summary>Evaluate Strings</summary>

* 3 string input node that gives the user ability to write their own python expression for a STRING output.
</details>

## **Examples:**
  
- HiResFix using the **Efficient Loader** & **Ksampler (Efficient)**

<img src="https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/main/workflows/HiResFix.png" width="720">

- Photobashing using the **Image Overlay** node

<img src="https://github.com/LucianoCirino/efficiency-nodes-comfyui/blob/main/workflows/ImgOverlay.png" width="720">


## **Install:**
To install, drop the "_**efficiency-nodes-comfyui**_" folder into the "_**...\ComfyUI\ComfyUI\custom_nodes**_" directory and restart UI.
