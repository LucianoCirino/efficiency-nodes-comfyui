import { app } from "../../../scripts/app.js";
import { addMenuHandler } from "./common/utils.js";
import { findWidgetByName } from "./common/utils.js";

function replaceNode(oldNode, newNodeName) {
    const newNode = LiteGraph.createNode(newNodeName);
    if (!newNode) {
        return;
    }
    app.graph.add(newNode);

    newNode.pos = oldNode.pos.slice();
    newNode.size = oldNode.size.slice();

    // Transfer widget values
    const widgetMapping = {
        "ckpt_name": "base_ckpt_name",
        "vae_name": "vae_name",
        "clip_skip": "base_clip_skip",
        "positive": "positive",
        "negative": "negative",
        "prompt_style": "prompt_style",
        "empty_latent_width": "empty_latent_width",
        "empty_latent_height": "empty_latent_height",
        "batch_size": "batch_size"
    };

    let effectiveWidgetMapping = widgetMapping;

    // Invert the mapping when going from "Eff. Loader SDXL" to "Efficient Loader"
    if (oldNode.type === "Eff. Loader SDXL" && newNodeName === "Efficient Loader") {
        effectiveWidgetMapping = {};
        for (const [key, value] of Object.entries(widgetMapping)) {
            effectiveWidgetMapping[value] = key;
        }
    }

    oldNode.widgets.forEach(widget => {
        const newName = effectiveWidgetMapping[widget.name];
        if (newName) {
            const newWidget = findWidgetByName(newNode, newName);
            if (newWidget) {
                newWidget.value = widget.value;
            }
        }
    });

    // Hardcoded transfer for specific outputs based on the output names from the nodes in the image
    const outputMapping = {
        "MODEL": null,           // Not present in "Eff. Loader SDXL"
        "CONDITIONING+": null,   // Not present in "Eff. Loader SDXL"
        "CONDITIONING-": null,   // Not present in "Eff. Loader SDXL"
        "LATENT": "LATENT",
        "VAE": "VAE",
        "CLIP": null,            // Not present in "Eff. Loader SDXL"
        "DEPENDENCIES": "DEPENDENCIES"
    };

    // Transfer connections from old node outputs to new node outputs based on the outputMapping
    oldNode.outputs.forEach((output, index) => {
        if (output && output.links && outputMapping[output.name]) {
            const newOutputName = outputMapping[output.name];
            
            // If the new node does not have this output, skip
            if (newOutputName === null) {
                return;
            }
            
            const newOutputIndex = newNode.findOutputSlot(newOutputName);
            if (newOutputIndex !== -1) {
                output.links.forEach(link => {
                    const targetLinkInfo = oldNode.graph.links[link];
                    if (targetLinkInfo) {
                        const targetNode = oldNode.graph.getNodeById(targetLinkInfo.target_id);
                        if (targetNode) {
                            newNode.connect(newOutputIndex, targetNode, targetLinkInfo.target_slot);
                        }
                    }
                });
            }
        }
    });

    // Remove old node
    app.graph.remove(oldNode);
}

function replaceNodeMenuCallback(currentNode, targetNodeName) {
    return function() {
        replaceNode(currentNode, targetNodeName);
    };
}

function showSwapMenu(value, options, e, menu, node) {
    const swapOptions = [];

    if (node.type !== "Efficient Loader") {
        swapOptions.push({
            content: "Efficient Loader",
            callback: replaceNodeMenuCallback(node, "Efficient Loader")
        });
    }

    if (node.type !== "Eff. Loader SDXL") {
        swapOptions.push({
            content: "Eff. Loader SDXL",
            callback: replaceNodeMenuCallback(node, "Eff. Loader SDXL")
        });
    }

    new LiteGraph.ContextMenu(swapOptions, {
        event: e,
        callback: null,
        parentMenu: menu,
        node: node
    });

    return false;  // This ensures the original context menu doesn't proceed
}

// Extension Definition
app.registerExtension({
    name: "efficiency.SwapLoaders",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (["Efficient Loader", "Eff. Loader SDXL"].includes(nodeData.name)) {
            addMenuHandler(nodeType, function (insertOption) {
                insertOption({
                    content: "🔄 Swap with...",
                    has_submenu: true,
                    callback: showSwapMenu
                });
            });
        }
    },
});
