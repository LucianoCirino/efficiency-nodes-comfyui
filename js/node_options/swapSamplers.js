import { app } from "../../../scripts/app.js";
import { addMenuHandler } from "./common/utils.js";
import { findWidgetByName } from "./common/utils.js";

function replaceNode(oldNode, newNodeName) {
    // Create new node
    const newNode = LiteGraph.createNode(newNodeName);
    if (!newNode) {
        return;
    }
    app.graph.add(newNode);

    // Position new node at the same position as the old node
    newNode.pos = oldNode.pos.slice();

    // Define widget mappings
    const mappings = {
        "KSampler (Efficient) <-> KSampler Adv. (Efficient)": {
            seed: "noise_seed",
            cfg: "cfg",
            sampler_name: "sampler_name",
            scheduler: "scheduler",
            preview_method: "preview_method",
            vae_decode: "vae_decode"
        },
        "KSampler (Efficient) <-> KSampler SDXL (Eff.)": {
            seed: "noise_seed",
            cfg: "cfg",
            sampler_name: "sampler_name",
            scheduler: "scheduler",
            preview_method: "preview_method",
            vae_decode: "vae_decode"
        },
        "KSampler Adv. (Efficient) <-> KSampler SDXL (Eff.)": {
            noise_seed: "noise_seed",
            steps: "steps",
            cfg: "cfg",
            sampler_name: "sampler_name",
            scheduler: "scheduler",
            start_at_step: "start_at_step",
            preview_method: "preview_method",
            vae_decode: "vae_decode"}
    };

    const swapKey = `${oldNode.type} <-> ${newNodeName}`;

    let widgetMapping = {};

    // Check if a reverse mapping is needed
    if (!mappings[swapKey]) {
        const reverseKey = `${newNodeName} <-> ${oldNode.type}`;
        const reverseMapping = mappings[reverseKey];
        if (reverseMapping) {
            widgetMapping = Object.entries(reverseMapping).reduce((acc, [key, value]) => {
                acc[value] = key;
                return acc;
            }, {});
        }
    } else {
        widgetMapping = mappings[swapKey];
    }

    if (oldNode.type === "KSampler (Efficient)" && (newNodeName === "KSampler Adv. (Efficient)" || newNodeName === "KSampler SDXL (Eff.)")) {
        const denoise = Math.min(Math.max(findWidgetByName(oldNode, "denoise").value, 0), 1); // Ensure denoise is between 0 and 1
        const steps = Math.min(Math.max(findWidgetByName(oldNode, "steps").value, 0), 10000); // Ensure steps is between 0 and 10000

        const total_steps = Math.floor(steps / denoise);
        const start_at_step = total_steps - steps;

        findWidgetByName(newNode, "steps").value = Math.min(Math.max(total_steps, 0), 10000); // Ensure total_steps is between 0 and 10000
        findWidgetByName(newNode, "start_at_step").value = Math.min(Math.max(start_at_step, 0), 10000); // Ensure start_at_step is between 0 and 10000
    }
    else if ((oldNode.type === "KSampler Adv. (Efficient)" || oldNode.type === "KSampler SDXL (Eff.)") && newNodeName === "KSampler (Efficient)") {
        const stepsAdv = Math.min(Math.max(findWidgetByName(oldNode, "steps").value, 0), 10000); // Ensure stepsAdv is between 0 and 10000
        const start_at_step = Math.min(Math.max(findWidgetByName(oldNode, "start_at_step").value, 0), 10000); // Ensure start_at_step is between 0 and 10000

        const denoise = Math.min(Math.max((stepsAdv - start_at_step) / stepsAdv, 0), 1); // Ensure denoise is between 0 and 1
        const stepsTotal = stepsAdv - start_at_step;

        findWidgetByName(newNode, "denoise").value = denoise;
        findWidgetByName(newNode, "steps").value = Math.min(Math.max(stepsTotal, 0), 10000); // Ensure stepsTotal is between 0 and 10000
    }

    // Transfer widget values from old node to new node
    oldNode.widgets.forEach(widget => {
        const newName = widgetMapping[widget.name];
        if (newName) {
            const newWidget = findWidgetByName(newNode, newName);
            if (newWidget) {
                newWidget.value = widget.value;
            }
        }
    });

    // Determine the starting indices based on the node types
    let oldNodeInputStartIndex = 0;
    let newNodeInputStartIndex = 0;
    let oldNodeOutputStartIndex = 0;
    let newNodeOutputStartIndex = 0;

    if (oldNode.type === "KSampler SDXL (Eff.)" || newNodeName === "KSampler SDXL (Eff.)") {
        oldNodeInputStartIndex = (oldNode.type === "KSampler SDXL (Eff.)") ? 1 : 3;
        newNodeInputStartIndex = (newNodeName === "KSampler SDXL (Eff.)") ? 1 : 3;
        oldNodeOutputStartIndex = (oldNode.type === "KSampler SDXL (Eff.)") ? 1 : 3;
        newNodeOutputStartIndex = (newNodeName === "KSampler SDXL (Eff.)") ? 1 : 3;
    }

    // Transfer connections from old node to new node
    oldNode.inputs.slice(oldNodeInputStartIndex).forEach((input, index) => {
        if (input && input.link !== null) {
            const originLinkInfo = oldNode.graph.links[input.link];
            if (originLinkInfo) {
                const originNode = oldNode.graph.getNodeById(originLinkInfo.origin_id);
                if (originNode) {
                    originNode.connect(originLinkInfo.origin_slot, newNode, index + newNodeInputStartIndex);
                }
            }
        }
    });

    oldNode.outputs.slice(oldNodeOutputStartIndex).forEach((output, index) => {
        if (output && output.links) {
            output.links.forEach(link => {
                const targetLinkInfo = oldNode.graph.links[link];
                if (targetLinkInfo) {
                    const targetNode = oldNode.graph.getNodeById(targetLinkInfo.target_id);
                    if (targetNode) {
                        newNode.connect(index + newNodeOutputStartIndex, targetNode, targetLinkInfo.target_slot);
                    }
                }
            });
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

    if (node.type !== "KSampler (Efficient)") {
        swapOptions.push({
            content: "KSampler (Efficient)",
            callback: replaceNodeMenuCallback(node, "KSampler (Efficient)")
        });
    }
    if (node.type !== "KSampler Adv. (Efficient)") {
        swapOptions.push({
            content: "KSampler Adv. (Efficient)",
            callback: replaceNodeMenuCallback(node, "KSampler Adv. (Efficient)")
        });
    }
    if (node.type !== "KSampler SDXL (Eff.)") {
        swapOptions.push({
            content: "KSampler SDXL (Eff.)",
            callback: replaceNodeMenuCallback(node, "KSampler SDXL (Eff.)")
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
    name: "efficiency.SwapSamplers",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (["KSampler (Efficient)", "KSampler Adv. (Efficient)", "KSampler SDXL (Eff.)"].includes(nodeData.name)) {
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
