import { app } from "../../../scripts/app.js";
import { addMenuHandler } from "./common/utils.js";

function replaceNode(oldNode, newNodeName) {
    const newNode = LiteGraph.createNode(newNodeName);
    if (!newNode) {
        return;
    }
    app.graph.add(newNode);

    newNode.pos = oldNode.pos.slice();

    // Handle the special nodes with two outputs
    const nodesWithTwoOutputs = ["XY Input: LoRA Plot", "XY Input: Control Net Plot", "XY Input: Manual XY Entry"];
    let outputCount = nodesWithTwoOutputs.includes(oldNode.type) ? 2 : 1;

    // Transfer output connections from old node to new node
    oldNode.outputs.slice(0, outputCount).forEach((output, index) => {
        if (output && output.links) {
            output.links.forEach(link => {
                const targetLinkInfo = oldNode.graph.links[link];
                if (targetLinkInfo) {
                    const targetNode = oldNode.graph.getNodeById(targetLinkInfo.target_id);
                    if (targetNode) {
                        newNode.connect(index, targetNode, targetLinkInfo.target_slot);
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
    const xyInputNodes = [
        "XY Input: Seeds++ Batch",
        "XY Input: Add/Return Noise",
        "XY Input: Steps",
        "XY Input: CFG Scale",
        "XY Input: Sampler/Scheduler",
        "XY Input: Denoise",
        "XY Input: VAE",
        "XY Input: Prompt S/R",
        "XY Input: Aesthetic Score",
        "XY Input: Refiner On/Off",
        "XY Input: Checkpoint",
        "XY Input: Clip Skip",
        "XY Input: LoRA",
        "XY Input: LoRA Plot",
        "XY Input: LoRA Stacks",
        "XY Input: Control Net",
        "XY Input: Control Net Plot",
        "XY Input: Manual XY Entry"
    ];

    for (const nodeType of xyInputNodes) {
        if (node.type !== nodeType) {
            swapOptions.push({
                content: nodeType,
                callback: replaceNodeMenuCallback(node, nodeType)
            });
        }
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
    name: "efficiency.swapXYinputs",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name.startsWith("XY Input:")) {
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
