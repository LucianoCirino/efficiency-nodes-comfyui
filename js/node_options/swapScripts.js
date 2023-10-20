import { app } from "../../../scripts/app.js";
import { addMenuHandler } from "./common/utils.js";

function replaceNode(oldNode, newNodeName) {
    const newNode = LiteGraph.createNode(newNodeName);
    if (!newNode) {
        return;
    }
    app.graph.add(newNode);

    newNode.pos = oldNode.pos.slice();

    // Transfer connections from old node to new node
    // XY Plot and AnimateDiff have only one output
    if(["XY Plot", "AnimateDiff Script"].includes(oldNode.type)) {
        if (oldNode.outputs[0] && oldNode.outputs[0].links) {
            oldNode.outputs[0].links.forEach(link => {
                const targetLinkInfo = oldNode.graph.links[link];
                if (targetLinkInfo) {
                    const targetNode = oldNode.graph.getNodeById(targetLinkInfo.target_id);
                    if (targetNode) {
                        newNode.connect(0, targetNode, targetLinkInfo.target_slot);
                    }
                }
            });
        }
    } else {
        // Noise Control Script, HighRes-Fix Script, and Tiled Upscaler Script have 1 input and 1 output at index 0
        if (oldNode.inputs[0] && oldNode.inputs[0].link !== null) {
            const originLinkInfo = oldNode.graph.links[oldNode.inputs[0].link];
            if (originLinkInfo) {
                const originNode = oldNode.graph.getNodeById(originLinkInfo.origin_id);
                if (originNode) {
                    originNode.connect(originLinkInfo.origin_slot, newNode, 0);
                }
            }
        }

        if (oldNode.outputs[0] && oldNode.outputs[0].links) {
            oldNode.outputs[0].links.forEach(link => {
                const targetLinkInfo = oldNode.graph.links[link];
                if (targetLinkInfo) {
                    const targetNode = oldNode.graph.getNodeById(targetLinkInfo.target_id);
                    if (targetNode) {
                        newNode.connect(0, targetNode, targetLinkInfo.target_slot);
                    }
                }
            });
        }
    }

    // Remove old node
    app.graph.remove(oldNode);
}

function replaceNodeMenuCallback(currentNode, targetNodeName) {
    return function() {
        replaceNode(currentNode, targetNodeName);
    };
}

function showSwapMenu(value, options, e, menu, node) {
    const scriptNodes = [
        "XY Plot",
        "Noise Control Script",
        "HighRes-Fix Script",
        "Tiled Upscaler Script",
        "AnimateDiff Script"
    ];

    const swapOptions = scriptNodes.filter(n => n !== node.type).map(n => ({
        content: n,
        callback: replaceNodeMenuCallback(node, n)
    }));

    new LiteGraph.ContextMenu(swapOptions, {
        event: e,
        callback: null,
        parentMenu: menu,
        node: node
    });

    return false;
}

// Extension Definition
app.registerExtension({
    name: "efficiency.SwapScripts",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (["XY Plot", "Noise Control Script", "HighRes-Fix Script", "Tiled Upscaler Script", "AnimateDiff Script"].includes(nodeData.name)) {
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
