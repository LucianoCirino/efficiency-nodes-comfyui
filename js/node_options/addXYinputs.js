import { app } from "../../../scripts/app.js";
import { addMenuHandler, addNode } from "./common/utils.js";

const nodePxOffsets = 80;

function getXYInputNodes() {
    return [
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
}

function showAddXYInputMenu(type, e, menu, node) {
    const specialNodes = [
        "XY Input: LoRA Plot",
        "XY Input: Control Net Plot",
        "XY Input: Manual XY Entry"
    ];
    
    const values = getXYInputNodes().map(nodeType => {
        return {
            content: nodeType,
            callback: function() {
                const newNode = addNode(nodeType, node);

                // Calculate the left shift based on the width of the new node
                const shiftX = -(newNode.size[0] + 35);
                newNode.pos[0] += shiftX;

                if (specialNodes.includes(nodeType)) {
                    newNode.pos[1] += 20;
                    // Connect both outputs to the XY Plot's 2nd and 3rd input.
                    newNode.connect(0, node, 1);
                    newNode.connect(1, node, 2);
                } else if (type === 'X') {
                    newNode.pos[1] += 20;
                    newNode.connect(0, node, 1);  // Connect to 2nd input
                } else {
                    newNode.pos[1] += node.size[1] + 45;
                    newNode.connect(0, node, 2);  // Connect to 3rd input
                }
            }
        };
    });

    new LiteGraph.ContextMenu(values, {
        event: e,
        callback: null,
        parentMenu: menu,
        node: node
    });
    return false;
}

app.registerExtension({
    name: "efficiency.addXYinputs",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "XY Plot") {
            addMenuHandler(nodeType, function(insertOption) {
                insertOption({
                    content: "✏️ Add 𝚇 input...",
                    has_submenu: true,
                    callback: (value, options, e, menu, node) => showAddXYInputMenu('X', e, menu, node)
                });
                insertOption({
                    content: "✏️ Add 𝚈 input...",
                    has_submenu: true,
                    callback: (value, options, e, menu, node) => showAddXYInputMenu('Y', e, menu, node)
                });
            });
        }
    },
});
