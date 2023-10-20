import { app } from "../../../scripts/app.js";
import { addMenuHandler } from "./common/utils.js";
import { addNode } from "./common/utils.js";

const connectionMap = {
    "KSampler (Efficient)": ["input", 5],
    "KSampler Adv. (Efficient)": ["input", 5],
    "KSampler SDXL (Eff.)": ["input", 3],
    "XY Plot": ["output", 0],
    "Noise Control Script": ["input & output", 0],
    "HighRes-Fix Script": ["input & output", 0],
    "Tiled Upscaler Script": ["input & output", 0],
    "AnimateDiff Script": ["output", 0]
};

    /**
     * connect this node output to the input of another node
     * @method connect
     * @param {number_or_string} slot (could be the number of the slot or the string with the name of the slot)
     * @param {LGraphNode} node the target node
     * @param {number_or_string} target_slot the input slot of the target node (could be the number of the slot or the string with the name of the slot, or -1 to connect a trigger)
     * @return {Object} the link_info is created, otherwise null
    LGraphNode.prototype.connect = function(output_slot, target_node, input_slot)
    **/

function addAndConnectScriptNode(scriptType, selectedNode) {
    const selectedNodeType = connectionMap[selectedNode.type];
    const newNodeType = connectionMap[scriptType];

    // 1. Create the new node without position adjustments
    const newNode = addNode(scriptType, selectedNode, { shiftX: 0, shiftY: 0 });

    // 2. Adjust position of the new node based on conditions
    if (newNodeType[0].includes("input") && selectedNodeType[0].includes("output")) {
        newNode.pos[0] += selectedNode.size[0] + 50;
    } else if (newNodeType[0].includes("output") && selectedNodeType[0].includes("input")) {
        newNode.pos[0] -= (newNode.size[0] + 50);
    }

    // 3. Logic for connecting the nodes
    switch (selectedNodeType[0]) {
        case "output":
            if (newNodeType[0] === "input & output") {
                // For every node that was previously connected to the selectedNode's output
                const connectedNodes = selectedNode.getOutputNodes(selectedNodeType[1]);
                if (connectedNodes && connectedNodes.length) {
                    for (let connectedNode of connectedNodes) {
                        // Disconnect the node from selectedNode's output
                        selectedNode.disconnectOutput(selectedNodeType[1]);
                        // Connect the newNode's output to the previously connected node, 
                        // using the appropriate slot based on the type of the connectedNode
                        const targetSlot = (connectedNode.type in connectionMap) ? connectionMap[connectedNode.type][1] : 0;
                        newNode.connect(0, connectedNode, targetSlot);
                    }
                }
                // Connect selectedNode's output to newNode's input
                selectedNode.connect(selectedNodeType[1], newNode, newNodeType[1]);
            }
            break;

        case "input":
            if (newNodeType[0] === "output") {
                newNode.connect(0, selectedNode, selectedNodeType[1]);
            } else if (newNodeType[0] === "input & output") {
                const ogInputNode = selectedNode.getInputNode(selectedNodeType[1]);
                if (ogInputNode) {
                    ogInputNode.connect(0, newNode, 0);
                }
                newNode.connect(0, selectedNode, selectedNodeType[1]);
            }
            break;
        case "input & output":
            if (newNodeType[0] === "output") {
                newNode.connect(0, selectedNode, 0);
            } else if (newNodeType[0] === "input & output") {

                const connectedNodes = selectedNode.getOutputNodes(0);
                if (connectedNodes && connectedNodes.length) {
                    for (let connectedNode of connectedNodes) {
                        selectedNode.disconnectOutput(0);
                        newNode.connect(0, connectedNode, connectedNode.type in connectionMap ? connectionMap[connectedNode.type][1] : 0);
                    }
                }
                // Connect selectedNode's output to newNode's input
                selectedNode.connect(selectedNodeType[1], newNode, newNodeType[1]);
            }
            break;
    }

    return newNode;
}

function createScriptEntry(node, scriptType) {
    return {
        content: scriptType,
        callback: function() {
            addAndConnectScriptNode(scriptType, node);
        },
    };
}

function getScriptOptions(nodeType, node) {
    const allScriptTypes = [
        "XY Plot",
        "Noise Control Script",
        "HighRes-Fix Script",
        "Tiled Upscaler Script",
        "AnimateDiff Script"
    ];

    // Filter script types based on node type
    const scriptTypes = allScriptTypes.filter(scriptType => {
        const scriptBehavior = connectionMap[scriptType][0];
        
        if (connectionMap[nodeType][0] === "output") {
            return scriptBehavior.includes("input");  // Includes nodes that are "input" or "input & output"
        } else {
            return true;
        }
    });

    return scriptTypes.map(script => createScriptEntry(node, script));
}


function showAddScriptMenu(_, options, e, menu, node) {
    const scriptOptions = getScriptOptions(node.type, node);
    new LiteGraph.ContextMenu(scriptOptions, {
        event: e,
        callback: null,
        parentMenu: menu,
        node: node
    });
    return false;
}

// Extension Definition
app.registerExtension({
    name: "efficiency.addScripts",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (connectionMap[nodeData.name]) {
            addMenuHandler(nodeType, function(insertOption) {
                insertOption({
                    content: "📜 Add script...",
                    has_submenu: true,
                    callback: showAddScriptMenu
                });
            });
        }
    },
});

