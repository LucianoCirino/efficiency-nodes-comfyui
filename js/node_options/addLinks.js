import { app } from "../../../scripts/app.js";
import { addMenuHandler } from "./common/utils.js";
import { addNode } from "./common/utils.js";

function createKSamplerEntry(node, samplerType, subNodeType = null, isSDXL = false) {
    const samplerLabelMap = {
        "Eff": "KSampler (Efficient)",
        "Adv": "KSampler Adv. (Efficient)",
        "SDXL": "KSampler SDXL (Eff.)"
    };

    const subNodeLabelMap = {
        "XYPlot": "XY Plot",
        "NoiseControl": "Noise Control Script",
        "HiResFix": "HighRes-Fix Script",
        "TiledUpscale": "Tiled Upscaler Script",
        "AnimateDiff": "AnimateDiff Script"
    };

    const nicknameMap = {
        "KSampler (Efficient)": "KSampler",
        "KSampler Adv. (Efficient)": "KSampler(Adv)",
        "KSampler SDXL (Eff.)": "KSampler",
        "XY Plot": "XY Plot",
        "Noise Control Script": "NoiseControl",
        "HighRes-Fix Script": "HiResFix",
        "Tiled Upscaler Script": "TiledUpscale",
        "AnimateDiff Script": "AnimateDiff"
    };

    const kSamplerLabel = samplerLabelMap[samplerType];
    const subNodeLabel = subNodeLabelMap[subNodeType];

    const kSamplerNickname = nicknameMap[kSamplerLabel];
    const subNodeNickname = nicknameMap[subNodeLabel];

    const contentLabel = subNodeNickname ? `${kSamplerNickname} + ${subNodeNickname}` : kSamplerNickname;

    return {
        content: contentLabel,
        callback: function() {
            const kSamplerNode = addNode(kSamplerLabel, node, { shiftX: node.size[0] + 50 });

            // Standard connections for all samplers
            node.connect(0, kSamplerNode, 0);  // MODEL
            node.connect(1, kSamplerNode, 1);  // CONDITIONING+
            node.connect(2, kSamplerNode, 2);  // CONDITIONING-
            
            // Additional connections for non-SDXL
            if (!isSDXL) {
                node.connect(3, kSamplerNode, 3);  // LATENT
                node.connect(4, kSamplerNode, 4);  // VAE
            }

            if (subNodeLabel) {
                const subNode = addNode(subNodeLabel, node, { shiftX: 50, shiftY: node.size[1] + 50 });
                const dependencyIndex = isSDXL ? 3 : 5;
                node.connect(dependencyIndex, subNode, 0);
                subNode.connect(0, kSamplerNode, dependencyIndex);
            }
        },
    };
}

function createStackerNode(node, type) {
    const stackerLabelMap = {
        "LoRA": "LoRA Stacker",
        "ControlNet": "Control Net Stacker"
    };

    const contentLabel = stackerLabelMap[type];

    return {
        content: contentLabel,
        callback: function() {
            const stackerNode = addNode(contentLabel, node);
            
            // Calculate the left shift based on the width of the new node
            const shiftX = -(stackerNode.size[0] + 25);

            stackerNode.pos[0] += shiftX;  // Adjust the x position of the new node

            // Introduce a Y offset of 200 for ControlNet Stacker node
            if (type === "ControlNet") {
                stackerNode.pos[1] += 300;
            }

            // Connect outputs to the Efficient Loader based on type
            if (type === "LoRA") {
                stackerNode.connect(0, node, 0);
            } else if (type === "ControlNet") {
                stackerNode.connect(0, node, 1);
            }
        },
    };
}

function createXYPlotNode(node, type) {
    const contentLabel = "XY Plot";

    return {
        content: contentLabel,
        callback: function() {
            const xyPlotNode = addNode(contentLabel, node);

            // Center the X coordinate of the XY Plot node
            const centerXShift = (node.size[0] - xyPlotNode.size[0]) / 2;
            xyPlotNode.pos[0] += centerXShift;

            // Adjust the Y position to place it below the loader node
            xyPlotNode.pos[1] += node.size[1] + 60;

            // Depending on the node type, connect the appropriate output to the XY Plot node
            if (type === "Efficient") {
                node.connect(6, xyPlotNode, 0);
            } else if (type === "SDXL") {
                node.connect(3, xyPlotNode, 0);
            }
        },
    };
}

function getMenuValues(type, node) {
    const subNodeTypes = [null, "XYPlot", "NoiseControl", "HiResFix", "TiledUpscale", "AnimateDiff"];
    const excludedSubNodeTypes = ["NoiseControl", "HiResFix", "TiledUpscale", "AnimateDiff"];  // Nodes to exclude from the menu

    const menuValues = [];

    // Add the new node types to the menu first for the correct order
    menuValues.push(createStackerNode(node, "LoRA"));
    menuValues.push(createStackerNode(node, "ControlNet"));

    for (const subNodeType of subNodeTypes) {
        // Skip adding submenu items that are in the excludedSubNodeTypes array
        if (!excludedSubNodeTypes.includes(subNodeType)) {
            const menuEntry = createKSamplerEntry(node, type === "Efficient" ? "Eff" : "SDXL", subNodeType, type === "SDXL");
            menuValues.push(menuEntry);
        }
    }

    // Insert the standalone XY Plot option after the KSampler without any subNodeTypes and before any other KSamplers with subNodeTypes
    menuValues.splice(3, 0, createXYPlotNode(node, type));

    return menuValues;
}

function showAddLinkMenuCommon(value, options, e, menu, node, type) {
    const values = getMenuValues(type, node);
    new LiteGraph.ContextMenu(values, {
        event: e,
        callback: null,
        parentMenu: menu,
        node: node
    });
    return false;
}

// Extension Definition
app.registerExtension({
    name: "efficiency.addLinks",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const linkTypes = {
            "Efficient Loader": "Efficient",
            "Eff. Loader SDXL": "SDXL"
        };

        const linkType = linkTypes[nodeData.name];
        
        if (linkType) {
            addMenuHandler(nodeType, function(insertOption) {
                insertOption({
                    content: "⛓ Add link...",
                    has_submenu: true,
                    callback: (value, options, e, menu, node) => showAddLinkMenuCommon(value, options, e, menu, node, linkType)
                });
            });
        }
    },
});

