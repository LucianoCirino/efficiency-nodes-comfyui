// Additional functions and imports
import { app } from "../../../scripts/app.js";
import { addMenuHandler, findWidgetByName } from "./common/utils.js";

// A mapping for resolutions based on the type of the loader
const RESOLUTIONS = {
    "Efficient Loader": [
        {width: 512, height: 512},
        {width: 512, height: 768},
        {width: 512, height: 640},
        {width: 640, height: 512},
        {width: 640, height: 768},
        {width: 640, height: 640},
        {width: 768, height: 512},
        {width: 768, height: 768},
        {width: 768, height: 640},
    ],
    "Eff. Loader SDXL": [
        {width: 1024, height: 1024},
        {width: 1152, height: 896},
        {width: 896, height: 1152},
        {width: 1216, height: 832},
        {width: 832, height: 1216},
        {width: 1344, height: 768},
        {width: 768, height: 1344},
        {width: 1536, height: 640},
        {width: 640, height: 1536}
    ]
};

// Function to set the resolution of a node
function setNodeResolution(node, width, height) {
    let widthWidget = findWidgetByName(node, "empty_latent_width");
    let heightWidget = findWidgetByName(node, "empty_latent_height");

    if (widthWidget) {
        widthWidget.value = width;
    }

    if (heightWidget) {
        heightWidget.value = height;
    }
}

// The callback for the resolution submenu
function resolutionMenuCallback(node, width, height) {
    return function() {
        setNodeResolution(node, width, height);
    };
}

// Show the set resolution submenu
function showResolutionMenu(value, options, e, menu, node) {
    const resolutions = RESOLUTIONS[node.type];
    if (!resolutions) {
        return false;
    }

    const resolutionOptions = resolutions.map(res => ({
        content: `${res.width} x ${res.height}`,
        callback: resolutionMenuCallback(node, res.width, res.height)
    }));

    new LiteGraph.ContextMenu(resolutionOptions, {
        event: e,
        callback: null,
        parentMenu: menu,
        node: node
    });

    return false;  // This ensures the original context menu doesn't proceed
}

// Extension Definition
app.registerExtension({
    name: "efficiency.SetResolution",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (["Efficient Loader", "Eff. Loader SDXL"].includes(nodeData.name)) {
            addMenuHandler(nodeType, function (insertOption) {
                insertOption({
                    content: "📐 Set Resolution...",
                    has_submenu: true,
                    callback: showResolutionMenu
                });
            });
        }
    },
});
