import { app } from '../../../../scripts/app.js'
import { $el } from "../../../../scripts/ui.js";

export function addStylesheet(url) {
	if (url.endsWith(".js")) {
		url = url.substr(0, url.length - 2) + "css";
	}
	$el("link", {
		parent: document.head,
		rel: "stylesheet",
		type: "text/css",
		href: url.startsWith("http") ? url : getUrl(url),
	});
}

export function getUrl(path, baseUrl) {
	if (baseUrl) {
		return new URL(path, baseUrl).toString();
	} else {
		return new URL("../" + path, import.meta.url).toString();
	}
}

export async function loadImage(url) {
	return new Promise((res, rej) => {
		const img = new Image();
		img.onload = res;
		img.onerror = rej;
		img.src = url;
	});
}

export function addMenuHandler(nodeType, cb) {

    const GROUPED_MENU_ORDER = {
        "🔄 Swap with...": 0,
        "⛓ Add link...": 1,
        "📜 Add script...": 2,
        "🔍 View model info...": 3,
        "🌱 Seed behavior...": 4,
        "📐 Set Resolution...": 5,
        "✏️ Add 𝚇 input...": 6,
        "✏️ Add 𝚈 input...": 7
    };

    const originalGetOpts = nodeType.prototype.getExtraMenuOptions;

    nodeType.prototype.getExtraMenuOptions = function () {
        let r = originalGetOpts ? originalGetOpts.apply(this, arguments) || [] : [];

        const insertOption = (option) => {
            if (GROUPED_MENU_ORDER.hasOwnProperty(option.content)) {
                // Find the right position for the option
                let targetPos = r.length; // default to the end
                
                for (let i = 0; i < r.length; i++) {
                    if (GROUPED_MENU_ORDER.hasOwnProperty(r[i].content) && 
                        GROUPED_MENU_ORDER[option.content] < GROUPED_MENU_ORDER[r[i].content]) {
                        targetPos = i;
                        break;
                    }
                }
                // Insert the option at the determined position
                r.splice(targetPos, 0, option);
            } else {
                // If the option is not in the GROUPED_MENU_ORDER, simply add it to the end
                r.push(option);
            }
        };

        cb.call(this, insertOption);

        return r;
    };
}

export function findWidgetByName(node, widgetName) {
    return node.widgets.find(widget => widget.name === widgetName);
}

// Utility functions
export function addNode(name, nextTo, options) {
    options = { select: true, shiftX: 0, shiftY: 0, before: false, ...(options || {}) };
    const node = LiteGraph.createNode(name);
    app.graph.add(node);
    node.pos = [
        nextTo.pos[0] + options.shiftX,
        nextTo.pos[1] + options.shiftY,
    ];
    if (options.select) {
        app.canvas.selectNode(node, false);
    }
    return node;
}
