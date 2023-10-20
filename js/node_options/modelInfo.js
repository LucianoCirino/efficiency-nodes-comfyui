import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";
import { ModelInfoDialog } from "./common/modelInfoDialog.js";
import { addMenuHandler } from "./common/utils.js";

const MAX_TAGS = 500;

class LoraInfoDialog extends ModelInfoDialog {
	getTagFrequency() {
		if (!this.metadata.ss_tag_frequency) return [];

		const datasets = JSON.parse(this.metadata.ss_tag_frequency);
		const tags = {};
		for (const setName in datasets) {
			const set = datasets[setName];
			for (const t in set) {
				if (t in tags) {
					tags[t] += set[t];
				} else {
					tags[t] = set[t];
				}
			}
		}

		return Object.entries(tags).sort((a, b) => b[1] - a[1]);
	}

	getResolutions() {
		let res = [];
		if (this.metadata.ss_bucket_info) {
			const parsed = JSON.parse(this.metadata.ss_bucket_info);
			if (parsed?.buckets) {
				for (const { resolution, count } of Object.values(parsed.buckets)) {
					res.push([count, `${resolution.join("x")} * ${count}`]);
				}
			}
		}
		res = res.sort((a, b) => b[0] - a[0]).map((a) => a[1]);
		let r = this.metadata.ss_resolution;
		if (r) {
			const s = r.split(",");
			const w = s[0].replace("(", "");
			const h = s[1].replace(")", "");
			res.push(`${w.trim()}x${h.trim()} (Base res)`);
		} else if ((r = this.metadata["modelspec.resolution"])) {
			res.push(r + " (Base res");
		}
		if (!res.length) {
			res.push("⚠️ Unknown");
		}
		return res;
	}

	getTagList(tags) {
		return tags.map((t) =>
			$el(
				"li.pysssss-model-tag",
				{
					dataset: {
						tag: t[0],
					},
					$: (el) => {
						el.onclick = () => {
							el.classList.toggle("pysssss-model-tag--selected");
						};
					},
				},
				[
					$el("p", {
						textContent: t[0],
					}),
					$el("span", {
						textContent: t[1],
					}),
				]
			)
		);
	}

	addTags() {
		let tags = this.getTagFrequency();
		let hasMore;
		if (tags?.length) {
			const c = tags.length;
			let list;
			if (c > MAX_TAGS) {
				tags = tags.slice(0, MAX_TAGS);
				hasMore = $el("p", [
					$el("span", { textContent: `⚠️ Only showing first ${MAX_TAGS} tags ` }),
					$el("a", {
						href: "#",
						textContent: `Show all ${c}`,
						onclick: () => {
							list.replaceChildren(...this.getTagList(this.getTagFrequency()));
							hasMore.remove();
						},
					}),
				]);
			}
			list = $el("ol.pysssss-model-tags-list", this.getTagList(tags));
			this.tags = $el("div", [list]);
		} else {
			this.tags = $el("p", { textContent: "⚠️ No tag frequency metadata found" });
		}

		this.content.append(this.tags);

		if (hasMore) {
			this.content.append(hasMore);
		}
	}

	async addInfo() {
		this.addInfoEntry("Name", this.metadata.ss_output_name || "⚠️ Unknown");
		this.addInfoEntry("Base Model", this.metadata.ss_sd_model_name || "⚠️ Unknown");
		this.addInfoEntry("Clip Skip", this.metadata.ss_clip_skip || "⚠️ Unknown");

		this.addInfoEntry(
			"Resolution",
			$el(
				"select",
				this.getResolutions().map((r) => $el("option", { textContent: r }))
			)
		);

		super.addInfo();
		const p = this.addCivitaiInfo();
		this.addTags();

		const info = await p;
		if (info) {
			$el(
				"p",
				{
					parent: this.content,
					textContent: "Trained Words: ",
				},
				[
					$el("pre", {
						textContent: info.trainedWords.join(", "),
						style: {
							whiteSpace: "pre-wrap",
							margin: "10px 0",
							background: "#222",
							padding: "5px",
							borderRadius: "5px",
							maxHeight: "250px",
							overflow: "auto",
						},
					}),
				]
			);
			$el("div", {
				parent: this.content,
				innerHTML: info.description,
				style: {
					maxHeight: "250px",
					overflow: "auto",
				},
			});
		}
	}

	createButtons() {
		const btns = super.createButtons();

		function copyTags(e, tags) {
			const textarea = $el("textarea", {
				parent: document.body,
				style: {
					position: "fixed",
				},
				textContent: tags.map((el) => el.dataset.tag).join(", "),
			});
			textarea.select();
			try {
				document.execCommand("copy");
				if (!e.target.dataset.text) {
					e.target.dataset.text = e.target.textContent;
				}
				e.target.textContent = "Copied " + tags.length + " tags";
				setTimeout(() => {
					e.target.textContent = e.target.dataset.text;
				}, 1000);
			} catch (ex) {
				prompt("Copy to clipboard: Ctrl+C, Enter", text);
			} finally {
				document.body.removeChild(textarea);
			}
		}

		btns.unshift(
			$el("button", {
				type: "button",
				textContent: "Copy Selected",
				onclick: (e) => {
					copyTags(e, [...this.tags.querySelectorAll(".pysssss-model-tag--selected")]);
				},
			}),
			$el("button", {
				type: "button",
				textContent: "Copy All",
				onclick: (e) => {
					copyTags(e, [...this.tags.querySelectorAll(".pysssss-model-tag")]);
				},
			})
		);

		return btns;
	}
}

class CheckpointInfoDialog extends ModelInfoDialog {
	async addInfo() {
		super.addInfo();
		const info = await this.addCivitaiInfo();
		if (info) {
			this.addInfoEntry("Base Model", info.baseModel || "⚠️ Unknown");

			$el("div", {
				parent: this.content,
				innerHTML: info.description,
				style: {
					maxHeight: "250px",
					overflow: "auto",
				},
			});
		}
	}
}

const generateNames = (prefix, start, end) => {
    const result = [];
    if (start < end) {
        for (let i = start; i <= end; i++) {
            result.push(`${prefix}${i}`);
        }
    } else {
        for (let i = start; i >= end; i--) {
            result.push(`${prefix}${i}`);
        }
    }
	return result
}

// NOTE: Orders reversed so they appear in ascending order
const infoHandler = {
    "Efficient Loader": {
        "loras": ["lora_name"],
        "checkpoints": ["ckpt_name"]
    },
    "Eff. Loader SDXL": {
        "checkpoints": ["refiner_ckpt_name", "base_ckpt_name"]
    },
    "LoRA Stacker": {
        "loras": generateNames("lora_name_", 50, 1)
    },
    "XY Input: LoRA": {
        "loras": generateNames("lora_name_", 50, 1)
    },
    "HighRes-Fix Script": {
        "checkpoints": ["hires_ckpt_name"]
    }
};

// Utility functions and other parts of your code remain unchanged

app.registerExtension({
    name: "efficiency.ModelInfo",
    beforeRegisterNodeDef(nodeType) {
        const types = infoHandler[nodeType.comfyClass];

        if (types) {
            addMenuHandler(nodeType, function (insertOption) { // Here, we are calling addMenuHandler
                let submenuItems = [];  // to store submenu items

                const addSubMenuOption = (type, widgetNames) => {
                    widgetNames.forEach(widgetName => {
                        const widgetValue = this.widgets.find(w => w.name === widgetName)?.value;
                        
                        // Check if widgetValue is "None"
                        if (!widgetValue || widgetValue === "None") {
                            return;
                        }
                        
                        let value = widgetValue;
                        if (value.content) {
                            value = value.content;
                        }
                        const cls = type === "loras" ? LoraInfoDialog : CheckpointInfoDialog;

                        const label = widgetName; 

                        // Push to submenuItems
                        submenuItems.push({
                            content: label,
                            callback: async () => {
                                new cls(value).show(type, value);
                            },
                        });
                    });
                };

                if (typeof types === 'object') {
                    Object.keys(types).forEach(type => {
                        addSubMenuOption(type, types[type]);
                    });
                }

                // If we have submenu items, use insertOption
                if (submenuItems.length) {
                    insertOption({ // Using insertOption here
                        content: "🔍 View model info...",
                        has_submenu: true,
                        callback: (value, options, e, menu, node) => {
                            new LiteGraph.ContextMenu(submenuItems, {
                                event: e,
                                callback: null,
                                parentMenu: menu,
                                node: node
                            });

                            return false; // This ensures the original context menu doesn't proceed
                        }
                    });
                }
            });
        }
    },
});






