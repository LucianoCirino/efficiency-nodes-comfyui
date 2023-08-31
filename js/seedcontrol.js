import { app } from "../../scripts/app.js";

const LAST_SEED_BUTTON_LABEL = '🎲 Randomize / ♻️ Last Queued Seed';

const NODE_WIDGET_MAP = {
    "KSampler (Efficient)": "seed",
    "KSampler Adv. (Efficient)": "noise_seed",
    "KSampler SDXL (Eff.)": "noise_seed"
};

const SPECIFIC_WIDTH = 325; // Set to desired width

function setNodeWidthForMappedTitles(node) {
    if (NODE_WIDGET_MAP[node.getTitle()]) {
        node.setSize([SPECIFIC_WIDTH, node.size[1]]);
    }
}

class SeedControl {
    constructor(node, seedName) {
        this.lastSeed = -1;
        this.serializedCtx = {};
        this.node = node;
        this.holdFlag = false; // Flag to track if sampler_state was set to "Hold"
        this.usedLastSeedOnHoldRelease = false; // To track if we used the lastSeed after releasing hold

        let controlAfterGenerateIndex;
        this.samplerStateWidget = this.node.widgets.find(w => w.name === 'sampler_state');

        for (const [i, w] of this.node.widgets.entries()) {
            if (w.name === seedName) {
                this.seedWidget = w;
            } else if (w.name === 'control_after_generate') {
                controlAfterGenerateIndex = i;
                this.node.widgets.splice(i, 1);
            }
        }

        if (!this.seedWidget) {
            throw new Error('Something\'s wrong; expected seed widget');
        }

        this.lastSeedButton = this.node.addWidget("button", LAST_SEED_BUTTON_LABEL, null, () => {
            if (this.seedWidget.value != -1) {
                this.seedWidget.value = -1;
            } else if (this.lastSeed !== -1) {
                this.seedWidget.value = this.lastSeed;
            }
        }, { width: 50, serialize: false });

        setNodeWidthForMappedTitles(node);
        if (controlAfterGenerateIndex !== undefined) {
            const addedWidget = this.node.widgets.pop();
            this.node.widgets.splice(controlAfterGenerateIndex, 0, addedWidget);
            setNodeWidthForMappedTitles(node);
        }

        const max = Math.min(1125899906842624, this.seedWidget.options.max);
        const min = Math.max(-1125899906842624, this.seedWidget.options.min);
        const range = (max - min) / (this.seedWidget.options.step / 10);

        this.seedWidget.serializeValue = async (node, index) => {
            const currentSeed = this.seedWidget.value;
            this.serializedCtx = {
                wasRandom: currentSeed == -1,
            };

            // Check for the state transition and act accordingly.
            if (this.samplerStateWidget) {
                if (this.samplerStateWidget.value !== "Hold" && this.holdFlag && !this.usedLastSeedOnHoldRelease) {
                    this.serializedCtx.seedUsed = this.lastSeed;
                    this.usedLastSeedOnHoldRelease = true;
                    this.holdFlag = false; // Reset flag for the next cycle
                }
            }

            if (!this.usedLastSeedOnHoldRelease) {
                if (this.serializedCtx.wasRandom) {
                    this.serializedCtx.seedUsed = Math.floor(Math.random() * range) * (this.seedWidget.options.step / 10) + min;
                } else {
                    this.serializedCtx.seedUsed = this.seedWidget.value;
                }
            }

            if (node && node.widgets_values) {
                node.widgets_values[index] = this.serializedCtx.seedUsed;
            }else{
                // Update the last seed value and the button's label to show the current seed value
                this.lastSeed = this.serializedCtx.seedUsed;
                this.lastSeedButton.name = `🎲 Randomize / ♻️ ${this.lastSeed}`;
            }

            this.seedWidget.value = this.serializedCtx.seedUsed;

            if (this.serializedCtx.wasRandom) {
                this.lastSeed = this.serializedCtx.seedUsed;
                this.lastSeedButton.name = `🎲 Randomize / ♻️ ${this.lastSeed}`;
                if (this.samplerStateWidget.value === "Hold") {
                    this.holdFlag = true;
                }
            }

            if (this.usedLastSeedOnHoldRelease && this.samplerStateWidget.value !== "Hold") {
                // Reset the flag to ensure default behavior is restored
                this.usedLastSeedOnHoldRelease = false;
            }

            return this.serializedCtx.seedUsed;
        };

        this.seedWidget.afterQueued = () => {
            if (this.serializedCtx.wasRandom) {
                this.seedWidget.value = -1;
            }
            
            // Check if seed has changed to a non -1 value, and if so, update lastSeed
            if (this.seedWidget.value !== -1) {
                this.lastSeed = this.seedWidget.value;
            }

            // Update the button's label to show the current last seed value
            this.lastSeedButton.name = `🎲 Randomize / ♻️ ${this.lastSeed}`;

            this.serializedCtx = {};
        };
    }
}

app.registerExtension({
    name: "efficiency.seedcontrol",
    async beforeRegisterNodeDef(nodeType, nodeData, _app) {
        if (NODE_WIDGET_MAP[nodeData.name]) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated ? onNodeCreated.apply(this, []) : undefined;
                this.seedControl = new SeedControl(this, NODE_WIDGET_MAP[nodeData.name]);
                this.seedControl.seedWidget.value = -1;
            };
        }
    },
});