import { app } from "../../scripts/app.js";

const ext = {
    name: "efficiency.previewfix",
    ws: null,
    maxCount: 0,
    currentCount: 0,
    sendBlob: false,
    startProcessing: false,
    lastBlobURL: null,
    debug: false,

    log(...args) {
        if (this.debug) console.log(...args);
    },

    error(...args) {
        if (this.debug) console.error(...args);
    },

    async sendBlobDataAsDataURL(blobURL) {
        const blob = await fetch(blobURL).then(res => res.blob());
        const reader = new FileReader();
        reader.readAsDataURL(blob);
        reader.onloadend = () => this.ws.send(reader.result);
    },

    handleCommandMessage(data) {
        Object.assign(this, {
            maxCount: data.maxCount,
            sendBlob: data.sendBlob,
            startProcessing: data.startProcessing,
            currentCount: 0
        });

        if (!this.startProcessing && this.lastBlobURL) {
            this.log("[BlobURLLogger] Revoking last Blob URL:", this.lastBlobURL);
            URL.revokeObjectURL(this.lastBlobURL);
            this.lastBlobURL = null;
        }
    },

    init() {
        this.log("[BlobURLLogger] Initializing...");

        this.ws = new WebSocket('ws://127.0.0.1:8288');

        this.ws.addEventListener('open', () => this.log('[BlobURLLogger] WebSocket connection opened.'));
        this.ws.addEventListener('error', err => this.error('[BlobURLLogger] WebSocket Error:', err));
        this.ws.addEventListener('message', (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.maxCount !== undefined && data.sendBlob !== undefined && data.startProcessing !== undefined) {
                    this.handleCommandMessage(data);
                }
            } catch (err) {
                this.error('[BlobURLLogger] Error parsing JSON:', err);
            }
        });

        const originalCreateObjectURL = URL.createObjectURL;
        URL.createObjectURL = (object) => {
            const blobURL = originalCreateObjectURL.call(this, object);
            if (blobURL.startsWith('blob:') && this.startProcessing) {
                this.log("[BlobURLLogger] Blob URL created:", blobURL);
                this.lastBlobURL = blobURL;
                if (this.sendBlob && this.currentCount < this.maxCount) {
                    this.sendBlobDataAsDataURL(blobURL);
                }
                this.currentCount++;
            }
            return blobURL;
        };

        this.log("[BlobURLLogger] Hook attached.");
    }
};

function toggleWidgetVisibility(node, widgetName, isVisible) {
    const widget = node.widgets.find(w => w.name === widgetName);
    if (widget) {
        widget.visible = isVisible;
        node.setDirtyCanvas(true);
    }
}

function handleLoraNameChange(node, loraNameWidget) {
    const isNone = loraNameWidget.value === "None";
    toggleWidgetVisibility(node, "lora_model_strength", !isNone);
    toggleWidgetVisibility(node, "lora_clip_strength", !isNone);
}

app.registerExtension({
    ...ext,
    nodeCreated(node) {
        if (node.getTitle() === "Efficient Loader") {
            const loraNameWidget = node.widgets.find(w => w.name === "lora_name");
            if (loraNameWidget) {
                handleLoraNameChange(node, loraNameWidget);
                loraNameWidget.onChange = function() {
                    handleLoraNameChange(node, this);
                };
            }
        }
    }
});
