import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "efficiency.previewfix",
    lastExecutedNodeId: null,
    blobsToRevoke: [], // Array to accumulate blob URLs for revocation
    debug: false,

    log(...args) {
        if (this.debug) console.log(...args);
    },

    error(...args) {
        if (this.debug) console.error(...args);
    },

    shouldRevokeBlobForNode(nodeId) {
        const node = app.graph.getNodeById(nodeId);
        
        const validTitles = [
            "KSampler (Efficient)",
            "KSampler Adv. (Efficient)",
            "KSampler SDXL (Eff.)"
        ];

        if (!node || !validTitles.includes(node.title)) {
            return false;
        }

        const getValue = name => ((node.widgets || []).find(w => w.name === name) || {}).value;
        return getValue("preview_method") !== "none" && getValue("vae_decode").includes("true");
    },

    setup() {
        // Intercepting blob creation to store and immediately revoke the last blob URL
        const originalCreateObjectURL = URL.createObjectURL;
        URL.createObjectURL = (object) => {
            const blobURL = originalCreateObjectURL(object);
            if (blobURL.startsWith('blob:')) {
                this.log("[BlobURLLogger] Blob URL created:", blobURL);
                
                // If the current node meets the criteria, add the blob URL to the revocation list
                if (this.shouldRevokeBlobForNode(this.lastExecutedNodeId)) {
                    this.blobsToRevoke.push(blobURL);
                }
            }
            return blobURL;
        };

        // Listen to the start of the node execution to revoke all accumulated blob URLs
        api.addEventListener("executing", ({ detail }) => {
            if (this.lastExecutedNodeId !== detail || detail === null) {
                this.blobsToRevoke.forEach(blob => {
                    this.log("[BlobURLLogger] Revoking Blob URL:", blob);
                    URL.revokeObjectURL(blob);
                });
                this.blobsToRevoke = []; // Clear the list after revoking all blobs
            }
            
            // Update the last executed node ID
            this.lastExecutedNodeId = detail;
        });

        this.log("[BlobURLLogger] Hook attached.");
    },
});