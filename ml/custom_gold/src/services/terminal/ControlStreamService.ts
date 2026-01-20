
// Logic from chunk_598.ts (Migrations & Stdio Control Protocol)

import { randomUUID } from "node:crypto";

// --- Migration Service ---

/**
 * Handles one-time settings migrations between versions.
 */
export const MigrationService = {
    migrateSonnet45: () => {
        // Logic to clear legacy model settings for Sonnet 4.5 release
        console.log("Checking Sonnet 4.5 migration...");
    },
    migrateThinking: () => {
        // Normalizing thinking mode settings
    },
    runAll: () => {
        MigrationService.migrateSonnet45();
        MigrationService.migrateThinking();
    }
};

// --- Control Stream Service ---

/**
 * Implements a JSON-RPC-over-Stdio protocol for controlling a remote engine.
 * Used for permission prompts and MCP proxying in non-interactive environments.
 */
export class StructuredControlStream {
    private pendingRequests = new Map<string, { resolve: Function, reject: Function }>();

    constructor(private input: AsyncIterable<string>, private output: (line: string) => void) {
        this.startListening();
    }

    private async startListening() {
        for await (let line of this.input) {
            try {
                const msg = JSON.parse(line);
                if (msg.type === "control_response") {
                    const pending = this.pendingRequests.get(msg.request_id);
                    if (pending) {
                        this.pendingRequests.delete(msg.request_id);
                        pending.resolve(msg.response);
                    }
                }
            } catch (err) {
                console.error("Failed to parse control line:", err);
            }
        }
    }

    async sendRequest(subtype: string, params: any): Promise<any> {
        const requestId = randomUUID();
        const request = {
            type: "control_request",
            request_id: requestId,
            request: { subtype, ...params }
        };

        this.output(JSON.stringify(request) + "\n");

        return new Promise((resolve, reject) => {
            this.pendingRequests.set(requestId, { resolve, reject });

            // Timeout logic
            setTimeout(() => {
                if (this.pendingRequests.has(requestId)) {
                    this.pendingRequests.delete(requestId);
                    reject(new Error(`Control request ${subtype} timed out`));
                }
            }, 30000);
        });
    }

    /**
     * Proxies a tool permission check to the controlling process.
     */
    async proxyCanUseTool(toolName: string, input: any) {
        return this.sendRequest("can_use_tool", { tool_name: toolName, input });
    }
}
