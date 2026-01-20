import { McpServerManager, McpServerConfig } from "./McpServerManager.js";
import { getMcpClient } from "./McpClientManager.js";
import { log } from "../logger/loggerService.js";

const logger = log("mcp-cli-admin");

export const McpCliAdmin = {
    async addServer(name: string, config: McpServerConfig, scope: string = "user") {
        logger.info(`Adding MCP server "${name}" to scope "${scope}"`);
        await McpServerManager.addMcpServer(name, config, scope);
    },

    async removeServer(name: string, scope: string = "user") {
        logger.info(`Removing MCP server "${name}" from scope "${scope}"`);
        await McpServerManager.removeMcpServer(name, scope);
    },

    async listServers() {
        const { servers, errors } = await McpServerManager.getAllMcpServers();

        const serverList = await Promise.all(Object.entries(servers).map(async ([name, config]) => {
            const status = await this.getServerStatus(name, config);
            return {
                name,
                config,
                status,
                scope: (config as any).scope || "unknown"
            };
        }));

        return {
            servers: serverList,
            errors
        };
    },

    async getServerStatus(name: string, config: McpServerConfig): Promise<string> {
        try {
            const clientWrapper = await getMcpClient(name, config);
            if (clientWrapper.type === "connected") {
                return "✓ Connected";
            } else if (clientWrapper.type === "failed") {
                return `✗ Failed: ${clientWrapper.error?.message || "Unknown error"}`;
            } else if (clientWrapper.type === "needs-auth") {
                return "⚠ Needs authentication";
            } else if (clientWrapper.type === "disabled") {
                return "Disabled";
            }
            return "Unknown";
        } catch (err) {
            return `✗ Error: ${err instanceof Error ? err.message : String(err)}`;
        }
    }
};
