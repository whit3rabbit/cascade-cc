/**
 * Tool definitions for interacting with the Claude Chrome extension.
 * Logic from chunk_837.ts
 */
export const CHROME_EXTENSION_TOOLS = [
    {
        name: "read_console_logs",
        description: "Read console logs from a specific browser tab.",
        inputSchema: {
            type: "object",
            properties: {
                tabId: { type: "number", description: "The tab ID to read from." },
                pattern: { type: "string", description: "Regex to filter logs (e.g. 'error|warn')." }
            },
            required: ["tabId"]
        }
    },
    {
        name: "read_network_requests",
        description: "Read HTTP network requests from a specific browser tab.",
        inputSchema: {
            type: "object",
            properties: {
                tabId: { type: "number", description: "The tab ID content to analyze." },
                urlPattern: { type: "string", description: "Optional filter for request URLs." }
            },
            required: ["tabId"]
        }
    },
    {
        name: "shortcuts_list",
        description: "List all available browser-side shortcuts and workflows.",
        inputSchema: {
            type: "object",
            properties: {
                tabId: { type: "number", description: "Tab ID for browser context." }
            },
            required: ["tabId"]
        }
    },
    {
        name: "gif_creator",
        description: "Record a GIF of the browser interactions.",
        inputSchema: {
            type: "object",
            properties: {
                tabId: { type: "number", description: "Tab ID to record" },
                action: { type: "string", enum: ["start", "stop"], description: "Start or stop recording" },
                filename: { type: "string", description: "Filename for the GIF (when stopping)" }
            },
            required: ["tabId", "action"]
        }
    },
    {
        name: "tabs_context_mcp",
        description: "Get information about open tabs in the browser.",
        inputSchema: {
            type: "object",
            properties: {},
        }
    },
    {
        name: "tabs_create_mcp",
        description: "Create a new browser tab.",
        inputSchema: {
            type: "object",
            properties: {
                url: { type: "string", description: "The URL to open." }
            },
            required: ["url"]
        }
    },
    {
        name: "javascript_tool",
        description: "Execute JavaScript in the browser context (use with caution).",
        inputSchema: {
            type: "object",
            properties: {
                tabId: { type: "number" },
                script: { type: "string" }
            },
            required: ["tabId", "script"]
        }
    }
];
