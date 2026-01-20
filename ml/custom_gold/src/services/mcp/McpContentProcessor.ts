import { Buffer } from "node:buffer";
import * as path from "node:path";
import * as fs from "node:fs";
import { optimizeImage } from "../../utils/shared/imageOptimizer.js";
import { getProjectRoot } from "../../utils/shared/pathUtils.js";

const SUPPORTED_IMAGE_TYPES = new Set([
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp"
]);

const MAX_TOKEN_LIMIT = 30000; // Heuristic limit for "large" output

export async function processMcpContent(content: any, source: string): Promise<any[]> {
    switch (content.type) {
        case "text":
            return [{ type: "text", text: content.text }];
        case "image": {
            const buf = Buffer.from(String(content.data), "base64");
            const optimized = await optimizeImage(buf, undefined, content.mimeType);
            return [{
                type: "image",
                source: {
                    type: "base64",
                    media_type: optimized.mediaType,
                    data: optimized.base64
                }
            }];
        }
        case "resource": {
            const resource = content.resource;
            const context = `[Resource from ${source} at ${resource.uri}] `;

            if ("text" in resource) {
                return [{ type: "text", text: `${context}${resource.text}` }];
            } else if ("blob" in resource) {
                if (SUPPORTED_IMAGE_TYPES.has(resource.mimeType ?? "")) {
                    const buf = Buffer.from(resource.blob, "base64");
                    const optimized = await optimizeImage(buf, undefined, resource.mimeType);
                    const result: any[] = [];
                    if (context) {
                        result.push({ type: "text", text: context });
                    }
                    result.push({
                        type: "image",
                        source: {
                            type: "base64",
                            media_type: optimized.mediaType,
                            data: optimized.base64
                        }
                    });
                    return result;
                } else {
                    return [{ type: "text", text: `${context}Base64 data (${resource.mimeType || "unknown type"}) ${resource.blob}` }];
                }
            }
            return [];
        }
        case "resource_link": {
            let text = `[Resource link: ${content.resource.name}] ${content.resource.uri}`;
            if (content.resource.description) {
                text += ` (${content.resource.description})`;
            }
            return [{ type: "text", text }];
        }
        default:
            return [];
    }
}

function formatSchema(obj: any, depth = 2): string {
    if (obj === null) return "null";
    if (Array.isArray(obj)) {
        if (obj.length === 0) return "[]";
        return `[${formatSchema(obj[0], depth - 1)}]`;
    }
    if (typeof obj === "object") {
        if (depth <= 0) return "{...}";
        const entries = Object.entries(obj).slice(0, 10).map(([k, v]) => `${k}: ${formatSchema(v, depth - 1)}`);
        const ellipsis = Object.keys(obj).length > 10 ? ", ..." : "";
        return `{${entries.join(", ")}${ellipsis}}`;
    }
    return typeof obj;
}

export async function processMcpToolResult(result: any, toolName: string, serverName: string): Promise<any> {
    if (result && typeof result === "object") {
        if ("toolResult" in result) {
            return {
                content: String(result.toolResult),
                type: "toolResult"
            };
        }
        if ("structuredContent" in result && result.structuredContent !== undefined) {
            return {
                content: JSON.stringify(result.structuredContent),
                type: "structuredContent",
                schema: formatSchema(result.structuredContent)
            };
        }
        if ("content" in result && Array.isArray(result.content)) {
            const processed = (await Promise.all(result.content.map((c: any) => processMcpContent(c, serverName)))).flat();
            return {
                content: processed,
                type: "contentArray",
                schema: formatSchema(processed)
            };
        }
    }
    throw new Error(`Unexpected response format from tool ${toolName}`);
}

function hasImageContent(content: any): boolean {
    if (!content || typeof content === "string") return false;
    if (Array.isArray(content)) {
        return content.some(c => c.type === "image");
    }
    return false;
}

function checkLargeOutput(content: any): boolean {
    if (!content) return false;
    const str = typeof content === "string" ? content : JSON.stringify(content);
    return str.length > 100000; // Arbitrary 100KB limit for "large" check, or use token estimation
}

async function saveLargeOutput(content: string, filenameHint: string) {
    try {
        const projectRoot = getProjectRoot();
        const outputDir = path.join(projectRoot, ".claude", "mcp-outputs");
        if (!fs.existsSync(outputDir)) {
            fs.mkdirSync(outputDir, { recursive: true });
        }

        const timestamp = Date.now();
        const filename = `${filenameHint.replace(/[^a-zA-Z0-9-]/g, "_")}-${timestamp}.txt`;
        const filepath = path.join(outputDir, filename);

        await fs.promises.writeFile(filepath, content, "utf-8");
        return { filepath, originalSize: content.length };
    } catch (err) {
        return { error: err instanceof Error ? err.message : String(err), originalSize: content.length };
    }
}

export async function processAndSaveMcpToolResult(result: any, toolName: string, serverName: string): Promise<any> {
    const { content, type, schema } = await processMcpToolResult(result, toolName, serverName);

    // Always return content for IDE tools or if explicitly allowed
    if (serverName === "ide") return content;

    const isLarge = checkLargeOutput(content);
    const hasImage = hasImageContent(content);
    const enableLargeFiles = process.env.ENABLE_MCP_LARGE_OUTPUT_FILES === "true";

    if (!isLarge && !hasImage) return content;
    if (enableLargeFiles) return content; // If allowed, just return it? Logic in chunk_366 says if enabled, still save? "return await Y50(G)"

    // Logic from L85 seems to imply if it IS large, we try to save it.
    if (!isLarge && !hasImage) return content;

    // If we are here, we should probably save it
    const timestamp = Date.now();
    const filenameHint = `mcp-${serverName}-${toolName}`;
    const stringContent = typeof content === "string" ? content : JSON.stringify(content, null, 2);

    const saveResult = await saveLargeOutput(stringContent, filenameHint);

    if (saveResult.error) {
        return `Error: result (${stringContent.length.toLocaleString()} characters) exceeds maximum allowed size. Failed to save output to file: ${saveResult.error}. If this MCP server provides pagination or filtering tools, use them to retrieve specific portions of the data.`;
    }

    const schemaDesc = schema ? ` (schema: ${schema})` : "";
    const typeDesc = type === "toolResult" ? "Tool Result" : type;

    return `[${typeDesc}] Output saved to ${saveResult.filepath} (${saveResult.originalSize.toLocaleString()} characters)${schemaDesc}`;
}
