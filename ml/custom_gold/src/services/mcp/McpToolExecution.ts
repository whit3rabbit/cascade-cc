import { processAndSaveMcpToolResult } from "./McpContentProcessor.js";
import { log } from "../logger/loggerService.js";
import { logTelemetryEvent } from "../telemetry/telemetryInit.js";

const logger = log("mcp-tool-execution");

const LONG_RUNNING_THRESHOLD_MS = 30000;
const SLOW_TOOL_INTERVAL_MS = 50000; // dA2 logic (chunk_366)

export async function callMcpTool(client: any, toolName: string, args: any, meta?: any, signal?: AbortSignal) {
    const startTime = Date.now();
    let progressTimer: NodeJS.Timeout | undefined;

    // Log start (QQ)
    logger.info(`Calling MCP tool: ${toolName}`);

    // Set up progress logging for long running tools
    progressTimer = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        logger.info(`Tool '${toolName}' still running (${elapsed}s elapsed)`);
    }, LONG_RUNNING_THRESHOLD_MS);

    try {
        const timeout = 1200000; // w2A() -> 20 mins default? or use env var

        const result = await client.client.callTool({
            name: toolName,
            arguments: args,
            _meta: meta
        }, {
            signal: signal,
            timeout: timeout
        });

        if (progressTimer) clearInterval(progressTimer);

        const duration = Date.now() - startTime;
        const durationStr = duration < 1000 ? `${duration}ms` : `${Math.floor(duration / 1000)}s`;
        logger.info(`Tool '${toolName}' completed successfully in ${durationStr}`);

        // Telemetry for indexing tools
        // Logic from j12(Q) check: if indexing tool, log tengu_code_indexing_tool_used
        // We'll assume client.name is the hint.
        if (client.name === "code-indexing" || toolName.includes("index")) {
            logTelemetryEvent("tengu_code_indexing_tool_used", {
                tool: toolName,
                source: "mcp",
                success: true
            }).catch(() => { });
        }

        if (result.isError) {
            const errorMsg = result.content?.[0]?.text || result.error || "Unknown error";
            throw new Error(errorMsg);
        }

        return await processAndSaveMcpToolResult(result, toolName, client.name);

    } catch (error) {
        if (progressTimer) clearInterval(progressTimer);
        const duration = Date.now() - startTime;
        if (error instanceof Error && error.name !== "AbortError") {
            logger.error(`Tool '${toolName}' failed after ${Math.floor(duration / 1000)}s: ${error.message}`);
        }
        throw error;
    } finally {
        if (progressTimer) clearInterval(progressTimer);
    }
}
