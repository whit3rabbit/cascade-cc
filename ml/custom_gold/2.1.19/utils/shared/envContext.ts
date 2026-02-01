/**
 * File: src/utils/shared/envContext.ts
 * Role: Gathers environment and telemetry context for tracking and debugging.
 */

import { extname as getFileExtension } from "node:path";
import {
    getAgentContext,
    getSessionId,
    toBoolean,
    getClaudeCodeClientType,
    isTeammate,
    getAgentId,
    getBetaFlags,
    getEntrypoint,
    getEnvContext
} from './runtimeAndEnv.js';

// --- Constants ---
const FILE_EXTENSION_MAX_LENGTH = 10;
const FILE_EXTENSION_SEPARATOR = ",";
const FILENAME_EXTENSION_SEPARATOR = ".";

const KNOWN_FILE_EXTENSIONS = new Set([
    "js", "ts", "jsx", "tsx", "py", "java", "c", "cpp", "h", "hpp", "go", "rs", "rb", "dart", "kt", "scala", "swift", "php", "cs", "lua", "pl", "sh", "bat", "ps1"
]);

export const AzureArc = "Azure Arc";
export const AWS_US_GOV = "AWS US Gov";

const CLAUDE_CODE_TELEMETRY_SURFACE = "claude-code";

/**
 * Normalizes a potential "mcp_tool" value.
 */
export function normalizeMcpTool(input: string | undefined): string | undefined {
    if (input?.startsWith("mcp__")) {
        return "mcp_tool";
    }
    return input;
}

/**
 * Extracts the file extension and normalizes it.
 */
export function normalizeFileExtension(filePath: string): string | undefined {
    const extension = getFileExtension(filePath)?.toLowerCase();

    if (!extension || extension === FILENAME_EXTENSION_SEPARATOR) {
        return undefined;
    }

    const cleanedExtension = extension.slice(1);
    if (cleanedExtension.length > FILE_EXTENSION_MAX_LENGTH) {
        return "other";
    }

    return cleanedExtension;
}

/**
 * Processes file names and extensions to extract relevant normalized extensions.
 */
export function extractAndNormalizeFileExtensions(fileNames: string, extensionString?: string): string | undefined {
    if (!fileNames.includes(FILENAME_EXTENSION_SEPARATOR) && !extensionString) {
        return undefined;
    }

    const extensions = new Set<string>();
    const parts: string[] = [];

    if (extensionString) {
        const normalized = normalizeFileExtension(extensionString);
        if (normalized) {
            extensions.add(normalized);
            parts.push(normalized);
        }
    }

    for (const nameWithExt of fileNames.split(FILE_EXTENSION_SEPARATOR)) {
        if (!nameWithExt) continue;

        const segments = nameWithExt.split(FILENAME_EXTENSION_SEPARATOR);
        if (segments.length < 2) continue;

        // Check if the "filename" part looks like a known language
        const fileName = segments[0];
        if (!KNOWN_FILE_EXTENSIONS.has(fileName.toLowerCase())) {
            // Fallback to segments[segments.length - 1] as ext
            const extPart = segments[segments.length - 1];
            if (extPart.charCodeAt(0) !== 45) { // No leading dash
                const normalized = normalizeFileExtension("." + extPart);
                if (normalized && !extensions.has(normalized)) {
                    extensions.add(normalized);
                    parts.push(normalized);
                }
            }
            continue;
        }

        for (let i = 1; i < segments.length; i++) {
            const part = segments[i];
            if (part.charCodeAt(0) === 45) continue; // Skip "-min" etc.

            const normalized = normalizeFileExtension("." + part);
            if (normalized && !extensions.has(normalized)) {
                extensions.add(normalized);
                parts.push(normalized);
            }
        }
    }
    return parts.length > 0 ? parts.join(",") : undefined;
}

/**
 * Placeholder for process metrics.
 */
export function getProcessMetrics(): any {
    return undefined;
}

import { EnvService } from '../../services/config/EnvService.js';

/**
 * Gathers the full telemetry context.
 */
export async function getTelemetryContext(options: { model?: string } = {}): Promise<any> {
    const model = options.model ? String(options.model) : getEntrypoint();
    const betas = getBetaFlags(model);
    const coreEnvContext = await getEnvContext();

    return {
        model,
        sessionId: getSessionId(),
        userType: "external",
        ...(betas.length > 0 ? { betas: betas.join(",") } : {}),
        envContext: coreEnvContext,
        ...(EnvService.get("CLAUDE_CODE_ENTRYPOINT") && { entrypoint: EnvService.get("CLAUDE_CODE_ENTRYPOINT") }),
        ...(EnvService.get("CLAUDE_AGENT_SDK_VERSION") && { agentSdkVersion: EnvService.get("CLAUDE_AGENT_SDK_VERSION") }),
        isInteractive: String(EnvService.isTruthy("CLAUDE_CODE_INTERACTIVE")),
        clientType: getClaudeCodeClientType(),
        sweBenchRunId: EnvService.get("SWE_BENCH_RUN_ID") || "",
        sweBenchInstanceId: EnvService.get("SWE_BENCH_INSTANCE_ID") || "",
        sweBenchTaskId: EnvService.get("SWE_BENCH_TASK_ID") || "",
        ...getAgentContext(),
    };
}

/**
 * Encodes telemetry metadata for logging and transport.
 */
export function encodeTelemetryMetadata(context: any, metadata: Record<string, any> = {}): Record<string, string> {
    const result: Record<string, string> = {};

    for (const [key, value] of Object.entries(metadata)) {
        if (value !== undefined) {
            result[key] = String(value);
        }
    }

    for (const [key, value] of Object.entries(context)) {
        if (value === undefined) continue;

        if (key === "envContext") {
            result.env = JSON.stringify(value);
        } else if (key === "processMetrics") {
            result.process = JSON.stringify(value);
        } else {
            result[key] = String(value);
        }
    }
    return result;
}

/**
 * Prepares a telemetry event for Statsig or other analytics providers.
 */
export function prepareTelemetryEvent(context: any, metadata: Record<string, any> = {}): any {
    const { envContext, processMetrics, ...rest } = context;

    return {
        ...metadata,
        ...rest,
        env: envContext,
        ...(processMetrics && { process: processMetrics }),
        surface: CLAUDE_CODE_TELEMETRY_SURFACE,
    };
}
