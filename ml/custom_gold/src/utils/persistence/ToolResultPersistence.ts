import * as fs from "fs/promises";
import * as path from "path";
import { formatSize } from "../shared/formatUtils.js";

const PERSISTED_OUTPUT_TAG = "<persisted-output>";
const PERSISTED_OUTPUT_END_TAG = "</persisted-output>";
const PREVIEW_LENGTH = 2000;
const TOOL_RESULTS_DIR = "tool-results";
const MAX_INLINE_SIZE = 400000; // iA2

export interface PersistedToolResult {
    filepath: string;
    originalSize: number;
    isJson: boolean;
    preview: string;
    hasMore: boolean;
    error?: string;
}

export function formatFsError(error: any): string {
    const code = error.code;
    if (code) {
        switch (code) {
            case "ENOENT": return `Directory not found: ${error.path ?? "unknown path"}`;
            case "EACCES": return `Permission denied: ${error.path ?? "unknown path"}`;
            case "ENOSPC": return "No space left on device";
            case "EROFS": return "Read-only file system";
            case "EMFILE": return "Too many open files";
            case "EEXIST": return `File already exists: ${error.path ?? "unknown path"}`;
            default: return `${code}: ${error.message}`;
        }
    }
    return error.message;
}

export async function persistToolResult(content: any, toolUseId: string, projectPath: string): Promise<PersistedToolResult> {
    const resultsDir = path.join(projectPath, ".claude", TOOL_RESULTS_DIR); // Assuming .claude dir
    try {
        await fs.mkdir(resultsDir, { recursive: true });
    } catch { }

    const isJson = typeof content !== 'string';
    const ext = isJson ? "json" : "txt";
    const filepath = path.join(resultsDir, `${toolUseId}.${ext}`);
    const data = isJson ? JSON.stringify(content, null, 2) : content;

    try {
        await fs.writeFile(filepath, data, "utf-8");
    } catch (error) {
        return {
            filepath,
            originalSize: data.length,
            isJson,
            preview: "",
            hasMore: false,
            error: formatFsError(error)
        };
    }

    const { preview, hasMore } = createPreview(data, PREVIEW_LENGTH);
    return {
        filepath,
        originalSize: data.length,
        isJson,
        preview,
        hasMore
    };
}

export function createPreview(content: string, length: number): { preview: string, hasMore: boolean } {
    if (content.length <= length) return { preview: content, hasMore: false };

    // Try to find a newline near the cut-off
    const lastNewline = content.slice(0, length).lastIndexOf('\n');
    const cutOff = lastNewline > length * 0.5 ? lastNewline : length;

    return {
        preview: content.slice(0, cutOff),
        hasMore: true
    };
}

export function formatPersistedOutput(result: PersistedToolResult): string {
    let output = `${PERSISTED_OUTPUT_TAG}\n`;
    output += `Output too large (${formatSize(result.originalSize)}). Full output saved to: ${result.filepath}\n\n`;
    output += `Preview (first ${formatSize(PREVIEW_LENGTH)}):\n`;
    output += result.preview;
    output += result.hasMore ? `\n...` : `\n`;
    output += PERSISTED_OUTPUT_END_TAG;
    return output;
}

export async function maybePersistToolResult(result: any, toolName: string, projectPath: string): Promise<any> {
    const content = result.content;
    if (!content) return result;

    const size = typeof content === 'string' ? content.length : JSON.stringify(content).length;
    if (size <= MAX_INLINE_SIZE) return result;

    // Check feature flag if applicable (gZ("tengu_tool_result_persistence"))
    // Assuming enabled for now or handled upstream

    const persisted = await persistToolResult(content, result.tool_use_id, projectPath);
    if (persisted.error) return result; // Fallback to inline if error? Or keep error?

    const newContent = formatPersistedOutput(persisted);
    return {
        ...result,
        content: newContent
    };
}

