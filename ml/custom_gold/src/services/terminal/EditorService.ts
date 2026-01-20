import { z } from "zod";
import { execSync, spawnSync } from "child_process";
import fs from "fs";
import path from "path";
import os from "os";
import { randomUUID } from "crypto";
import Ajv from "ajv";

// Logic from chunk_486.ts

/**
 * Validates a schema using AJV
 */
export function validateSchema(schema: any, data: any): { valid: boolean; errors?: string } {
    try {
        // Cast to any to avoid TS construction error with default import interop
        const ajv = new (Ajv as any)({ allErrors: true });
        if (!ajv.validateSchema(schema)) {
            return { valid: false, errors: `Invalid JSON Schema: ${ajv.errorsText(ajv.errors)}` };
        }
        const validate = ajv.compile(schema);
        const valid = validate(data);
        if (!valid) {
            return {
                valid: false,
                errors: validate.errors?.map((e: any) => `${e.instancePath || 'root'}: ${e.message}`).join(", ")
            };
        }
        return { valid: true };
    } catch (err: any) {
        return { valid: false, errors: err.message };
    }
}

// --- Structured Output Tool (ar, PC0) ---

export const StructuredOutputTool = {
    name: "StructuredOutput",
    async description() { return "Return structured output in the requested format"; },
    async prompt() {
        return "Use this tool to return your final response in the requested structured format. You MUST call this tool exactly once at the end of your response to provide the structured output.";
    },
    inputSchema: z.object({}).passthrough(),
    outputSchema: z.string().describe("Structured output tool result"),
    async call(input: any) {
        return { data: "Structured output provided successfully", structured_output: input };
    },
    async checkPermissions(input: any) {
        return { behavior: "allow", updatedInput: input };
    },
    renderToolUseMessage(input: any) {
        const keys = Object.keys(input);
        if (keys.length === 0) return null;
        if (keys.length <= 3) return keys.map(k => `${k}: ${JSON.stringify(input[k])}`).join(", ");
        return `${keys.length} fields: ${keys.slice(0, 3).join(", ")}…`;
    },
    userFacingName: () => "StructuredOutput",
    renderToolUseRejectedMessage() { return "Structured output rejected"; },
    renderToolUseErrorMessage() { return "Structured output error"; },
    renderToolUseProgressMessage() { return null; },
    renderToolResultMessage(result: any) { return result; },
    mapToolResultToToolResultBlockParam(result: any, toolUseId: string) {
        return {
            tool_use_id: toolUseId,
            type: "tool_result",
            content: result
        };
    },
    isMcp: false,
    isEnabled: () => true,
    isConcurrencySafe: () => true,
    isReadOnly: () => true,
    isDestructive: () => false,
    isOpenWorld: () => false,
};

export function createStructuredOutputTool(schema: any) {
    try {
        const ajv = new (Ajv as any)({ allErrors: true });
        if (!ajv.validateSchema(schema)) {
            throw new Error(`Invalid JSON Schema: ${ajv.errorsText(ajv.errors)}`);
        }
        const validate = ajv.compile(schema);

        return {
            ...StructuredOutputTool,
            inputJSONSchema: schema,
            async call(input: any) {
                const valid = validate(input);
                if (!valid) {
                    const errors = validate.errors?.map((e: any) => `${e.instancePath || 'root'}: ${e.message}`).join(", ");
                    throw new Error(`Output does not match required schema: ${errors}`);
                }
                return { data: "Structured output provided successfully", structured_output: input };
            }
        };
    } catch {
        return null;
    }
}

// --- Editor Service (rr, vC0) ---

// Editors that support waiting
const EDITOR_MAPPINGS: Record<string, string> = {
    code: "code -w",
    subl: "subl --wait"
};

const SUPPORTED_EDITORS = ["code", "subl", "atom", "gedit", "notepad++", "notepad"];

function isSupportedEditor(cmd: string): boolean {
    const base = cmd.split(" ")[0] ?? "";
    return SUPPORTED_EDITORS.some(e => base.includes(e));
}

function hasCommand(cmd: string): boolean {
    try {
        const checkCmd = process.platform === "win32" ? "where" : "which";
        execSync(`${checkCmd} ${cmd}`, { stdio: "ignore" });
        return true;
    } catch {
        return false;
    }
}

export function getEditorCommand(): string | undefined {
    if (process.env.VISUAL?.trim()) return process.env.VISUAL.trim();
    if (process.env.EDITOR?.trim()) return process.env.EDITOR.trim();
    if (process.platform === "win32") return "start /wait notepad";
    return ["code", "vi", "nano"].find(cmd => hasCommand(cmd));
}

function createTempFile(prefix = "claude-prompt", ext = ".md"): string {
    const uuid = randomUUID();
    return path.join(os.tmpdir(), `${prefix}-${uuid}${ext}`);
}

export function openFileInEditor(filePath: string): string | null {
    const editor = getEditorCommand();
    if (!editor) return null;
    if (!fs.existsSync(filePath)) return null;

    // Ink pausing logic (GT map) is not currently implemented in this scope
    // Simplified execution:

    // Check if we need to clear screen manually if pausing was possible
    // process.stdout.write("\x1B[?1049h\x1B[?1004l\x1B[0m\x1B[?25h\x1B[2J\x1B[H");

    try {
        const cmd = EDITOR_MAPPINGS[editor] ?? editor;
        execSync(`${cmd} "${filePath}"`, { stdio: "inherit" });
        return fs.readFileSync(filePath, "utf-8");
    } catch (err) {
        return null;
    } finally {
        // Restore screen if we cleared it
        // process.stdout.write("\x1B[?1049l\x1B[?1004h\x1B[?25l");
    }
}

export function editInEditor(content: string): string | null {
    const tempPath = createTempFile();
    try {
        fs.writeFileSync(tempPath, content, { encoding: "utf-8", flush: true });
        const result = openFileInEditor(tempPath);
        if (result === null) return null;

        // Strip trailing newline if it looks like a single line edit or specific pattern
        if (result.endsWith("\n") && !result.endsWith("\n\n")) {
            return result.slice(0, -1);
        }
        return result;
    } finally {
        try {
            if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
        } catch { }
    }
}
