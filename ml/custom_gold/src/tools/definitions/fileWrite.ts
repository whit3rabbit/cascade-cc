
import { z } from "zod";
import fs from "fs";
import path from "path";
import { Tool } from "./tool.js";
import { log, logError } from "../../services/logger/loggerService.js";
import { DiagnosticsManager } from "../../services/diagnostics/DiagnosticsManager.js";
import { FileHistoryManager } from "../../services/history/FileHistoryManager.js";
import { LspManager } from "../../services/lsp/LspManager.js";
import { createPatch } from "../../utils/shared/diffUtils.js";
import { resolvePath } from "../../utils/shared/pathUtils.js";

const bZ2 = 16000;
const eJ5 = "<response clipped><NOTE>To save on context only part of this file has been shown to you. You should retry this tool after you have searched inside the file with Grep in order to find the line numbers of what you are looking for.</NOTE>";

function Wa({ content, startLine }: { content: string, startLine: number }) {
    const lines = content.split(/\r?\n/);
    return lines.map((line, i) => `${(i + startLine).toString().padStart(6)}\t${line}`).join('\n');
}

const FileWriteInputSchema = z.object({
    file_path: z.string().describe("The absolute path to the file to write (must be absolute, not relative)"),
    content: z.string().describe("The content to write to the file")
});

const FileWriteOutputSchema = z.object({
    type: z.enum(["create", "update"]).describe("Whether a new file was created or an existing file was updated"),
    filePath: z.string().describe("The path to the file that was written"),
    content: z.string().describe("The content that was written to the file"),
    structuredPatch: z.array(z.any()).describe("Diff patch showing the changes"),
    originalFile: z.string().nullable().describe("The original file content before the write (null for new files)")
});

export const FileWriteTool: Tool = {
    name: "FileWriteTool",
    strict: true,
    input_examples: [{
        file_path: "/Users/username/project/src/newFile.ts",
        content: `export function hello() {
  console.log("Hello, World!");
}`
    }],
    description: async () => "Write a file to the local filesystem.",
    userFacingName: (input: any) => `Write ${path.basename(input.file_path || '')}`,
    getToolUseSummary: (input: any) => `Writing ${path.basename(input.file_path || '')}`,
    prompt: async () => "Write files to the filesystem.",
    isEnabled: () => true,
    inputSchema: FileWriteInputSchema,
    outputSchema: FileWriteOutputSchema,
    isConcurrencySafe: () => false,
    isReadOnly: () => false,
    getPath: (input: any) => input.file_path,
    isSearchOrReadCommand: () => ({ isSearch: false, isRead: false }),

    async checkPermissions(input: any, context: any) {
        // Validation logic usually handles permissions via wF
        return true;
    },

    async validateInput({ file_path: A }: any, context: any) {
        const filePath = resolvePath(A);
        const appState = await context.getAppState();

        // Permission check stub (wF)
        // if (wF(filePath, appState.toolPermissionContext, "edit", "deny") !== null) ...

        if (!fs.existsSync(filePath)) {
            return { result: true };
        }

        const state = context.readFileState.get(filePath);
        if (!state) {
            return {
                result: false,
                message: "File has not been read yet. Read it first before writing to it.",
                errorCode: 2
            };
        }

        const mtime = fs.statSync(filePath).mtimeMs;
        if (mtime > state.timestamp) {
            return {
                result: false,
                message: "File has been modified since read, either by the user or by a linter. Read it again before attempting to write it.",
                errorCode: 3
            };
        }

        return { result: true };
    },

    async call({ file_path: A, content: Q }: any, context: any) {
        const filePath = resolvePath(A);
        const dirPath = path.dirname(filePath);
        const { readFileState, updateFileHistoryState } = context;

        // Diagnostics
        try {
            const diagManager = (DiagnosticsManager as any).getInstance?.() || new DiagnosticsManager();
            await diagManager.beforeFileEdited(filePath);
        } catch (e) {
            logError("FileWrite: Diagnostics failed", e);
        }

        const exists = fs.existsSync(filePath);
        if (exists) {
            const mtime = fs.statSync(filePath).mtimeMs;
            const state = readFileState.get(filePath);
            if (!state || mtime > state.timestamp) {
                throw new Error("File has been unexpectedly modified. Read it again before attempting to write it.");
            }
        }

        const originalContent = exists ? fs.readFileSync(filePath, "utf-8") : null;

        // History tracking
        if (updateFileHistoryState) {
            try {
                const historyManager = (FileHistoryManager as any).getInstance?.() || new FileHistoryManager();
                // await historyManager.trackFileModification(updateFileHistoryState, filePath, context.taskId);
            } catch (e) {
                logError("FileWrite: History tracking failed", e);
            }
        }

        if (!fs.existsSync(dirPath)) {
            fs.mkdirSync(dirPath, { recursive: true });
        }

        fs.writeFileSync(filePath, Q, "utf-8");

        // LSP Notification
        try {
            const lspManager = (LspManager as any).getInstance?.() || new LspManager();
            if (lspManager.changeFile) {
                await lspManager.changeFile(filePath, Q);
                await lspManager.saveFile(filePath);
            }
        } catch (e) {
            logError(`FileWrite: LSP notification failed for ${filePath}`, e);
        }

        readFileState.set(filePath, {
            content: Q,
            timestamp: fs.statSync(filePath).mtimeMs,
            offset: undefined,
            limit: undefined
        });

        // Telemetry markers (ev) would go here

        if (originalContent !== null) {
            const patch = createPatch(A, originalContent, Q);
            return {
                data: {
                    type: "update",
                    filePath: A,
                    content: Q,
                    structuredPatch: patch,
                    originalFile: originalContent
                }
            };
        }

        return {
            data: {
                type: "create",
                filePath: A,
                content: Q,
                structuredPatch: [],
                originalFile: null
            }
        };
    },

    mapToolResultToToolResultBlockParam(result: any, toolUseId: string) {
        const { filePath, content, type } = result;
        if (type === "create") {
            return {
                tool_use_id: toolUseId,
                type: "tool_result",
                content: `File created successfully at: ${filePath}`
            };
        } else {
            const lines = content.split(/\r?\n/);
            const truncated = lines.length > bZ2;
            const displayedContent = truncated ? lines.slice(0, bZ2).join('\n') + eJ5 : content;
            return {
                tool_use_id: toolUseId,
                type: "tool_result",
                content: `The file ${filePath} has been updated. Here's the result of running \`cat -n\` on a snippet of the edited file:\n${Wa({ content: displayedContent, startLine: 1 })}`
            };
        }
    }
};
