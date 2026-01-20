import { z } from "zod";
import path from "node:path";
import fs from "node:fs";
import { Tool } from "./tool.js";
import {
    getFileDescription,
    formatByteSize,
    processImageFile,
    readTextFileRange,
    readTextFileWithLimits,
    readPdfFile,
    SYSTEM_REMINDER,
    getFileExceedsSizeError
} from "../../utils/file-system/fileReader.js";
import { formatNotebookResult, parseNotebook } from "../../utils/shared/notebookUtils.js";

const BINARY_EXTENSIONS = new Set([
    "mp3",
    "wav",
    "flac",
    "ogg",
    "aac",
    "m4a",
    "wma",
    "aiff",
    "opus",
    "mp4",
    "avi",
    "mov",
    "wmv",
    "flv",
    "mkv",
    "webm",
    "m4v",
    "mpeg",
    "mpg",
    "zip",
    "rar",
    "tar",
    "gz",
    "bz2",
    "7z",
    "xz",
    "z",
    "tgz",
    "iso",
    "exe",
    "dll",
    "so",
    "dylib",
    "app",
    "msi",
    "deb",
    "rpm",
    "bin",
    "dat",
    "db",
    "sqlite",
    "sqlite3",
    "mdb",
    "idx",
    "doc",
    "docx",
    "xls",
    "xlsx",
    "ppt",
    "pptx",
    "odt",
    "ods",
    "odp",
    "ttf",
    "otf",
    "woff",
    "woff2",
    "eot",
    "psd",
    "ai",
    "eps",
    "sketch",
    "fig",
    "xd",
    "blend",
    "obj",
    "3ds",
    "max",
    "class",
    "jar",
    "war",
    "pyc",
    "pyo",
    "rlib",
    "swf",
    "fla"
]);
const IMAGE_EXTENSIONS = new Set(["png", "jpg", "jpeg", "gif", "webp"]);

const FileReadInputSchema = z.object({
    file_path: z.string().describe("The absolute path to the file to read"),
    offset: z.number().optional().describe("The line number to start reading from. Only provide if the file is too large to read at once"),
    limit: z.number().optional().describe("The number of lines to read. Only provide if the file is too large to read at once.")
});

const FileReadOutputSchema = z.discriminatedUnion("type", [
    z.object({
        type: z.literal("text"),
        file: z.object({
            filePath: z.string().describe("The path to the file that was read"),
            content: z.string().describe("The content of the file"),
            numLines: z.number().describe("Number of lines in the returned content"),
            startLine: z.number().describe("The starting line number"),
            totalLines: z.number().describe("Total number of lines in the file")
        })
    }),
    z.object({
        type: z.literal("image"),
        file: z.object({
            base64: z.string().describe("Base64-encoded image data"),
            type: z.enum(["image/jpeg", "image/png", "image/gif", "image/webp"]).describe("The MIME type of the image"),
            originalSize: z.number().describe("Original file size in bytes"),
            dimensions: z.object({
                originalWidth: z.number().optional().describe("Original image width in pixels"),
                originalHeight: z.number().optional().describe("Original image height in pixels"),
                displayWidth: z.number().optional().describe("Displayed image width in pixels (after resizing)"),
                displayHeight: z.number().optional().describe("Displayed image height in pixels (after resizing)")
            }).optional().describe("Image dimension info for coordinate mapping")
        })
    }),
    z.object({
        type: z.literal("notebook"),
        file: z.object({
            filePath: z.string().describe("The path to the notebook file"),
            cells: z.array(z.any()).describe("Array of notebook cells")
        })
    }),
    z.object({
        type: z.literal("pdf"),
        file: z.object({
            filePath: z.string().describe("The path to the PDF file"),
            base64: z.string().describe("Base64-encoded PDF data"),
            originalSize: z.number().describe("Original file size in bytes")
        })
    })
]);

const DEFAULT_MAX_TOKENS = 25000;
const DEFAULT_MAX_BYTES = 100 * 1024;

function formatTextResult(file: { content: string; startLine: number; totalLines: number }) {
    if (file.content) return `${file.content}${SYSTEM_REMINDER}`;
    if (file.totalLines === 0) {
        return "<system-reminder>Warning: the file exists but the contents are empty.</system-reminder>";
    }
    return `<system-reminder>Warning: the file exists but is shorter than the provided offset (${file.startLine}). The file has ${file.totalLines} lines.</system-reminder>`;
}

export const FileReadTool: Tool = {
    name: "FileReadTool",
    strict: true,
    input_examples: [
        { file_path: "/Users/username/project/src/index.ts" },
        { file_path: "/Users/username/project/README.md", limit: 100, offset: 0 }
    ],
    description: async () => `Reads a file from the local filesystem.
The tool can read text files, images (png, jpg, jpeg, gif, webp), PDFs, and Jupyter Notebooks (.ipynb).
For large files, use the 'offset' and 'limit' parameters to read in chunks.
Binary files are generally not supported, except for the specific types mentioned above.`,
    prompt: async () => "Read files to understand their content.",
    inputSchema: FileReadInputSchema,
    outputSchema: FileReadOutputSchema,
    userFacingName: (input: any) => `Read ${path.basename(input.file_path || "")}`,
    getToolUseSummary: (input: any) => `Reading ${path.basename(input.file_path || "")}`,
    isEnabled: () => true,
    isConcurrencySafe: () => true,
    isReadOnly: () => true,
    isSearchOrReadCommand: () => ({ isSearch: false, isRead: true }),
    getPath: (input: any) => input.file_path || process.cwd(),

    async checkPermissions() {
        return true;
    },

    renderToolUseMessage: (input: any, options: any) => getFileDescription({ ...input, content: "" }, options),
    renderToolUseTag: () => null,
    renderToolUseProgressMessage: () => null,
    renderToolResultMessage: () => null,
    renderToolUseRejectedMessage: () => null,
    renderToolUseErrorMessage: () => null,

    async validateInput(input: any) {
        const filePath = path.resolve(input.file_path);

        if (filePath.startsWith("\\\\") || filePath.startsWith("//")) {
            return { result: true };
        }

        if (!fs.existsSync(filePath)) {
            const cwd = process.cwd();
            let message = "File does not exist.";
            if (cwd !== filePath) {
                message += ` Current working directory: ${cwd}`;
            }
            return { result: false, message, errorCode: 2 };
        }

        const ext = path.extname(filePath).toLowerCase();
        const extNoDot = ext.slice(1);

        if (BINARY_EXTENSIONS.has(extNoDot) && !IMAGE_EXTENSIONS.has(extNoDot) && extNoDot !== "pdf") {
            return {
                result: false,
                message: `This tool cannot read binary files. The file appears to be a binary ${ext} file. Please use appropriate tools for binary file analysis.`,
                errorCode: 4
            };
        }

        const stats = fs.statSync(filePath);
        if (stats.size === 0 && IMAGE_EXTENSIONS.has(extNoDot)) {
            return { result: false, message: "Empty image files cannot be processed.", errorCode: 5 };
        }

        if (!IMAGE_EXTENSIONS.has(extNoDot) && extNoDot !== "ipynb" && extNoDot !== "pdf") {
            if (stats.size > DEFAULT_MAX_BYTES && !input.offset && !input.limit) {
                return {
                    result: false,
                    message: getFileExceedsSizeError(stats.size, DEFAULT_MAX_BYTES),
                    meta: { fileSize: stats.size },
                    errorCode: 6
                };
            }
        }

        return { result: true };
    },

    async call(input: any, context: any) {
        const { file_path: filePath, offset = 1, limit } = input;
        const maxTokens = context?.fileReadingLimits?.maxTokens ?? DEFAULT_MAX_TOKENS;
        const ext = path.extname(filePath).toLowerCase().slice(1);

        if (ext === "ipynb") {
            const cells = parseNotebook(filePath);
            const json = JSON.stringify(cells);
            if (json.length > DEFAULT_MAX_BYTES) {
                throw new Error(`Notebook content (${formatByteSize(json.length)}) exceeds maximum allowed size (${formatByteSize(DEFAULT_MAX_BYTES)}). Use cat with jq to read specific portions:\n  cat "${filePath}" | jq '.cells[:20]' # First 20 cells\n  cat "${filePath}" | jq '.cells[100:120]' # Cells 100-120\n  cat "${filePath}" | jq '.cells | length' # Count total cells\n  cat "${filePath}" | jq '.cells[] | select(.cell_type=="code") | .source' # All code sources`);
            }
            await readTextFileWithLimits(json, ext, { maxSizeBytes: DEFAULT_MAX_BYTES, maxTokens });
            return {
                data: {
                    type: "notebook",
                    file: {
                        filePath,
                        cells
                    }
                }
            };
        }

        if (IMAGE_EXTENSIONS.has(ext)) {
            const imageResult = await processImageFile(filePath, maxTokens, ext);
            return {
                data: imageResult
            };
        }

        if (ext === "pdf") {
            const pdfResult = await readPdfFile(filePath);
            return {
                data: pdfResult
            };
        }

        const { content, lineCount, totalLines } = readTextFileRange(filePath, offset, limit);

        if (content.length > DEFAULT_MAX_BYTES) {
            throw new Error(getFileExceedsSizeError(content.length, DEFAULT_MAX_BYTES));
        }

        await readTextFileWithLimits(content, ext, { maxSizeBytes: DEFAULT_MAX_BYTES, maxTokens });

        return {
            data: {
                type: "text",
                file: {
                    filePath,
                    content,
                    numLines: lineCount,
                    startLine: offset,
                    totalLines
                }
            }
        };
    },

    mapToolResultToToolResultBlockParam(result: any, toolUseId: string) {
        switch (result.type) {
            case "image":
                return {
                    tool_use_id: toolUseId,
                    type: "tool_result",
                    content: [
                        {
                            type: "image",
                            source: {
                                type: "base64",
                                data: result.file.base64,
                                media_type: result.file.type
                            }
                        }
                    ]
                };
            case "notebook":
                return formatNotebookResult(result.file.cells, toolUseId);
            case "pdf":
                return {
                    tool_use_id: toolUseId,
                    type: "tool_result",
                    content: `PDF file read: ${result.file.filePath} (${formatByteSize(result.file.originalSize)})`
                };
            case "text":
            default:
                return {
                    tool_use_id: toolUseId,
                    type: "tool_result",
                    content: formatTextResult(result.file)
                };
        }
    }
};
