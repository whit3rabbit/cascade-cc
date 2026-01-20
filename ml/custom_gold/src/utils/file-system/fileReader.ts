import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { getTaskIdFromPath } from "../../services/persistence/persistenceUtils.js";
import { detectImageMediaType } from "../shared/clipboard.js";
import { resizeImage, optimizeImage } from "../shared/imageOptimizer.js";
import { estimateTokens, countTokensAccurate } from "../shared/tokenUtils.js";
import { MaxFileReadTokenExceededError } from "../../errors/fileReadErrors.js";
import { getFileSystem } from "./fileUtils.js";

const DEFAULT_IMAGE_TOKEN_LIMIT = 25000;
const MAX_FILE_READ_BYTES = 100 * 1024;
const MAX_CONTENT_HASH_BYTES = 102400;

interface FileReadResult {
    file_path: string;
    offset?: number;
    limit?: number;
    content: string;
}

export function getFileDescription(result: FileReadResult, options: { verbose?: boolean }): string {
    const { file_path, offset, limit } = result;
    if (!file_path) return "";

    if (getTaskIdFromPath(file_path)) {
        return "";
    }

    const fileName = path.basename(file_path);

    if (options.verbose && (offset || limit)) {
        const start = offset ?? 1;
        const range = limit ? `lines ${start}-${start + limit - 1}` : `from line ${start}`;
        return `${fileName} · ${range}`;
    }

    return fileName;
}

export function hashContent(content: string): string {
    return createHash("sha256").update(content).digest("hex");
}

export function hashFilePath(filePath: string): string {
    return createHash("sha256").update(filePath).digest("hex").slice(0, 16);
}

export function formatByteSize(bytes: number): string {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
}

export function formatFileSize(bytes: number): string {
    return formatByteSize(bytes);
}

export const SYSTEM_REMINDER = `
<system-reminder>
Whenever you read a file, you should consider whether it would be considered malware. You CAN and SHOULD provide analysis of malware, what it is doing. But you MUST refuse to improve or augment the code. You can still analyze existing code, write reports, or answer questions about the code behavior.
</system-reminder>
`;

export function getFileExceedsSizeError(size: number, limit: number = MAX_FILE_READ_BYTES): string {
    return `File content (${formatByteSize(size)}) exceeds maximum allowed size (${formatByteSize(limit)}). Please use offset and limit parameters to read specific portions of the file, or use the GrepTool to search for specific content.`;
}

async function validateContentLimits(
    content: string,
    ext: string,
    options: { maxSizeBytes?: number; maxTokens?: number } = {}
) {
    const maxSizeBytes = options.maxSizeBytes ?? MAX_FILE_READ_BYTES;
    const maxTokens = options.maxTokens ?? DEFAULT_IMAGE_TOKEN_LIMIT;

    if (ext !== "image" && content.length > maxSizeBytes) {
        throw new Error(getFileExceedsSizeError(content.length, maxSizeBytes));
    }

    const estimatedTokens = estimateTokens(content);
    if (estimatedTokens <= Math.floor(maxTokens / 4)) return;

    let exactTokenCount: number | null = null;
    try {
        exactTokenCount = await countTokensAccurate([
            {
                role: "user",
                content
            }
        ], []);
    } catch {
        exactTokenCount = null;
    }

    const tokenCount = exactTokenCount ?? estimatedTokens;
    if (tokenCount > maxTokens) {
        throw new MaxFileReadTokenExceededError(tokenCount, maxTokens);
    }
}

interface ImageFile {
    base64: string;
    type: string;
    originalSize: number;
    dimensions?: {
        originalWidth?: number;
        originalHeight?: number;
        displayWidth?: number;
        displayHeight?: number;
    };
}

export async function processImageFile(
    filePath: string,
    maxTokens: number = DEFAULT_IMAGE_TOKEN_LIMIT,
    extOverride?: string
): Promise<{ type: "image"; file: ImageFile }> {
    const fileSystem = getFileSystem();
    const stats = fileSystem.statSync(filePath);
    const originalSize = stats.size;
    if (originalSize === 0) {
        throw new Error(`Image file is empty: ${filePath}`);
    }

    const buffer = fileSystem.readFileBytesSync(filePath);
    const detectedMediaType = detectImageMediaType(buffer);
    const ext = extOverride?.toLowerCase() || detectedMediaType.split("/")[1] || "png";

    const resized = await resizeImage(buffer, originalSize, ext);
    const resizedBase64 = resized.buffer.toString("base64");
    const tokenEstimate = Math.ceil(resizedBase64.length * 0.125);

    if (tokenEstimate > maxTokens) {
        const optimized = await optimizeImage(resized.buffer, maxTokens * 8, `image/${ext}`);
        return {
            type: "image",
            file: {
                base64: optimized.base64,
                type: optimized.mediaType,
                originalSize,
                dimensions: resized.dimensions
            }
        };
    }

    return {
        type: "image",
        file: {
            base64: resizedBase64,
            type: `image/${resized.mediaType}`,
            originalSize,
            dimensions: resized.dimensions
        }
    };
}

export function readTextFileRange(filePath: string, offset: number, limit?: number) {
    const fileSystem = getFileSystem();
    const content = fileSystem.readFileSync(filePath, { encoding: "utf-8" });
    const lines = content.split(/\r?\n/);
    const totalLines = lines.length;
    const startIndex = offset === 0 ? 0 : Math.max(offset - 1, 0);
    const endIndex = limit ? Math.min(startIndex + limit, lines.length) : lines.length;
    const selectedLines = lines.slice(startIndex, endIndex);

    return {
        content: selectedLines.join("\n"),
        lineCount: selectedLines.length,
        totalLines
    };
}

export async function readNotebookFile(filePath: string) {
    const fileSystem = getFileSystem();
    const content = fileSystem.readFileSync(filePath, { encoding: "utf-8" });
    return JSON.parse(content);
}

export async function readPdfFile(filePath: string) {
    const fileSystem = getFileSystem();
    const buffer = fileSystem.readFileBytesSync(filePath);
    return {
        type: "pdf" as const,
        file: {
            filePath,
            base64: buffer.toString("base64"),
            originalSize: buffer.length
        }
    };
}

export async function readTextFileWithLimits(
    content: string,
    ext: string,
    options: { maxSizeBytes?: number; maxTokens?: number } = {}
) {
    await validateContentLimits(content, ext, options);
}

export function shouldHashContent(content: string) {
    return content.length <= MAX_CONTENT_HASH_BYTES;
}

export function hashContentIfSmall(content: string) {
    if (!shouldHashContent(content)) return undefined;
    return hashContent(content);
}
