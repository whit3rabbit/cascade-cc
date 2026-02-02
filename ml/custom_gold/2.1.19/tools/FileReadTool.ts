/**
 * File: src/tools/FileReadTool.ts
 * Role: Reads files from the filesystem.
 */

import { promises as fs } from 'fs';
import { resolve, isAbsolute } from 'path';

export interface FileReadInput {
    file_path: string;
    offset?: number;
    limit?: number;
}

export const FileReadTool = {
    name: "FileRead",
    description: "Read the contents of a file.",
    isConcurrencySafe: true,
    prompt: `
Read the contents of a file from the local filesystem.
- You can read the entire file or a specific range of lines.
- For large files, use offset and limit to read in chunks.
- The path can be absolute or relative to the current working directory.
`,
    inputSchema: {
        type: "object",
        properties: {
            file_path: {
                type: "string",
                description: "The absolute or relative path to the file to read."
            },
            offset: {
                type: "number",
                description: "The line number to start reading from (1-indexed)."
            },
            limit: {
                type: "number",
                description: "The number of lines to read."
            }
        },
        required: ["file_path"]
    },
    async call(input: FileReadInput, context: any) {
        let { file_path, offset, limit } = input;
        const cwd = context.cwd || process.cwd();

        if (!isAbsolute(file_path)) {
            file_path = resolve(cwd, file_path);
        }

        try {
            // Check if file exists
            await fs.access(file_path);

            // Simple binary check based on extension (Gold source list)
            const binaryExtensions = new Set(["mp3", "wav", "flac", "ogg", "aac", "m4a", "wma", "aiff", "opus", "mp4", "avi", "mov", "wmv", "flv", "mkv", "webm", "m4v", "mpeg", "mpg", "zip", "rar", "tar", "gz", "bz2", "7z", "xz", "z", "tgz", "iso", "exe", "dll", "so", "dylib", "app", "msi", "deb", "rpm", "bin", "dat", "db", "sqlite", "sqlite3", "mdb", "idx", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "odt", "ods", "odp", "ttf", "otf", "woff", "woff2", "eot", "psd", "ai", "eps", "sketch", "fig", "xd", "blend", "obj", "3ds", "max", "class", "jar", "war", "pyc", "pyo", "rlib", "swf", "fla"]);
            const ext = file_path.split('.').pop()?.toLowerCase();
            if (ext && binaryExtensions.has(ext)) {
                return {
                    is_error: true,
                    content: `This tool cannot read binary files. The file appears to be a binary ${ext} file.`
                };
            }

            const content = await fs.readFile(file_path, 'utf8');
            const lines = content.split('\n');
            const totalLines = lines.length;

            let result = content;
            let meta = "";

            if (typeof offset === 'number' || typeof limit === 'number') {
                const start = offset && offset > 0 ? offset - 1 : 0; // 1-based to 0-based
                let end = limit ? start + limit : totalLines;

                if (end > totalLines) end = totalLines;

                result = lines.slice(start, end).join('\n');
                meta = `\n(Showing lines ${start + 1}-${end} of ${totalLines})`;
            }

            // Gold Standard formatting: <file_content> tags
            return `<file_content file_path="${file_path}">\n${result}\n</file_content>${meta}`;

        } catch (error: any) {
            return {
                is_error: true,
                content: `Failed to read file ${file_path}: ${error.message}`
            };
        }
    }
};
