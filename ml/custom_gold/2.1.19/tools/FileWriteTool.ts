/**
 * File: src/tools/FileWriteTool.ts
 * Role: Writes content to the filesystem.
 */

import { promises as fs } from 'node:fs';
import { resolve, isAbsolute, dirname } from 'node:path';

export interface FileWriteInput {
    file_path: string;
    content: string;
}

export const FileWriteTool = {
    name: "FileWrite",
    description: "Write the contents of a file.",
    isConcurrencySafe: false,
    prompt: `
Write content to a file on the local filesystem.
- If the file exists, it will be overwritten.
- If the parent directories do not exist, they will be created.
- Ensure the path is correct before writing.
`,
    inputSchema: {
        type: "object",
        properties: {
            file_path: {
                type: "string",
                description: "The absolute or relative path to the file to write."
            },
            content: {
                type: "string",
                description: "The full content to write to the file."
            }
        },
        required: ["file_path", "content"]
    },
    async call(input: FileWriteInput, context: any) {
        let { file_path, content } = input;
        const cwd = context.cwd || process.cwd();

        if (!isAbsolute(file_path)) {
            file_path = resolve(cwd, file_path);
        }

        try {
            // Ensure directory exists
            const dir = dirname(file_path);
            await fs.mkdir(dir, { recursive: true });

            // Write file
            await fs.writeFile(file_path, content, 'utf8');

            return `Successfully wrote to ${file_path}`;
        } catch (error: any) {
            return {
                is_error: true,
                content: `Failed to write to file ${file_path}: ${error.message}`
            };
        }
    }
};
