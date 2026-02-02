/**
 * File: src/tools/FileEditTool.ts
 * Role: Edits files using search and replace.
 */

import { promises as fs } from 'fs';
import { resolve, isAbsolute } from 'path';

export interface FileEditInput {
    file_path: string;
    old_string: string;
    new_string: string;
    replace_all?: boolean;
}

export const FileEditTool = {
    name: "FileEdit",
    description: "Edit an existing file.",
    isConcurrencySafe: false,
    async call(input: FileEditInput, context: any) {
        let { file_path, old_string, new_string, replace_all } = input;
        const cwd = context.cwd || process.cwd();

        if (!isAbsolute(file_path)) {
            file_path = resolve(cwd, file_path);
        }

        try {
            const content = await fs.readFile(file_path, 'utf8');

            if (!content.includes(old_string)) {
                return {
                    is_error: true,
                    content: `Could not find exact match for old_string in ${file_path}. No changes made.`
                };
            }

            let newContent;
            if (replace_all) {
                newContent = content.replaceAll(old_string, new_string);
            } else {
                newContent = content.replace(old_string, new_string);
            }

            await fs.writeFile(file_path, newContent, 'utf8');

            return `Successfully updated ${file_path}`;

        } catch (error: any) {
            return {
                is_error: true,
                content: `Failed to edit file ${file_path}: ${error.message}`
            };
        }
    }
};
