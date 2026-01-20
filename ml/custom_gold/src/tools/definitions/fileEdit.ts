import { z } from "zod";
import fs from "fs/promises";

import { Tool } from "./tool.js";
import { resolvePath } from "../../utils/shared/pathUtils.js";

const FileEditInputSchema = z.object({
    file_path: z.string().describe("The absolute path to the file to modify"),
    old_string: z.string().describe("The text to replace"),
    new_string: z.string().describe("The text to replace it with (must be different from old_string)"),
    replace_all: z.boolean().optional().describe("Replace all occurences of old_string (default false)")
});

const FileEditOutputSchema = z.object({
    success: z.boolean(),
    message: z.string().optional(),
    diff: z.string().optional()
});

export const FileEditTool: Tool = {
    name: "FileEdit",
    strict: true,
    input_examples: [
        { path: "/path/to/file.txt", old_string: "foo", new_string: "bar" },
        { path: "/path/to/file.txt", old_string: "foo", new_string: "bar", replace_all: true }
    ],
    description: async () => "Edit a file by replacing a specific string with a new string. This is efficient for making small changes to large files.",
    userFacingName: () => "FileEdit",
    getToolUseSummary: (input: any) => `Editing ${input.path}`,
    prompt: async () => "Edit a file by replacing text.",
    isEnabled: () => true,
    inputSchema: FileEditInputSchema,
    outputSchema: FileEditOutputSchema,
    isConcurrencySafe: () => true,
    isReadOnly: () => false,
    isSearchOrReadCommand: () => ({ isSearch: false, isRead: false }),

    async call(input: any) {
        const { path: filePath, old_string, new_string, replace_all } = input;
        const absPath = resolvePath(filePath);

        try {
            const content = await fs.readFile(absPath, 'utf8');

            if (!content.includes(old_string)) {
                return {
                    is_error: true,
                    content: `Could not find the string "${old_string}" in ${filePath}`
                };
            }

            let newContent;
            if (replace_all) {
                newContent = content.split(old_string).join(new_string);
            } else {
                newContent = content.replace(old_string, new_string);
            }

            await fs.writeFile(absPath, newContent, 'utf8');

            return {
                data: {
                    success: true,
                    message: `Successfully replaced text in ${filePath}`
                }
            };
        } catch (error: any) {
            return {
                is_error: true,
                content: `Error editing file: ${error.message}`
            };
        }
    },

    mapToolResultToToolResultBlockParam(result: any, toolUseId: string) {
        if (result.is_error) {
            return {
                tool_use_id: toolUseId,
                type: "tool_result",
                content: result.content,
                is_error: true
            };
        }
        return {
            tool_use_id: toolUseId,
            type: "tool_result",
            content: result.data.message
        };
    }
};
