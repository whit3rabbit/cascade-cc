import { z } from "zod";
import * as fs from "fs";
import * as path from "path";
import React from 'react';
import { Text, Box } from "ink";

import { getOriginalCwd } from "../../services/session/sessionStore.js";
import { Tool } from "./tool.js";

const CodeBlock = ({ code }: { code: string, language?: string }) => (
    <Box borderStyle="round" borderColor="gray" padding={1}>
        <Text>{code}</Text>
    </Box>
);

const NotebookEditInputSchema = z.object({
    notebook_path: z.string().describe("The absolute path to the Jupyter notebook file to edit (must be absolute, not relative)"),
    cell_id: z.string().optional().describe("The ID of the cell to edit. When inserting a new cell, the new cell will be inserted after the cell with this ID, or at the beginning if not specified."),
    new_source: z.string().describe("The new source for the cell"),
    cell_type: z.enum(["code", "markdown"]).optional().describe("The type of the cell (code or markdown). If not specified, it defaults to the current cell type. If using edit_mode=insert, this is required."),
    edit_mode: z.enum(["replace", "insert", "delete"]).optional().describe("The type of edit to make (replace, insert, delete). Defaults to replace.")
});

const NotebookEditOutputSchema = z.object({
    new_source: z.string().describe("The new source code that was written to the cell"),
    cell_id: z.string().optional().describe("The ID of the cell that was edited"),
    cell_type: z.enum(["code", "markdown"]).describe("The type of the cell"),
    language: z.string().describe("The programming language of the notebook"),
    edit_mode: z.string().describe("The edit mode that was used"),
    error: z.string().optional().describe("Error message if the operation failed")
});

function parseCellIndex(cellId: string): number | undefined {
    const match = cellId.match(/^cell-(\d+)$/);
    if (match && match[1]) {
        return parseInt(match[1], 10);
    }
    return undefined;
}

export const NotebookEditTool: Tool = {
    name: "NotebookEdit",
    description: async () => "Replace the contents of a specific cell in a Jupyter notebook.",
    prompt: async () => "Completely replaces the contents of a specific cell in a Jupyter notebook (.ipynb file) with new source. Jupyter notebooks are interactive documents that combine code, text, and visualizations, commonly used for data analysis and scientific computing. The notebook_path parameter must be an absolute path, not a relative path. The cell_number is 0-indexed. Use edit_mode=insert to add a new cell at the index specified by cell_number. Use edit_mode=delete to delete the cell at the index specified by cell_number.",
    userFacingName: () => "Edit Notebook",
    getToolUseSummary: (input) => {
        if (!input?.notebook_path) return "editing notebook";
        return path.relative(process.cwd(), input.notebook_path);
    },
    isEnabled: () => true,
    inputSchema: NotebookEditInputSchema,
    outputSchema: NotebookEditOutputSchema,
    isConcurrencySafe: () => false,
    isReadOnly: () => false,
    isSearchOrReadCommand: () => ({ isSearch: false, isRead: false }),

    renderToolUseMessage(input, { verbose }) {
        const { notebook_path, cell_id, new_source, cell_type, edit_mode } = input;
        if (!notebook_path || !new_source || !cell_type) return null;
        if (verbose) {
            return `${notebook_path}@${cell_id}, content: ${new_source.slice(0, 30)}…, cell_type: ${cell_type}, edit_mode: ${edit_mode ?? "replace"}`;
        }
        return `${path.relative(process.cwd(), notebook_path)}@${cell_id}`;
    },

    renderToolResultMessage(result) {
        if (result.error) {
            return <Text color="red">{result.error}</Text>;
        }
        return (
            <Box flexDirection="column">
                <Text>Updated cell <Text bold>{result.cell_id}</Text>:</Text>
                <Box marginLeft={2}>
                    <CodeBlock code={result.new_source} language="python" />
                </Box>
            </Box>
        );
    },

    async validateInput({ notebook_path, cell_type, cell_id, edit_mode = "replace" }) {
        const resolvedPath = path.isAbsolute(notebook_path) ? notebook_path : path.resolve(getOriginalCwd(), notebook_path);

        if (!fs.existsSync(resolvedPath)) {
            return { result: false, message: "Notebook file does not exist.", errorCode: 1 };
        }
        if (path.extname(resolvedPath) !== ".ipynb") {
            return { result: false, message: "File must be a Jupyter notebook (.ipynb file). For editing other file types, use the FileEdit tool.", errorCode: 2 };
        }
        if (edit_mode !== "replace" && edit_mode !== "insert" && edit_mode !== "delete") {
            return { result: false, message: "Edit mode must be replace, insert, or delete.", errorCode: 4 };
        }
        if (edit_mode === "insert" && !cell_type) {
            return { result: false, message: "Cell type is required when using edit_mode=insert.", errorCode: 5 };
        }

        try {
            const content = fs.readFileSync(resolvedPath, 'utf-8');
            const notebook = JSON.parse(content);

            if (!cell_id) {
                if (edit_mode !== "insert") {
                    return { result: false, message: "Cell ID must be specified when not inserting a new cell.", errorCode: 7 };
                }
            } else if (notebook.cells.findIndex((c: any) => c.id === cell_id) === -1) {
                const index = parseCellIndex(cell_id);
                if (index !== undefined) {
                    if (!notebook.cells[index]) {
                        return { result: false, message: `Cell with index ${index} does not exist in notebook.`, errorCode: 7 };
                    }
                } else {
                    return { result: false, message: `Cell with ID "${cell_id}" not found in notebook.`, errorCode: 8 };
                }
            }
        } catch {
            return { result: false, message: "Notebook is not valid JSON.", errorCode: 6 };
        }

        return { result: true };
    },

    async call({ notebook_path, new_source, cell_id, cell_type, edit_mode = "replace" }, _context) {
        const resolvedPath = path.isAbsolute(notebook_path) ? notebook_path : path.resolve(getOriginalCwd(), notebook_path);

        // Note: Logic for file history state update (CKA) is omitted for now as it depends on other services

        try {
            const content = fs.readFileSync(resolvedPath, 'utf-8');
            const notebook = JSON.parse(content);

            let cellIndex;
            if (!cell_id) {
                cellIndex = 0;
            } else {
                cellIndex = notebook.cells.findIndex((c: any) => c.id === cell_id);
                if (cellIndex === -1) {
                    const parsed = parseCellIndex(cell_id);
                    if (parsed !== undefined) cellIndex = parsed;
                }
                if (edit_mode === "insert") cellIndex += 1;
            }

            let mode = edit_mode;
            if (mode === "replace" && cellIndex === notebook.cells.length) {
                mode = "insert";
                if (!cell_type) cell_type = "code";
            }

            const language = notebook.metadata.language_info?.name ?? "python";
            let newCellId = undefined;

            if (notebook.nbformat > 4 || (notebook.nbformat === 4 && notebook.nbformat_minor >= 5)) {
                if (mode === "insert") {
                    newCellId = Math.random().toString(36).substring(2, 15);
                } else if (cell_id !== null) {
                    newCellId = cell_id;
                }
            }

            if (mode === "delete") {
                notebook.cells.splice(cellIndex, 1);
            } else if (mode === "insert") {
                let newCell;
                if (cell_type === "markdown") {
                    newCell = {
                        cell_type: "markdown",
                        id: newCellId,
                        source: Array.isArray(new_source) ? new_source : new_source.split('\n'),
                        metadata: {}
                    };
                } else {
                    newCell = {
                        cell_type: "code",
                        id: newCellId,
                        source: Array.isArray(new_source) ? new_source : new_source.split('\n'),
                        metadata: {},
                        execution_count: null,
                        outputs: []
                    };
                }
                notebook.cells.splice(cellIndex, 0, newCell);
            } else {
                const cell = notebook.cells[cellIndex];
                cell.source = Array.isArray(new_source) ? new_source : new_source.split('\n');
                if (cell.cell_type === "code") {
                    cell.execution_count = null;
                    cell.outputs = [];
                }
                if (cell_type && cell_type !== cell.cell_type) {
                    cell.cell_type = cell_type;
                }
            }

            fs.writeFileSync(resolvedPath, JSON.stringify(notebook, null, 1), 'utf-8');

            return {
                data: {
                    new_source,
                    cell_type: cell_type ?? "code",
                    language,
                    edit_mode: mode,
                    cell_id: newCellId || undefined,
                    error: ""
                }
            };

        } catch (error: any) {
            return {
                data: {
                    new_source,
                    cell_type: cell_type ?? "code",
                    language: "python",
                    edit_mode: "replace",
                    error: error.message,
                    cell_id
                }
            };
        }
    },

    mapToolResultToToolResultBlockParam(result, toolUseId) {
        if (result.error) {
            return {
                tool_use_id: toolUseId,
                type: "tool_result",
                content: result.error,
                is_error: true
            };
        }

        switch (result.edit_mode) {
            case "replace":
                return {
                    tool_use_id: toolUseId,
                    type: "tool_result",
                    content: `Updated cell ${result.cell_id} with ${result.new_source}`
                };
            case "insert":
                return {
                    tool_use_id: toolUseId,
                    type: "tool_result",
                    content: `Inserted cell ${result.cell_id} with ${result.new_source}`
                };
            case "delete":
                return {
                    tool_use_id: toolUseId,
                    type: "tool_result",
                    content: `Deleted cell ${result.cell_id}`
                };
            default:
                return {
                    tool_use_id: toolUseId,
                    type: "tool_result",
                    content: "Unknown edit mode"
                };
        }
    }
};
