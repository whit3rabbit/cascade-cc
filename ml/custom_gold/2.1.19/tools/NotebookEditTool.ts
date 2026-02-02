
import { readFile, writeFile } from 'fs/promises';
import { existsSync } from 'fs';
import { basename } from 'path';

interface NotebookCell {
    cell_type: "code" | "markdown";
    source: string[];
    metadata: any;
    execution_count?: number | null;
    outputs?: any[];
    id?: string;
}

interface Notebook {
    cells: NotebookCell[];
    metadata: any;
    nbformat: number;
    nbformat_minor: number;
}

export interface NotebookEditInput {
    notebook_path: string;
    new_source: string;
    cell_id?: string;
    cell_type?: "code" | "markdown";
    edit_mode?: "replace" | "insert" | "delete";
}

export const NotebookEditTool = {
    name: "NotebookEdit",
    description: "Edit, insert, or delete cells in a Jupyter Notebook (.ipynb)",
    isConcurrencySafe: false,
    async call(input: NotebookEditInput) {
        const { notebook_path, new_source, cell_id, cell_type, edit_mode = "replace" } = input;

        if (!existsSync(notebook_path)) {
            return {
                is_error: true,
                content: `Notebook not found: ${notebook_path}`
            };
        }

        try {
            const content = await readFile(notebook_path, 'utf-8');
            const notebook: Notebook = JSON.parse(content);

            if (!notebook.cells) {
                return {
                    is_error: true,
                    content: `Invalid notebook format: no cells found.`
                };
            }

            let targetIndex = -1;
            if (cell_id) {
                targetIndex = notebook.cells.findIndex(c => c.id === cell_id);
                if (targetIndex === -1 && /^\d+$/.test(cell_id)) {
                    targetIndex = parseInt(cell_id, 10);
                }
            } else {
                targetIndex = 0;
            }

            // Fallback for replace at end or invalid index
            let activeMode = edit_mode;
            if (activeMode === 'replace' && (targetIndex === -1 || targetIndex >= notebook.cells.length)) {
                activeMode = 'insert';
                targetIndex = notebook.cells.length;
            }

            // Split source into lines as per ipynb format
            const sourceLines = new_source.split('\n').map((line, i, arr) =>
                i === arr.length - 1 ? line : line + '\n'
            );

            if (activeMode === 'delete') {
                if (targetIndex === -1 || targetIndex >= notebook.cells.length) {
                    return { is_error: true, content: `Cell with ID/index ${cell_id} not found for deletion.` };
                }
                notebook.cells.splice(targetIndex, 1);
            } else if (activeMode === 'insert') {
                // Insert after targetIndex, or at start
                const newId = Math.random().toString(36).substring(2, 15);
                const newCell: NotebookCell = {
                    cell_type: cell_type || 'code',
                    source: sourceLines,
                    metadata: {},
                    id: newId,
                    execution_count: null,
                    outputs: []
                };
                const insertPos = cell_id ? targetIndex + 1 : 0;
                notebook.cells.splice(insertPos, 0, newCell);
            } else {
                // Replace
                if (targetIndex === -1) {
                    return {
                        is_error: true,
                        content: `Cell ID/index is required for 'replace' mode, or cell was not found.`
                    };
                }

                const cell = notebook.cells[targetIndex];
                cell.source = sourceLines;
                if (cell_type) {
                    cell.cell_type = cell_type;
                }
                if (cell.cell_type === 'code') {
                    cell.execution_count = null;
                    cell.outputs = [];
                }
            }

            await writeFile(notebook_path, JSON.stringify(notebook, null, 1), 'utf-8');

            return {
                is_error: false,
                content: `Successfully ${edit_mode}ed cell in ${basename(notebook_path)}`
            };

        } catch (error: any) {
            return {
                is_error: true,
                content: `Failed to edit notebook: ${error.message}`
            };
        }
    }
};
