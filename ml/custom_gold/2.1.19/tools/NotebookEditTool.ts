
import { readFile, writeFile } from 'fs/promises';
import { existsSync } from 'fs';
import { basename } from 'path';

interface NotebookCell {
    cell_type: "code" | "markdown";
    source: string | string[];
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

            // Target index adjustment for insertion
            if (edit_mode === "insert" && cell_id && targetIndex !== -1) {
                targetIndex += 1;
            } else if (edit_mode === "insert" && !cell_id) {
                targetIndex = 0;
            }

            let activeMode = edit_mode;
            if (activeMode === 'replace' && (targetIndex === -1 || targetIndex >= notebook.cells.length)) {
                activeMode = 'insert';
                targetIndex = notebook.cells.length;
            }

            // Determine if we should use ID (nbformat >= 4.5)
            let newCellId: string | undefined = undefined;
            if (notebook.nbformat > 4 || (notebook.nbformat === 4 && notebook.nbformat_minor >= 5)) {
                if (activeMode === "insert") {
                    newCellId = Math.random().toString(36).substring(2, 15);
                } else if (cell_id) {
                    newCellId = cell_id;
                }
            }

            if (activeMode === 'delete') {
                if (targetIndex === -1 || targetIndex >= notebook.cells.length) {
                    return { is_error: true, content: `Cell with ID/index ${cell_id} not found for deletion.` };
                }
                notebook.cells.splice(targetIndex, 1);
            } else if (activeMode === 'insert') {
                const newCell: NotebookCell = {
                    cell_type: cell_type || 'code',
                    source: new_source,
                    metadata: {},
                    id: newCellId,
                    execution_count: null,
                    outputs: []
                };
                notebook.cells.splice(targetIndex, 0, newCell);
            } else {
                // Replace
                if (targetIndex === -1) {
                    return {
                        is_error: true,
                        content: `Cell ID/index is required for 'replace' mode, or cell was not found.`
                    };
                }

                const cell = notebook.cells[targetIndex];
                cell.source = new_source;
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
                content: `Successfully ${activeMode}ed cell in ${basename(notebook_path)}`
            };

        } catch (error: any) {
            return {
                is_error: true,
                content: `Failed to edit notebook: ${error.message}`
            };
        }
    }
};

