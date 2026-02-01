
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
            }

            // Split source into lines as per ipynb format
            const sourceLines = new_source.split('\n').map(line => line + '\n');
            // Remove last newline char from the very last line if it exists to be clean, 
            // but usually ipynb lines end with \n except maybe the last one. 
            // The split adds \n to all. 
            if (sourceLines.length > 0) {
                const last = sourceLines[sourceLines.length - 1];
                if (last.endsWith('\n\n')) { // Check for double newline
                    sourceLines[sourceLines.length - 1] = last.slice(0, -1);
                }
            }

            if (edit_mode === 'delete') {
                if (targetIndex === -1) {
                    return { is_error: true, content: `Cell with ID ${cell_id} not found for deletion.` };
                }
                notebook.cells.splice(targetIndex, 1);
            } else if (edit_mode === 'insert') {
                // Insert after cell_id, or at start if no cell_id
                const newCell: NotebookCell = {
                    cell_type: cell_type || 'code',
                    source: sourceLines,
                    metadata: {},
                    id: Math.random().toString(36).substring(2, 10), // Simple ID gen
                    execution_count: null,
                    outputs: []
                };
                const insertPos = targetIndex !== -1 ? targetIndex + 1 : 0;
                notebook.cells.splice(insertPos, 0, newCell);
            } else {
                // Replace
                if (targetIndex === -1 && cell_id) {
                    return { is_error: true, content: `Cell with ID ${cell_id} not found for replacement.` };
                }

                // If no cell_id provided for replace, logic implies appending or failing?
                // The schema description says "insert... inserts after cell with this ID".
                // "defaults to replace".
                // If I just have path and source, maybe I am replacing the *file*?
                // But the tool is structured around cells.
                // If no cell_id is provided in replace mode, we might assume specific behavior or return error.
                // Given the ambiguity, I'll error if cell_id missing for replace, unless logic suggests otherwise.
                // Use case: modifying a specific cell.
                if (targetIndex === -1) {
                    return {
                        is_error: true,
                        content: `Cell ID is required for 'replace' mode, or cell was not found.`
                    };
                }

                const cell = notebook.cells[targetIndex];
                cell.source = sourceLines;
                if (cell_type) {
                    cell.cell_type = cell_type;
                }
                // Clear outputs on code change? usually yes.
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
