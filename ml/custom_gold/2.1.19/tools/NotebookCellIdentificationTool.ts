/**
 * File: src/tools/NotebookCellIdentificationTool.ts
 * Role: Lists cells in a Jupyter Notebook with their IDs and types.
 */

import { readFile, writeFile } from 'fs/promises';
import { existsSync } from 'fs';
import { basename } from 'path';

interface NotebookCell {
    cell_type: "code" | "markdown";
    source: string[];
    metadata: any;
    id?: string;
    [key: string]: any;
}

interface Notebook {
    cells: NotebookCell[];
    metadata: any;
    nbformat: number;
    nbformat_minor: number;
}

export interface NotebookCellIdentificationInput {
    notebook_path: string;
}

export const NotebookCellIdentificationTool = {
    name: "NotebookCellIdentification",
    description: "Identifies cells in a Jupyter Notebook, returning a list of cell IDs, types, and content snippets.",
    async call(input: NotebookCellIdentificationInput) {
        const { notebook_path } = input;

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

            let modified = false;
            const cellSummaries = notebook.cells.map((cell, index) => {
                // Ensure ID exists for nbformat 4.5+ compatibility
                if (!cell.id) {
                    cell.id = Math.random().toString(36).substring(2, 10);
                    modified = true;
                }

                const sourceText = cell.source.join('');
                const snippet = sourceText.length > 100 ? sourceText.slice(0, 100) + "..." : sourceText;

                return {
                    index,
                    id: cell.id,
                    type: cell.cell_type,
                    snippet: snippet.replace(/\n/g, ' ')
                };
            });

            if (modified) {
                await writeFile(notebook_path, JSON.stringify(notebook, null, 1), 'utf-8');
            }

            const header = "| Index | ID | Type | Content Snippet |\n|-------|----|------|-----------------|\n";
            const rows = cellSummaries.map(s => `| ${s.index} | ${s.id} | ${s.type} | ${s.snippet} |`).join('\n');

            return {
                is_error: false,
                content: `Cells in ${basename(notebook_path)}:\n\n${header}${rows}`
            };

        } catch (error: any) {
            return {
                is_error: true,
                content: `Failed to identify notebook cells: ${error.message}`
            };
        }
    }
};
