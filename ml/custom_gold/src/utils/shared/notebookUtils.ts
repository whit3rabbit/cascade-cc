import { getFileSystem } from "../file-system/fileUtils.js";

// Assuming ig is a truncation helper, likely from chunk_1.ts or similar
// import { truncateText } from "./textUtils.js";

function truncateNotebookText(text: string | string[]): string {
    if (!text) return "";
    const content = Array.isArray(text) ? text.join("") : text;
    // Placeholder for the actual truncation logic if needed
    return content;
}

function extractImage(data: Record<string, string>) {
    if (typeof data["image/png"] === "string") {
        return {
            image_data: data["image/png"].replace(/\s/g, ""),
            media_type: "image/png"
        };
    }
    if (typeof data["image/jpeg"] === "string") {
        return {
            image_data: data["image/jpeg"].replace(/\s/g, ""),
            media_type: "image/jpeg"
        };
    }
    return undefined;
}

export function outputToContentBlocks(output: any) {
    switch (output.output_type) {
        case "stream":
            return [{
                type: "text",
                text: `\n${truncateNotebookText(output.text)}`
            }];
        case "execute_result":
        case "display_data": {
            const blocks: any[] = [];
            if (output.data?.["text/plain"]) {
                blocks.push({
                    type: "text",
                    text: `\n${truncateNotebookText(output.data["text/plain"])}`
                });
            }
            const image = output.data && extractImage(output.data);
            if (image) {
                blocks.push({
                    type: "image",
                    source: {
                        data: image.image_data,
                        media_type: image.media_type,
                        type: "base64"
                    }
                });
            }
            return blocks;
        }
        case "error":
            return [{
                type: "text",
                text: `\n${truncateNotebookText(`${output.ename}: ${output.evalue}\n${output.traceback.join("\n")}`)}`
            }];
        default:
            return [];
    }
}

export function cellToContentBlocks(cell: any) {
    const blocks: any[] = [];
    const meta: string[] = [];

    if (cell.cellType !== "code") {
        meta.push(`<cell_type>${cell.cellType}</cell_type>`);
    }
    if (cell.language !== "python" && cell.cellType === "code") {
        meta.push(`<language>${cell.language}</language>`);
    }

    blocks.push({
        type: "text",
        text: `<cell id="${cell.cell_id}">${meta.join("")}${cell.source}</cell id="${cell.cell_id}">`
    });

    if (cell.outputs?.length) {
        const outputBlocks = cell.outputs.flatMap(outputToContentBlocks);
        blocks.push(...outputBlocks);
    }

    return blocks;
}

export function parseNotebook(filePath: string, cellId?: string) {
    const fs = getFileSystem();
    const content = fs.readFileSync(filePath, { encoding: "utf-8" });
    const notebook = JSON.parse(content);
    const language = notebook.metadata.language_info?.name ?? "python";

    const processCell = (cell: any, index: number, includeLargeOutputs: boolean) => {
        const id = cell.id ?? `cell-${index}`;
        const result: any = {
            cellType: cell.cell_type,
            source: Array.isArray(cell.source) ? cell.source.join("") : cell.source,
            execution_count: cell.cell_type === "code" ? cell.execution_count || undefined : undefined,
            cell_id: id
        };

        if (cell.cell_type === "code") {
            result.language = language;
        }

        if (cell.cell_type === "code" && cell.outputs?.length) {
            const outputs = cell.outputs.map((out: any) => {
                // Basic mapping for internal represention
                if (out.output_type === "stream") return { output_type: out.output_type, text: truncateNotebookText(out.text) };
                if (out.output_type === "execute_result" || out.output_type === "display_data") {
                    return {
                        output_type: out.output_type,
                        text: truncateNotebookText(out.data?.["text/plain"]),
                        image: out.data && extractImage(out.data)
                    };
                }
                if (out.output_type === "error") {
                    return {
                        output_type: out.output_type,
                        text: truncateNotebookText(`${out.ename}: ${out.evalue}\n${out.traceback.join("\n")}`)
                    };
                }
                return out;
            });

            // Limit output size unless explicitly requested (single cell mode)
            if (!includeLargeOutputs && JSON.stringify(outputs).length > 10000) {
                result.outputs = [{
                    output_type: "stream",
                    text: `Outputs are too large to include. Use cat with jq to inspect this cell's outputs.`
                }];
            } else {
                result.outputs = outputs;
            }
        }
        return result;
    };

    if (cellId) {
        const cell = notebook.cells.find((c: any) => c.id === cellId);
        if (!cell) throw new Error(`Cell with ID "${cellId}" not found in notebook`);
        return [processCell(cell, notebook.cells.indexOf(cell), true)];
    }

    return notebook.cells.map((cell: any, index: number) => processCell(cell, index, false));
}

export function formatNotebookResult(cells: any[], toolUseId: string) {
    const blocks = cells.flatMap(cellToContentBlocks);

    // Combine consecutive text blocks
    const mergedContent = blocks.reduce((acc: any[], current: any) => {
        if (acc.length === 0) return [current];
        const last = acc[acc.length - 1];
        if (last.type === "text" && current.type === "text") {
            last.text += `\n${current.text}`;
            return acc;
        }
        return [...acc, current];
    }, []);

    return {
        tool_use_id: toolUseId,
        type: "tool_result",
        content: mergedContent
    };
}
