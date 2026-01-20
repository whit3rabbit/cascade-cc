
// Logic from chunk_530.ts (LSP Formatting & Protocol)

// --- LSP Symbol Kind Name (cDA) ---
export const LSP_SYMBOL_KINDS: Record<number, string> = {
    1: "File", 2: "Module", 5: "Class", 6: "Method", 12: "Function", 13: "Variable"
};

export function getSymbolKindName(kind: number) {
    return LSP_SYMBOL_KINDS[kind] || "Unknown";
}

// --- LSP Result Formatters ---
export function formatReferences(refs: any[], projectPath: string) {
    if (!refs || refs.length === 0) return "No references found.";
    return `Found ${refs.length} references.`;
}

export function formatHover(hover: any) {
    if (!hover) return "No hover information available.";
    return hover.contents.value || hover.contents;
}

// --- LSP Protocol Mapper (V27) ---
export function createLspRequest(input: any, fileUri: string) {
    const position = { line: input.line - 1, character: input.character - 1 };

    switch (input.operation) {
        case "goToDefinition":
            return { method: "textDocument/definition", params: { textDocument: { uri: fileUri }, position } };
        case "findReferences":
            return { method: "textDocument/references", params: { textDocument: { uri: fileUri }, position, context: { includeDeclaration: true } } };
        case "hover":
            return { method: "textDocument/hover", params: { textDocument: { uri: fileUri }, position } };
        default:
            throw new Error(`Unsupported operation: ${input.operation}`);
    }
}
