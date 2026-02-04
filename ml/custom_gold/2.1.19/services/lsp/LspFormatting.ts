/**
 * File: src/services/lsp/LspFormatting.ts
 * Role: Formats LSP results for human-readable output.
 */

import * as path from 'node:path';

function getSymbolKind(kind: number): string {
    const kinds = [
        "File", "Module", "Namespace", "Package", "Class", "Method", "Property",
        "Field", "Constructor", "Enum", "Interface", "Function", "Variable",
        "Constant", "String", "Number", "Boolean", "Array", "Object", "Key",
        "Null", "EnumMember", "Struct", "Event", "Operator", "TypeParameter"
    ];
    return kinds[kind - 1] || "Unknown";
}

function formatLocation(uri: string, cwd?: string): string {
    if (uri.startsWith('file://')) {
        const filePath = uri.replace('file://', '');
        if (cwd) {
            return path.relative(cwd, filePath);
        }
        return filePath;
    }
    return uri;
}

export function formatWorkspaceSymbolResult(result: any[], cwd?: string): string {
    if (!result || result.length === 0) {
        return "No symbols found in workspace. This may occur if the workspace is empty, or if the LSP server has not finished indexing the project.";
    }

    const validResults = result.filter((item: any) => item && item.location && item.location.uri);
    if (validResults.length === 0) {
        return "No symbols found in workspace.";
    }

    const byFile = new Map<string, any[]>();
    for (const item of validResults) {
        const file = formatLocation(item.location.uri, cwd);
        if (!byFile.has(file)) {
            byFile.set(file, []);
        }
        byFile.get(file)!.push(item);
    }

    const output = [`Found ${validResults.length} symbol${validResults.length === 1 ? "" : "s"} in workspace:`];
    for (const [file, items] of byFile) {
        output.push(`\n${file}:`);
        for (const item of items) {
            const kind = getSymbolKind(item.kind);
            const line = item.location.range.start.line + 1;
            let desc = `  ${item.name} (${kind}) - Line ${line}`;
            if (item.containerName) {
                desc += ` in ${item.containerName}`;
            }
            output.push(desc);
        }
    }
    return output.join('\n');
}

export function formatDocumentSymbolResult(result: any[]): string {
    if (!result || result.length === 0) {
        return "No symbols found in document.";
    }

    // Handle DocumentSymbol[] (hierarchical) vs SymbolInformation[] (flat)
    if (result[0] && "location" in result[0]) {
        // Flat SymbolInformation
        return formatWorkspaceSymbolResult(result);
    }

    // Hierarchical DocumentSymbol
    const output = ["Document symbols:"];
    const formatSymbol = (symbol: any, indent: string = "") => {
        const kind = getSymbolKind(symbol.kind);
        const line = symbol.range.start.line + 1;
        output.push(`${indent}${symbol.name} (${kind}) - Line ${line}`);
        if (symbol.children) {
            for (const child of symbol.children) {
                formatSymbol(child, indent + "  ");
            }
        }
    };

    for (const item of result) {
        formatSymbol(item, "  ");
    }
    return output.join('\n');
}

function formatCallHierarchyItem(item: any, cwd?: string): string {
    if (!item.uri) return `${item.name} (${getSymbolKind(item.kind)}) - <unknown location>`;
    const file = formatLocation(item.uri, cwd);
    const line = item.range.start.line + 1;
    let desc = `${item.name} (${getSymbolKind(item.kind)}) - ${file}:${line}`;
    if (item.detail) desc += ` [${item.detail}]`;
    return desc;
}

export function formatIncomingCallsResult(result: any[], cwd?: string): string {
    if (!result || result.length === 0) return "No incoming calls found.";

    // Group by file
    const byFile = new Map<string, any[]>();
    for (const item of result) {
        if (!item.from) continue;
        const file = formatLocation(item.from.uri, cwd);
        if (!byFile.has(file)) byFile.set(file, []);
        byFile.get(file)!.push(item);
    }

    const output = [`Found ${result.length} incoming call${result.length === 1 ? "" : "s"}:`];
    for (const [file, items] of byFile) {
        output.push(`\n${file}:`);
        for (const item of items) {
            const from = item.from;
            const kind = getSymbolKind(from.kind);
            const line = from.range.start.line + 1;
            let desc = `  ${from.name} (${kind}) - Line ${line}`;
            if (item.fromRanges && item.fromRanges.length > 0) {
                const calls = item.fromRanges.map((r: any) => `${r.start.line + 1}:${r.start.character + 1}`).join(", ");
                desc += ` [calls at: ${calls}]`;
            }
            output.push(desc);
        }
    }
    return output.join('\n');
}

export function formatOutgoingCallsResult(result: any[], cwd?: string): string {
    if (!result || result.length === 0) return "No outgoing calls found.";

    const byFile = new Map<string, any[]>();
    for (const item of result) {
        if (!item.to) continue;
        const file = formatLocation(item.to.uri, cwd);
        if (!byFile.has(file)) byFile.set(file, []);
        byFile.get(file)!.push(item);
    }

    const output = [`Found ${result.length} outgoing call${result.length === 1 ? "" : "s"}:`];
    for (const [file, items] of byFile) {
        output.push(`\n${file}:`);
        for (const item of items) {
            const to = item.to;
            const kind = getSymbolKind(to.kind);
            const line = to.range.start.line + 1;
            let desc = `  ${to.name} (${kind}) - Line ${line}`;
            if (item.fromRanges && item.fromRanges.length > 0) {
                const calls = item.fromRanges.map((r: any) => `${r.start.line + 1}:${r.start.character + 1}`).join(", ");
                desc += ` [called from: ${calls}]`;
            }
            output.push(desc);
        }
    }
    return output.join('\n');
}

export function formatLspResult(operation: string, result: any, cwd?: string): string {
    if (!result) return "No result.";

    // Check if result is error
    if (result.error) return `Error: ${result.error}`;

    switch (operation) {
        case 'documentSymbol':
            return formatDocumentSymbolResult(Array.isArray(result) ? result : [result]);
        case 'workspaceSymbol':
            return formatWorkspaceSymbolResult(Array.isArray(result) ? result : [result], cwd);
        case 'incomingCalls':
            return formatIncomingCallsResult(Array.isArray(result) ? result : [result], cwd);
        case 'outgoingCalls':
            return formatOutgoingCallsResult(Array.isArray(result) ? result : [result], cwd);
        case 'prepareCallHierarchy':
            if (Array.isArray(result)) {
                return result.map(item => formatCallHierarchyItem(item, cwd)).join('\n');
            }
            return formatCallHierarchyItem(result, cwd);
        case 'goToDefinition':
        case 'findReferences':
        case 'goToImplementation':
            // These usually return Location[] or LocationLink[]
            if (Array.isArray(result)) {
                if (result.length === 0) return "No locations found.";
                const output = [`Found ${result.length} location${result.length === 1 ? "" : "s"}:`];
                for (const loc of result) {
                    const uri = loc.uri || loc.targetUri;
                    const range = loc.range || loc.targetRange;
                    if (uri && range) {
                        const file = formatLocation(uri, cwd);
                        output.push(`  ${file}:${range.start.line + 1}:${range.start.character + 1}`);
                    }
                }
                return output.join('\n');
            }
            return JSON.stringify(result, null, 2);
        default:
            if (typeof result === 'string') return result;
            return JSON.stringify(result, null, 2);
    }
}
