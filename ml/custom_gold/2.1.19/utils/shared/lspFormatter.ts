/**
 * File: src/utils/shared/lspFormatter.ts
 * Role: Formats Language Server Protocol (LSP) results (symbols, call hierarchies, etc.) for display.
 */

import { formatUri, formatLocation } from "./formatUri.js";

// --- LSP Types ---

export interface LspRange {
    start: { line: number; character: number };
    end: { line: number; character: number };
}

export interface LspLocation {
    uri: string;
    range: LspRange;
}

export interface LspSymbol {
    name: string;
    kind: number;
    location?: LspLocation;
    range?: LspRange;
    containerName?: string;
    detail?: string;
}

export interface CallHierarchyItem {
    name: string;
    kind: number;
    uri: string;
    range: LspRange;
    selectionRange: LspRange;
    detail?: string;
}

export interface CallHierarchyIncomingCall {
    from: CallHierarchyItem;
    fromRanges: LspRange[];
}

export interface CallHierarchyOutgoingCall {
    to: CallHierarchyItem;
    fromRanges: LspRange[];
}

export interface LspHover {
    contents: string | string[] | { kind: string; value: string } | { kind: string; value: string }[];
    range?: LspRange;
}

/**
 * Maps LSP SymbolKind integer to a human-readable string.
 */
export function getSymbolKindString(kind: number): string {
    const kinds: Record<number, string> = {
        1: "File", 2: "Module", 3: "Namespace", 4: "Package", 5: "Class",
        6: "Method", 7: "Property", 8: "Field", 9: "Constructor", 10: "Enum",
        11: "Interface", 12: "Function", 13: "Variable", 14: "Constant", 15: "String",
        16: "Number", 17: "Boolean", 18: "Array", 19: "Object", 20: "Key",
        21: "Snippet", 22: "Color", 23: "EnumMember", 24: "Struct", 25: "Event",
        26: "Operator", 27: "TypeParameter"
    };
    return kinds[kind] || "Unknown";
}

/**
 * Formats document symbols for display.
 */
export function formatDocumentSymbols(symbols: LspSymbol[]): string {
    if (!symbols || symbols.length === 0) {
        return "No symbols found in document.";
    }

    const output = ["Document symbols:"];
    for (const s of symbols) {
        const kind = getSymbolKindString(s.kind);
        const line = s.location ? ` - Line ${s.location.range.start.line + 1}` : "";
        output.push(`  ${s.name} (${kind})${line}`);
    }
    return output.join("\n");
}

/**
 * Extracts text content from LSP MarkupContent.
 */
function extractMarkupContent(content: any): string {
    if (Array.isArray(content)) {
        return content.map(extractMarkupContent).join("\n\n");
    }
    if (typeof content === "string") return content;
    return content?.value ?? "";
}

/**
 * Formats hover information.
 */
export function formatHover(hover: LspHover | undefined): string {
    if (!hover) {
        return "No hover information available.";
    }

    const text = extractMarkupContent(hover.contents);
    if (hover.range) {
        return `Hover info at ${hover.range.start.line + 1}:${hover.range.start.character + 1}:\n\n${text}`;
    }
    return text;
}

/**
 * Formats workspace symbols grouped by file.
 */
export function formatWorkspaceSymbols(symbols: LspSymbol[], workspaceRoot?: string): string {
    if (!symbols || symbols.length === 0) {
        return "No symbols found in workspace.";
    }

    const grouped = new Map<string, LspSymbol[]>();
    for (const s of symbols) {
        if (!s.location) continue;
        const uri = s.location.uri;
        const list = grouped.get(uri) || [];
        list.push(s);
        grouped.set(uri, list);
    }

    const output = [`Found ${symbols.length} symbols in workspace:`];
    for (const [uri, group] of grouped) {
        const displayUri = formatUri(uri, workspaceRoot);
        output.push(`\n${displayUri}:`);
        for (const s of group) {
            const kind = getSymbolKindString(s.kind);
            const line = s.location?.range.start.line !== undefined ? s.location.range.start.line + 1 : "?";
            output.push(`  ${s.name} (${kind}) - Line ${line}`);
        }
    }
    return output.join("\n");
}

/**
 * Formats incoming calls.
 */
export function formatIncomingCalls(calls: CallHierarchyIncomingCall[]): string {
    if (!calls || calls.length === 0) return "No incoming calls found.";

    const output = [`Found ${calls.length} incoming calls:`];
    for (const c of calls) {
        const kind = getSymbolKindString(c.from.kind);
        output.push(`  ${c.from.name} (${kind}) - ${c.from.uri}:${c.from.range.start.line + 1}`);
    }
    return output.join("\n");
}

/**
 * Formats outgoing calls.
 */
export function formatOutgoingCalls(calls: CallHierarchyOutgoingCall[]): string {
    if (!calls || calls.length === 0) return "No outgoing calls found.";

    const output = [`Found ${calls.length} outgoing calls:`];
    for (const c of calls) {
        const kind = getSymbolKindString(c.to.kind);
        output.push(`  ${c.to.name} (${kind}) - ${c.to.uri}:${c.to.range.start.line + 1}`);
    }
    return output.join("\n");
}

/**
 * Unified entry point for processing LSP results.
 */
export function formatLspResult(operation: string, result: any, workspaceRoot?: string): string {
    switch (operation) {
        case "documentSymbol": return formatDocumentSymbols(result);
        case "workspaceSymbol": return formatWorkspaceSymbols(result, workspaceRoot);
        case "incomingCalls": return formatIncomingCalls(result);
        case "outgoingCalls": return formatOutgoingCalls(result);
        case "hover": return formatHover(result);
        default: return JSON.stringify(result, null, 2);
    }
}
