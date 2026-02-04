/**
 * File: src/tools/LspTool.ts
 * Role: Provides LSP capabilities (Go to Definition, Find References, etc.) to the LLM.
 */

import { z } from 'zod';
import { LspServerManager } from '../services/lsp/LspServerManager.js';
import { LspToolInputSchema, LspToolOutputSchema, LspOperation, LspToolResult } from '../services/terminal/schemas.js';
import { formatLspResult } from '../services/lsp/LspFormatting.js';
import * as path from 'node:path';

export const LspTool = {
    name: "LspTool",
    description: `Interact with Language Server Protocol (LSP) servers to get code intelligence features.

Supported operations:
- goToDefinition: Find where a symbol is defined
- findReferences: Find all references to a symbol
- hover: Get hover information (documentation, type info) for a symbol
- documentSymbol: Get all symbols (functions, classes, variables) in a document
- workspaceSymbol: Search for symbols across the entire workspace
- goToImplementation: Find implementations of an interface or abstract method
- prepareCallHierarchy: Get call hierarchy item at a position (functions/methods)
- incomingCalls: Find all functions/methods that call the function at a position
- outgoingCalls: Find all functions/methods called by the function at a position

All operations require:
- filePath: The file to operate on
- lineNumber: The line number (1-based, as shown in editors)
- character: The character offset (1-based, as shown in editors)`,
    inputSchema: LspToolInputSchema,
    userFacingName: () => "Language Server",

    async call(input: LspOperation, context?: any): Promise<LspToolResult> {
        const serverManager = LspServerManager.getInstance();
        const { operation, filePath, lineNumber, character } = input;
        const cwd = process.cwd();
        const absolutePath = path.resolve(cwd, filePath);

        // LSP uses 0-based positions
        const position = {
            line: lineNumber - 1,
            character: character - 1
        };

        const methodMap: Record<string, string> = {
            "goToDefinition": "textDocument/definition",
            "findReferences": "textDocument/references",
            "hover": "textDocument/hover",
            "documentSymbol": "textDocument/documentSymbol",
            "workspaceSymbol": "workspace/symbol",
            "goToImplementation": "textDocument/implementation",
            "prepareCallHierarchy": "textDocument/prepareCallHierarchy",
            "incomingCalls": "callHierarchy/incomingCalls",
            "outgoingCalls": "callHierarchy/outgoingCalls"
        };

        const method = methodMap[operation];
        let params: any;

        if (operation === "workspaceSymbol") {
            params = { query: "" }; // As seen in chunk956
        } else if (operation === "documentSymbol") {
            params = { textDocument: { uri: `file://${absolutePath}` } };
        } else {
            params = {
                textDocument: { uri: `file://${absolutePath}` },
                position
            };
        }

        if (operation === "findReferences") {
            params.context = { includeDeclaration: true };
        }

        try {
            // Special handling for call hierarchy
            let result;
            if (operation === "incomingCalls" || operation === "outgoingCalls") {
                // Must first call prepareCallHierarchy
                const prepareParams = {
                    textDocument: { uri: `file://${absolutePath}` },
                    position
                };
                const items = await serverManager.sendRequest(absolutePath, "textDocument/prepareCallHierarchy", prepareParams);
                if (!items || !Array.isArray(items) || items.length === 0) {
                    return {
                        operation,
                        result: "No call hierarchy items found at this position.",
                        filePath: absolutePath,
                        resultCount: 0
                    };
                }
                const item = items[0];
                result = await serverManager.sendRequest(absolutePath, method, { item });
            } else {
                result = await serverManager.sendRequest(absolutePath, method, params);
            }

            const formatted = formatLspResult(operation, result, cwd);
            let resultCount = 0;
            let fileCount = 0;

            if (Array.isArray(result)) {
                resultCount = result.length;
                const uniqueUris = new Set(result.map((r: any) => r.uri || r.location?.uri).filter(Boolean));
                fileCount = uniqueUris.size;
            } else if (result) {
                resultCount = 1;
                fileCount = 1;
            }

            return {
                operation,
                result: formatted,
                filePath: absolutePath,
                resultCount,
                fileCount
            };
        } catch (error: any) {
            return {
                operation,
                result: `Error: ${error.message}`,
                filePath: absolutePath
            };
        }
    }
};
