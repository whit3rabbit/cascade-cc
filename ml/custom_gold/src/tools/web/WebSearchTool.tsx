
// Logic from chunk_529.ts (Web Search & LSP Utilities)

import React from 'react';
import { Box, Text } from 'ink';

import { z } from "zod";

// --- Web Search Tool (hV1) ---
export const WebSearchTool = {
    name: "web_search",
    description: "Searches the web for information.",
    inputSchema: z.object({
        query: z.string().describe("Search query"),
        allowed_domains: z.array(z.string()).optional().describe("Optional whitelist"),
        blocked_domains: z.array(z.string()).optional().describe("Optional blacklist")
    }),
    async call(input: any) {
        console.log(`Searching web for: ${input.query}`);
        // Pipeline: Query -> Fetch Results -> Format
        return { data: { query: input.query, results: [], durationSeconds: 1.5 } };
    },
    renderResultMessage(result: any) {
        const count = result.data?.results?.length ?? 0;
        const duration = result.data?.durationSeconds ?? 0;
        return (
            <Box justifyContent="space-between" width="100%">
                <Text>Did {count} searches in {duration}s</Text>
            </Box>
        );
    }
};

// --- LSP Result Formatter (yq0) ---
export function formatLspDefinitionResult(result: any, projectPath: string) {
    if (!result) return "No definition found.";

    const formatLocation = (loc: any) => {
        const path = loc.uri.replace(/^file:\/\//, "");
        return `${path}:${loc.range.start.line + 1}:${loc.range.start.character + 1}`;
    };

    if (Array.isArray(result)) {
        return `Found ${result.length} definitions:\n${result.map(l => "  " + formatLocation(l)).join("\n")}`;
    }
    return `Defined in ${formatLocation(result)}`;
}
