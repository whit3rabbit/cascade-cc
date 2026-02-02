/**
 * File: src/tools/WebSearchTool.ts
 * Role: Performs a web search.
 */

import { terminalLog } from '../utils/shared/runtime.js';

export interface WebSearchInput {
    query: string;
    allowed_domains?: string[];
    blocked_domains?: string[];
}

export interface WebSearchHit {
    title: string;
    url: string;
}

export interface WebSearchOutput {
    query: string;
    results: (WebSearchHit | string)[];
    durationSeconds: number;
}

export const WebSearchTool = {
    name: "WebSearch",
    description: "Performs a web search for the specified query.",
    isConcurrencySafe: true,
    async call(input: WebSearchInput, context: any) {
        const startTime = performance.now();
        const { query, allowed_domains, blocked_domains } = input;
        const { onProgress } = context || {};

        if (onProgress) {
            onProgress({ type: 'query_update', query });
        } else {
            terminalLog(`Searching for: ${query}...`);
        }

        // Simulate network delay
        await new Promise(r => setTimeout(r, 1200));

        const hits: WebSearchHit[] = [
            { title: `${query} - Wikipedia`, url: `https://en.wikipedia.org/wiki/${encodeURIComponent(query)}` },
            { title: `Top 10 facts about ${query}`, url: `https://example.com/facts/${encodeURIComponent(query)}` },
            { title: `${query} Official Site`, url: `https://${query.toLowerCase().replace(/\s+/g, '')}.com` },
            { title: `Latest news on ${query}`, url: `https://news.example.com/search?q=${encodeURIComponent(query)}` },
            { title: `Community discussion on ${query}`, url: `https://reddit.com/r/${query.toLowerCase().replace(/\s+/g, '')}` }
        ];

        // Filter by domains if provided
        let filteredHits = hits;
        if (allowed_domains && allowed_domains.length > 0) {
            filteredHits = hits.filter(h => allowed_domains.some(d => h.url.includes(d)));
        }
        if (blocked_domains && blocked_domains.length > 0) {
            filteredHits = hits.filter(h => !blocked_domains.some(d => h.url.includes(d)));
        }

        const durationSeconds = (performance.now() - startTime) / 1000;

        if (onProgress) {
            onProgress({ type: 'search_results_received', resultCount: filteredHits.length, query });
        }

        return {
            is_error: false,
            content: {
                query,
                results: [
                    {
                        toolUseId: Math.random().toString(36).substring(2, 10),
                        content: filteredHits
                    }
                ],
                durationSeconds
            }
        };
    },

    mapToolResultToToolResultBlockParam(output: WebSearchOutput, toolUseId: string) {
        let content = `Web search results for query: "${output.query}"\n\n`;

        output.results.forEach(res => {
            if (typeof res === "string") {
                content += res + "\n\n";
            } else {
                if ('content' in res && Array.isArray(res.content)) {
                    content += `Links: ${JSON.stringify(res.content)}\n\n`;
                } else {
                    content += `No links found.\n\n`;
                }
            }
        });

        content += `\nREMINDER: You MUST include the sources above in your response to the user using markdown hyperlinks.`;

        return {
            toolUseId,
            type: "tool_result",
            content: content.trim()
        };
    }
};
