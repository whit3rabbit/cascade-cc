import { ConversationService } from '../services/conversation/ConversationService.js';
import { EnvService } from '../services/config/EnvService.js';

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
    results: (WebSearchHit | { toolUseId: string; content: WebSearchHit[] } | string)[];
    durationSeconds: number;
}

function getWebSearchPrompt(): string {
    const now = new Date().toISOString();
    const year = new Date().getFullYear();
    return `
- Allows Claude to search the web and use the results to inform responses
- Provides up-to-date information for current events and recent data
- Returns search result information formatted as search result blocks, including links as markdown hyperlinks
- Use this tool for accessing information beyond Claude's knowledge cutoff
- Searches are performed automatically within a single API call

CRITICAL REQUIREMENT - You MUST follow this:
  - After answering the user's question, you MUST include a "Sources:" section at the end of your response
  - In the Sources section, list all relevant URLs from the search results as markdown hyperlinks: [Title](URL)
  - This is MANDATORY - never skip including sources in your response
  - Example format:

    [Your answer here]

    Sources:
    - [Source Title 1](https://example.com/1)
    - [Source Title 2](https://example.com/2)

Usage notes:
  - Domain filtering is supported to include or block specific websites
  - Web search is only available in the US

IMPORTANT - Use the correct year in search queries:
  - Today's date is ${now}. You MUST use this year when searching for recent information, documentation, or current events.
  - Example: If the user asks for "latest React docs", search for "React documentation ${year}", NOT "React documentation ${year - 1}"
`;
}

export const WebSearchTool = {
    name: "WebSearch",
    description: "Performs a web search for the specified query.",
    isConcurrencySafe: true,
    async call(input: WebSearchInput, context: any) {
        const startTime = performance.now();
        const { query } = input;

        const subAgentPrompt = `Perform a web search for the query: ${query}`;
        const extraToolSchemas = [{
            type: "web_search_20250305",
            name: "web_search",
            allowed_domains: input.allowed_domains,
            blocked_domains: input.blocked_domains,
            max_uses: 8
        }];

        const loop = ConversationService.conversationLoop(
            [{ role: "user", content: subAgentPrompt }],
            "You are an assistant for performing a web search tool use",
            {
                ...context,
                extraToolSchemas,
                querySource: "web_search_tool"
            }
        );

        let results: WebSearchOutput['results'] = [];
        let queryCounter = 0;

        for await (const event of loop) {
            if (event.type === 'stream_event') {
                const chunk = event.event;
                // Handle progress updates if necessary (matching gold reference structure)
                if (chunk.type === 'content_block_delta' && chunk.delta?.type === 'input_json_delta') {
                    // Extracting query updates from partial JSON if we were to match exactly, 
                    // but for now we'll focus on results.
                }
            } else if (event.type === 'assistant') {
                const assistantMessage = event.message;
                for (const content of assistantMessage.content) {
                    if (content.type === 'text') {
                        results.push(content.text);
                    } else if (content.type === 'web_search_tool_result') {
                        results.push({
                            toolUseId: content.toolUseId,
                            content: content.content
                        });
                    }
                }
            }
        }

        const durationSeconds = (performance.now() - startTime) / 1000;

        return {
            is_error: false,
            content: {
                query,
                results,
                durationSeconds
            }
        };
    },

    prompt() {
        return getWebSearchPrompt();
    },

    mapToolResultToToolResultBlockParam(output: WebSearchOutput, toolUseId: string) {
        let content = `Web search results for query: "${output.query}"\n\n`;

        output.results.forEach(res => {
            if (typeof res === "string") {
                content += res + "\n\n";
            } else if ('toolUseId' in res) {
                content += `Links: ${JSON.stringify(res.content)}\n\n`;
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
