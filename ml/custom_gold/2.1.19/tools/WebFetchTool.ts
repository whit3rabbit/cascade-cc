/**
 * File: src/tools/WebFetchTool.ts
 * Role: Fetches content from a URL and converts it to markdown.
 */

import axios from 'axios';
import TurndownService from 'turndown';
import { isUrlAllowed } from '../services/sandbox/SandboxSettings.js';
import { terminalLog } from '../utils/shared/runtime.js';
import { EnvService } from '../services/config/EnvService.js';

export interface WebFetchInput {
    url: string;
    prompt?: string;
}

export const WebFetchTool = {
    name: "WebFetch",
    description: "Fetches content from a specified URL and converts HTML to markdown.",
    async call(input: WebFetchInput, _context: any) {
        let { url, prompt: _prompt } = input;

        // basic URL validation and upgrade
        if (!url.startsWith('http')) {
            url = 'https://' + url;
        }

        if (!isUrlAllowed(url)) {
            return {
                is_error: true,
                content: `URL blocked by sandbox policy: ${url}`
            };
        }

        try {
            terminalLog(`Fetching ${url}...`);
            const response = await axios.get(url, {
                timeout: 5000,
                headers: {
                    'User-Agent': 'Mozilla/5.0 (compatible; ClaudeCode/2.1)'
                }
            });

            const html = response.data;
            if (typeof html !== 'string') {
                return {
                    is_error: true,
                    content: "Failed to retrieve string content from the URL."
                };
            }

            const turndownService = new TurndownService();
            const markdown = turndownService.turndown(html);

            // Limited truncation based on config
            const maxTokens = Number(EnvService.get('CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS')) || 32000;
            const maxChars = maxTokens * 4;
            let content = markdown;
            if (content.length > maxChars) {
                content = content.slice(0, maxChars) + "\n\n... [Content truncated due to length] ...";
            }

            return {
                is_error: false,
                content: content
            };
        } catch (error: any) {
            return {
                is_error: true,
                content: `Failed to fetch URL: ${error.message}`
            };
        }
    }
};
