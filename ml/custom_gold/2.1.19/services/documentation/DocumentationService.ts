/**
 * File: src/services/documentation/DocumentationService.ts
 * Role: Fetches and manages documentation content.
 */

import fs from 'fs';
import path from 'path';
import { fetch } from 'undici';
import { EnvService } from '../config/EnvService.js';

export class DocumentationService {
    private static DOCS_URL = 'https://code.claude.com/docs/llms.txt';
    private static DOCS_MAP_URL = 'https://code.claude.com/docs/en/claude_code_docs_map.md';
    private static CACHE_FILE = '.claude-docs-cache.txt';
    private static MAP_CACHE_FILE = '.claude-docs-map-cache.md';

    static async fetchDocs(url: string = this.DOCS_URL, cacheFile: string = this.CACHE_FILE): Promise<string> {
        try {
            const cachePath = path.join(process.cwd(), 'node_modules', '.cache', cacheFile);

            if (fs.existsSync(cachePath)) {
                const stats = fs.statSync(cachePath);
                const isOld = Date.now() - stats.mtimeMs > 24 * 60 * 60 * 1000; // 24h cache
                if (!isOld) return fs.readFileSync(cachePath, 'utf-8');
            }

            const response = await fetch(url);
            if (!response.ok) {
                if (fs.existsSync(cachePath)) return fs.readFileSync(cachePath, 'utf-8');
                throw new Error(`Failed to fetch docs: ${response.statusText}`);
            }
            const text = await response.text();

            const cacheDir = path.dirname(cachePath);
            if (!fs.existsSync(cacheDir)) {
                fs.mkdirSync(cacheDir, { recursive: true });
            }

            fs.writeFileSync(cachePath, text);
            return text;
        } catch (error) {
            if (EnvService.isTruthy("DEBUG_DOCS")) {
                console.error('Error fetching docs:', error);
            }
            return 'Documentation currently unavailable.';
        }
    }

    static async getSection(query: string): Promise<string> {
        const fullDocs = await this.fetchDocs();
        const docsMap = await this.fetchDocs(this.DOCS_MAP_URL, this.MAP_CACHE_FILE);

        if (!query) return fullDocs;

        // Better section finder logic
        // 1. Try to find headers in the docs map first
        const mapLines = docsMap.split('\n');
        const mapMatches = mapLines.filter(l => l.toLowerCase().includes(query.toLowerCase()) && l.startsWith('#'));

        // 2. Fallback to basic search in llms.txt
        const lines = fullDocs.split('\n');
        const matches: { index: number, line: string }[] = [];
        lines.forEach((line, index) => {
            if (line.toLowerCase().includes(query.toLowerCase())) {
                matches.push({ index, line });
            }
        });

        if (matches.length === 0 && mapMatches.length === 0) {
            return `No documentation found for "${query}".`;
        }

        // Return a snippet with context
        const result: string[] = [];
        if (mapMatches.length > 0) {
            result.push(`**Related sections from docs map:**\n${mapMatches.slice(0, 3).join('\n')}\n`);
        }

        if (matches.length > 0) {
            result.push(`**Found in core documentation:**`);
            const firstMatch = matches[0];
            const start = Math.max(0, firstMatch.index - 2);
            const end = Math.min(lines.length, firstMatch.index + 8);
            result.push('```md');
            result.push(lines.slice(start, end).join('\n'));
            result.push('```');
        }

        return result.join('\n');
    }
}
