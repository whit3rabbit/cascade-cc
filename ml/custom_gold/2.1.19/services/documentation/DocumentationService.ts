/**
 * File: src/services/documentation/DocumentationService.ts
 * Role: Fetches and manages documentation content.
 */

import fs from 'fs';
import path from 'path';
import { fetch } from 'undici';

export class DocumentationService {
    private static DOCS_URL = 'https://code.claude.com/docs/llms.txt';
    private static CACHE_FILE = '.claude-docs-cache.txt';

    static async fetchDocs(): Promise<string> {
        try {
            // Check cache first
            // In a real app we'd check age, but for now just existence
            // We use a temp dir for cache
            const cachePath = path.join(process.cwd(), 'node_modules', '.cache', this.CACHE_FILE);

            if (fs.existsSync(cachePath)) {
                return fs.readFileSync(cachePath, 'utf-8');
            }

            // Fetch from network
            const response = await fetch(this.DOCS_URL);
            if (!response.ok) {
                throw new Error(`Failed to fetch docs: ${response.statusText}`);
            }
            const text = await response.text();

            // Ensure cache dir exists
            const cacheDir = path.dirname(cachePath);
            if (!fs.existsSync(cacheDir)) {
                fs.mkdirSync(cacheDir, { recursive: true });
            }

            fs.writeFileSync(cachePath, text);
            return text;
        } catch (error) {
            console.error('Error fetching docs:', error);
            return 'Documentation currently unavailable. Please check your internet connection.';
        }
    }

    static async getSection(query: string): Promise<string> {
        const fullDocs = await this.fetchDocs();
        if (!query) return fullDocs;

        // Simple section finder (fuzzy)
        const lines = fullDocs.split('\n');
        const matches = lines.filter(l => l.toLowerCase().includes(query.toLowerCase()));

        if (matches.length === 0) {
            return `No documentation found for "${query}".`;
        }

        // Return a snippet around the match
        // This is a naive implementation; a real one would parse markdown sections
        return `Found ${matches.length} matches for "${query}":\n\n` + matches.slice(0, 10).join('\n') + (matches.length > 10 ? '\n...' : '');
    }
}
