/**
 * File: src/services/terminal/FileIndexService.ts
 * Role: Provides indexing and suggestion logic for files and directories in the project.
 */

import { execa } from 'execa';

export interface Suggestion {
    id: string;
    displayText: string;
    description?: string;
}

/**
 * Suggests slash commands based on input.
 * (Delegated to util but defined here for hook consistency).
 */
export async function suggestSlashCommands(input: string, commands: any[]): Promise<any[]> {
    // Dynamic import to avoid circular dep issues or side effects during test setup
    const { getSlashCommandSuggestions } = await import('../../utils/terminal/Suggestions.js');
    return getSlashCommandSuggestions(input, commands);
}

/**
 * Suggests files based on a query.
 */
export async function suggestFiles(query: string = ""): Promise<Suggestion[]> {
    try {
        // Use git ls-files as a fast way to get tracked files
        const { stdout } = await execa('git', ['ls-files']);
        const files = stdout.split('\n').filter(Boolean);

        if (!query) return files.slice(0, 10).map(f => ({ id: f, displayText: `@${f}` }));

        const filtered = files
            .filter(f => f.toLowerCase().includes(query.toLowerCase()))
            .slice(0, 10);

        return filtered.map(f => ({
            id: f,
            displayText: `@${f}`,
            description: `File in project`
        }));
    } catch (error) {
        // Fallback or handle non-git repos
        return [];
    }
}

/**
 * Suggests directories based on a prefix.
 */
export async function suggestDirectories(prefix: string = ""): Promise<Suggestion[]> {
    try {
        // Simplified directory suggestion
        const { stdout } = await execa('find', ['.', '-maxdepth', '2', '-type', 'd', '-not', '-path', '*/.*']);
        const dirs = stdout.split('\n')
            .filter(Boolean)
            .map(d => d.replace(/^\.\//, ''))
            .filter(d => d && d !== '.');

        if (!prefix) return dirs.slice(0, 10).map(d => ({ id: d, displayText: d }));

        const filtered = dirs
            .filter(d => d.toLowerCase().startsWith(prefix.toLowerCase()))
            .slice(0, 10);

        return filtered.map(d => ({
            id: d,
            displayText: d,
            description: `Directory`
        }));
    } catch (error) {
        return [];
    }
}
