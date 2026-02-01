import { execa } from 'execa';
import { readdir, stat } from 'node:fs/promises';
import { join, relative } from 'node:path';
import { Fuse } from '../../vendor/Fuse.js';

export interface Suggestion {
    id: string;
    displayText: string;
    description?: string;
}

/**
 * Recursively lists all files in a directory, ignoring common hidden/temp folders.
 */
async function recursiveReaddir(dir: string, baseDir: string = dir): Promise<string[]> {
    const results: string[] = [];
    const IGNORED_DIRS = new Set(['.git', 'node_modules', 'dist', 'binaries', '.next', '.venv']);

    const entries = await readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
        const fullPath = join(dir, entry.name);
        const relPath = relative(baseDir, fullPath);

        if (entry.isDirectory()) {
            if (!IGNORED_DIRS.has(entry.name)) {
                results.push(...(await recursiveReaddir(fullPath, baseDir)));
            }
        } else {
            results.push(relPath);
        }
    }
    return results;
}

/**
 * Suggests slash commands based on input.
 */
export async function suggestSlashCommands(input: string, commands: any[]): Promise<any[]> {
    const { getSlashCommandSuggestions } = await import('../../utils/terminal/Suggestions.js');
    return getSlashCommandSuggestions(input, commands);
}

/**
 * Suggests files based on a query.
 */
export async function suggestFiles(query: string = ""): Promise<Suggestion[]> {
    let files: string[] = [];

    try {
        // 1. Try Git first
        const { stdout } = await execa('git', ['ls-files'], { cwd: process.cwd() });
        files = stdout.split('\n').filter(Boolean);
    } catch (error) {
        // 2. Fallback to recursive readdir
        try {
            files = await recursiveReaddir(process.cwd());
        } catch (fallbackError) {
            return [];
        }
    }

    if (!query) {
        return files.slice(0, 10).map(f => ({ id: f, displayText: `@${f}` }));
    }

    const fileData = files.map(f => ({ path: f }));
    const fuse = new Fuse(fileData, {
        threshold: 0.4,
        keys: ['path']
    });

    const results = fuse.search(query);
    return results.slice(0, 10).map(r => ({
        id: (r.item as any).path,
        displayText: `@${(r.item as any).path}`,
        description: `File in project`
    }));
}

/**
 * Suggests directories based on a prefix.
 */
export async function suggestDirectories(prefix: string = ""): Promise<Suggestion[]> {
    let dirs: string[] = [];
    try {
        // Try find command first (Unix-like systems)
        const { stdout } = await execa('find', ['.', '-maxdepth', '2', '-type', 'd', '-not', '-path', '*/.*']);
        dirs = stdout.split('\n')
            .filter(Boolean)
            .map(d => d.replace(/^\.\//, ''))
            .filter(d => d && d !== '.');
    } catch (error) {
        // Fallback to shallow readdir
        try {
            const entries = await readdir(process.cwd(), { withFileTypes: true });
            dirs = entries.filter(e => e.isDirectory() && !e.name.startsWith('.')).map(e => e.name);
        } catch (e) {
            return [];
        }
    }

    if (!prefix) return dirs.slice(0, 10).map(d => ({ id: d, displayText: d }));

    const filtered = dirs
        .filter(d => d.toLowerCase().startsWith(prefix.toLowerCase()))
        .slice(0, 10);

    return filtered.map(d => ({
        id: d,
        displayText: d,
        description: `Directory`
    }));
}
