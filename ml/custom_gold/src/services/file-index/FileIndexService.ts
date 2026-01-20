
import { execSync } from 'child_process';
import { join, relative, sep } from 'path';
import { readdirSync } from 'fs';
import { getCwd } from '../terminal/terminalUtils.js';
import { Fuse, SearchResult } from '../../vendor/Fuse.js';
import { getParentDirectories } from './FileIndexUtils.js';

/**
 * Service for indexing and searching project files.
 */

export interface FileSuggestion {
    id: string;
    displayText: string;
    metadata?: {
        score?: number;
    };
}

let fileListCache: string[] = [];
let lastRefreshTime: number = 0;
const REFRESH_THRESHOLD = 60000; // 1 minute

export function isGitRepo(cwd: string): boolean {
    try {
        execSync('git rev-parse --is-inside-work-tree', { cwd, stdio: 'ignore' });
        return true;
    } catch {
        return false;
    }
}

export function getGitToplevel(cwd: string): string | null {
    try {
        return execSync('git rev-parse --show-toplevel', { cwd, encoding: 'utf8' }).trim();
    } catch {
        return null;
    }
}

async function getFilesUsingGit(cwd: string, respectGitignore: boolean): Promise<string[] | null> {
    if (!isGitRepo(cwd)) return null;
    try {
        const cmd = respectGitignore ? 'git ls-files --cached --others --exclude-standard' : 'git ls-files --cached --others';
        const output = execSync(cmd, { cwd, encoding: 'utf8' });
        return output.split('\n').filter(Boolean);
    } catch {
        return null;
    }
}

async function getFilesUsingRipgrep(cwd: string, respectGitignore: boolean): Promise<string[]> {
    try {
        const ignoreArg = respectGitignore ? '' : '--no-ignore-vcs';
        const output = execSync(`rg --files --follow --hidden --glob "!.git/" ${ignoreArg}`, { cwd, encoding: 'utf8' });
        return output.split('\n').filter(Boolean).map(f => relative(cwd, join(cwd, f)));
    } catch {
        return [];
    }
}

async function getMcpFiles(): Promise<string[]> {
    // Placeholder for MCP server file indexing
    return [];
}

export async function refreshFileIndex(): Promise<string[]> {
    const cwd = getCwd();
    const respectGitignore = true; // Default behavior

    const [projectFiles, mcpFiles] = await Promise.all([
        (async () => {
            let files = await getFilesUsingGit(cwd, respectGitignore);
            if (files === null) {
                files = await getFilesUsingRipgrep(cwd, respectGitignore);
            }
            return files;
        })(),
        getMcpFiles()
    ]);


    const allFiles = [...projectFiles, ...mcpFiles];
    const allDirs = getParentDirectories(allFiles);
    fileListCache = [...allFiles, ...allDirs];
    lastRefreshTime = Date.now();
    return fileListCache;
}

export async function suggestFiles(query: string): Promise<FileSuggestion[]> {
    if (Date.now() - lastRefreshTime > REFRESH_THRESHOLD || fileListCache.length === 0) {
        await refreshFileIndex();
    }

    if (!query || query === '.' || query === './') {
        const cwd = getCwd();
        return readdirSync(cwd, { withFileTypes: true })
            .map(dirent => {
                const name = dirent.name + (dirent.isDirectory() ? sep : '');
                return { id: `file-${name}`, displayText: name };
            })
            .slice(0, 15);
    }

    const fuse = new Fuse(fileListCache.map(f => ({ path: f })), {
        keys: ['path'],
        threshold: 0.5
    });

    const results = fuse.search(query);
    return results.slice(0, 15).map((r: SearchResult<any>) => ({
        id: `file-${r.item.path}`,
        displayText: r.item.path,
        metadata: { score: r.score }
    }));
}

export async function suggestDirectories(query: string): Promise<FileSuggestion[]> {
    if (Date.now() - lastRefreshTime > REFRESH_THRESHOLD || fileListCache.length === 0) {
        await refreshFileIndex();
    }

    const allDirs = getParentDirectories(fileListCache);

    if (!query || query === '.' || query === './') {
        return allDirs.slice(0, 15).map(dir => ({ id: `dir-${dir}`, displayText: dir }));
    }

    const fuse = new Fuse(allDirs.map(d => ({ path: d })), {
        keys: ['path'],
        threshold: 0.5
    });

    const results = fuse.search(query);
    return results.slice(0, 15).map((r: SearchResult<any>) => ({
        id: `dir-${r.item.path}`,
        displayText: r.item.path,
        metadata: { score: r.score }
    }));
}
