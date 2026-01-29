/**
 * File: src/utils/marketplace/SourceParser.ts
 * Role: Logic for parsing marketplace plugin source strings into structured targets.
 */

import { resolve } from 'node:path';
import { homedir } from 'node:os';
import { existsSync, statSync } from 'node:fs';

export type MarketplaceSource =
    | { source: 'git'; url: string; ref?: string }
    | { source: 'url'; url: string }
    | { source: 'file'; path: string }
    | { source: 'directory'; path: string }
    | { source: 'github'; repository: string; ref?: string }
    | { error: string };

/**
 * Parses a string input into a structured marketplace source.
 */
export function parseMarketplaceSource(input: string): MarketplaceSource | null {
    const trimmedInput = input.trim();
    if (!trimmedInput) return null;

    // 1. Git SSH/Protocol URLs
    const gitUrlMatch = trimmedInput.match(/^([a-zA-Z0-9._-]+@[^:]+:.+?(?:\.git)?)(#(.+))?$/);
    if (gitUrlMatch?.[1]) {
        return {
            source: 'git',
            url: gitUrlMatch[1],
            ref: gitUrlMatch[3]
        };
    }

    // 2. HTTP/HTTPS URLs
    if (trimmedInput.startsWith('http://') || trimmedInput.startsWith('https://')) {
        const urlMatch = trimmedInput.match(/^([^#]+)(#(.+))?$/);
        const url = urlMatch?.[1] || trimmedInput;
        const ref = urlMatch?.[3];

        if (url.endsWith('.git')) {
            return { source: 'git', url, ref };
        }

        try {
            const parsedUrl = new URL(url);
            if (parsedUrl.hostname === 'github.com' || parsedUrl.hostname === 'www.github.com') {
                const pathMatch = parsedUrl.pathname.match(/^\/([^/]+\/[^/]+?)(\/|\.git|$)/);
                if (pathMatch?.[1]) {
                    return {
                        source: 'git',
                        url: url.endsWith('.git') ? url : `${url}.git`,
                        ref
                    };
                }
            }
        } catch {
            return { source: 'url', url };
        }
        return { source: 'url', url };
    }

    // 3. Local File Paths
    if (trimmedInput.startsWith('./') || trimmedInput.startsWith('../') || trimmedInput.startsWith('/') || trimmedInput.startsWith('~')) {
        const resolvedPath = resolve(trimmedInput.startsWith('~') ? trimmedInput.replace(/^~/, homedir()) : trimmedInput);
        if (!existsSync(resolvedPath)) {
            return { error: `Path does not exist: ${resolvedPath}` };
        }

        const stat = statSync(resolvedPath);
        if (stat.isFile()) {
            if (resolvedPath.endsWith('.json')) {
                return { source: 'file', path: resolvedPath };
            }
            return { error: `File must be a JSON marketplace manifest: ${resolvedPath}` };
        }
        if (stat.isDirectory()) {
            return { source: 'directory', path: resolvedPath };
        }
        return { error: `Unsupported file type at path: ${resolvedPath}` };
    }

    // 4. GitHub Repository Paths (owner/repo)
    if (trimmedInput.includes('/') && !trimmedInput.startsWith('@')) {
        // Simple heuristic: must contain one slash and no colons
        if (trimmedInput.includes(':')) return null;

        const githubMatch = trimmedInput.match(/^([^#]+)(#(.+))?$/);
        const repo = githubMatch?.[1] || trimmedInput;
        const ref = githubMatch?.[3];

        return { source: 'github', repository: repo, ref };
    }

    return null;
}
