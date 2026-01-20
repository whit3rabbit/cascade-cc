import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { MarketplaceSource } from './MarketplaceSchemas.js';

export type MarketplaceSourceResult = MarketplaceSource | { error: string };

/**
 * Parses a marketplace source string into a structured MarketplaceSource object.
 * Based on nH1 in chunk_566.ts.
 */
export function parseMarketplaceSource(input: string): MarketplaceSourceResult | null {
    const trimmed = input.trim();
    if (!trimmed) return null;

    // 1. Git SSH URL: user@host:repo.git#ref
    const gitSshMatch = trimmed.match(/^([a-zA-Z0-9._-]+@[^:]+:.+?(?:\.git)?)(#(.+))?$/);
    if (gitSshMatch?.[1]) {
        return {
            source: 'git',
            url: gitSshMatch[1],
            ref: gitSshMatch[3]
        };
    }

    // 2. HTTP/HTTPS URL
    if (trimmed.startsWith('http://') || trimmed.startsWith('https://')) {
        const urlPartMatch = trimmed.match(/^([^#]+)(#(.+))?$/);
        const urlStr = urlPartMatch?.[1] || trimmed;
        const ref = urlPartMatch?.[3];

        if (urlStr.endsWith('.git')) {
            return { source: 'git', url: urlStr, ref };
        }

        try {
            const url = new URL(urlStr);
            if (url.hostname === 'github.com' || url.hostname === 'www.github.com') {
                const parts = url.pathname.split('/').filter(Boolean);
                if (parts.length >= 2) {
                    // It's a github repo URL
                    const repoUrl = `https://github.com/${parts[0]}/${parts[1]}.git`;
                    return { source: 'git', url: repoUrl, ref };
                }
            }
        } catch (e) {
            // Fallback to generic URL
        }

        return { source: 'url', url: urlStr };
    }

    // 3. Local paths
    if (trimmed.startsWith('./') || trimmed.startsWith('../') || trimmed.startsWith('/') || trimmed.startsWith('~')) {
        let resolvedPath = trimmed;
        if (trimmed.startsWith('~')) {
            resolvedPath = path.join(os.homedir(), trimmed.slice(1));
        }
        resolvedPath = path.resolve(resolvedPath);

        if (!fs.existsSync(resolvedPath)) {
            return { error: `Path does not exist: ${resolvedPath}`, source: 'file' }; // source is placeholder here
        }

        const stats = fs.statSync(resolvedPath);
        if (stats.isFile()) {
            if (resolvedPath.endsWith('.json')) {
                return { source: 'file', path: resolvedPath };
            } else {
                return { error: `File path must point to a .json file, but got: ${resolvedPath}`, source: 'file' };
            }
        } else if (stats.isDirectory()) {
            return { source: 'directory', path: resolvedPath };
        } else {
            return { error: `Path is neither a file nor a directory: ${resolvedPath}`, source: 'file' };
        }
    }

    // 4. GitHub shorthand: owner/repo#ref
    if (trimmed.includes('/') && !trimmed.startsWith('@')) {
        if (trimmed.includes(':')) return null; // Likely not shorthand if it has a colon
        const parts = trimmed.match(/^([^#]+)(#(.+))?$/);
        const repo = parts?.[1] || trimmed;
        const ref = parts?.[3];
        return { source: 'github', repo, ref };
    }

    return null;
}
