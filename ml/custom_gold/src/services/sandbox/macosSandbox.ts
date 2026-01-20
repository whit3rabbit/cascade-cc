import * as path from "node:path";
import { resolvePath } from "./pathResolver.js";
import { scanSensitiveFiles } from "./sensitiveFiles.js";

/**
 * Translates a glob pattern to a sandbox-exec compatible regex.
 * Deobfuscated from bA1 in chunk_220.ts.
 */
function translateGlobToRegex(pattern: string): string {
    return "^" + pattern
        .replace(/[.^$+{}()|\\]/g, "\\$&")
        .replace(/\[([^\]]*?)$/g, "\\[$1")
        .replace(/\*\*\//g, "__GLOBSTAR_SLASH__")
        .replace(/\*\*/g, "__GLOBSTAR__")
        .replace(/\*/g, "[^/]*")
        .replace(/\?/g, "[^/]")
        .replace(/__GLOBSTAR_SLASH__/g, "(.*/)?")
        .replace(/__GLOBSTAR__/g, ".*") + "$";
}

/**
 * Gets all parent directories of a path.
 * Deobfuscated from FEB in chunk_220.ts.
 */
function getParentPaths(p: string): string[] {
    const parents: string[] = [];
    let current = path.dirname(p);
    while (current !== "/" && current !== ".") {
        parents.push(current);
        const next = path.dirname(current);
        if (next === current) break;
        current = next;
    }
    return parents;
}

/**
 * Generates file-write-unlink denial rules for paths and their parents.
 * Deobfuscated from CEB in chunk_220.ts.
 */
function getUnlinkDenyProfile(paths: string[], logTag: string): string[] {
    const rules: string[] = [];
    for (const p of paths) {
        const resolved = resolvePath(p);
        if (resolved === "/" || resolved === ".") continue;

        if (resolved.includes("*") || resolved.includes("?") || resolved.includes("[") || resolved.includes("]")) {
            const regex = translateGlobToRegex(resolved);
            rules.push(`(deny file-write-unlink (regex ${JSON.stringify(regex)}) (with message ${JSON.stringify(logTag)}))`);

            // For globs, we also deny unlinking the static prefix parents
            const staticPart = resolved.split(/[*?[\]]/)[0];
            if (staticPart && staticPart !== "/") {
                const baseDir = staticPart.endsWith("/") ? staticPart.slice(0, -1) : path.dirname(staticPart);
                if (baseDir !== "/" && baseDir !== ".") {
                    rules.push(`(deny file-write-unlink (literal ${JSON.stringify(baseDir)}) (with message ${JSON.stringify(logTag)}))`);
                    for (const parent of getParentPaths(baseDir)) {
                        rules.push(`(deny file-write-unlink (literal ${JSON.stringify(parent)}) (with message ${JSON.stringify(logTag)}))`);
                    }
                }
            }
        } else {
            rules.push(`(deny file-write-unlink (subpath ${JSON.stringify(resolved)}) (with message ${JSON.stringify(logTag)}))`);
            for (const parent of getParentPaths(resolved)) {
                rules.push(`(deny file-write-unlink (literal ${JSON.stringify(parent)}) (with message ${JSON.stringify(logTag)}))`);
            }
        }
    }
    return rules;
}

/**
 * Generates a macOS sandbox profile for read restrictions.
 * Deobfuscated from C93 in chunk_220.ts.
 */
export function getMacReadProfile(denyOnly: string[], logTag: string): string {
    if (!denyOnly || denyOnly.length === 0) return "(allow file-read*)";

    const rules = ["(allow file-read*)"];

    for (const p of denyOnly) {
        const resolved = resolvePath(p);
        if (resolved.includes("*") || resolved.includes("?") || resolved.includes("[") || resolved.includes("]")) {
            const regex = translateGlobToRegex(resolved);
            rules.push(`(deny file-read* (regex ${JSON.stringify(regex)}) (with message ${JSON.stringify(logTag)}))`);
        } else {
            rules.push(`(deny file-read* (subpath ${JSON.stringify(resolved)}) (with message ${JSON.stringify(logTag)}))`);
        }
    }

    // Also deny unlinking these paths
    rules.push(...getUnlinkDenyProfile(denyOnly, logTag));

    return rules.join("\n");
}

/**
 * Generates a macOS sandbox profile for write restrictions.
 * Deobfuscated from $93 in chunk_220.ts.
 */
export async function getMacWriteProfile(
    allowOnly: string[],
    denyWithinAllow: string[],
    logTag: string,
    allowGitConfig: boolean = false
): Promise<string> {
    const rules: string[] = [];

    // Allow common temp dirs
    const tmp = process.env.TMPDIR;
    if (tmp) {
        let baseTmp = tmp.replace(/\/T\/?$/, "");
        const tmpPaths = [baseTmp];
        if (baseTmp.startsWith("/private/var/")) tmpPaths.push(baseTmp.replace("/private", ""));
        else if (baseTmp.startsWith("/var/")) tmpPaths.push("/private" + baseTmp);

        for (const t of tmpPaths) {
            rules.push(`(allow file-write* (subpath ${JSON.stringify(resolvePath(t))}) (with message ${JSON.stringify(logTag)}))`);
        }
    }

    // Explicit allows
    for (const p of allowOnly || []) {
        const resolved = resolvePath(p);
        if (resolved.includes("*") || resolved.includes("?") || resolved.includes("[") || resolved.includes("]")) {
            rules.push(`(allow file-write* (regex ${JSON.stringify(translateGlobToRegex(resolved))}) (with message ${JSON.stringify(logTag)}))`);
        } else {
            rules.push(`(allow file-write* (subpath ${JSON.stringify(resolved)}) (with message ${JSON.stringify(logTag)}))`);
        }
    }

    // Deny sensitive files even within allowed paths
    // Deobfuscated from E93 in chunk_224.ts
    const sensitive = await scanSensitiveFiles(process.cwd(), 3, allowGitConfig);
    const allDeny = Array.from(new Set([...(denyWithinAllow || []), ...sensitive]));

    for (const p of allDeny) {
        const resolved = resolvePath(p);
        if (resolved.includes("*") || resolved.includes("?") || resolved.includes("[") || resolved.includes("]")) {
            rules.push(`(deny file-write* (regex ${JSON.stringify(translateGlobToRegex(resolved))}) (with message ${JSON.stringify(logTag)}))`);
        } else {
            rules.push(`(deny file-write* (subpath ${JSON.stringify(resolved)}) (with message ${JSON.stringify(logTag)}))`);
        }
    }

    // Also deny unlinking these paths
    rules.push(...getUnlinkDenyProfile(allDeny, logTag));

    return rules.length > 0 ? rules.join("\n") : "(allow file-write*)";
}
