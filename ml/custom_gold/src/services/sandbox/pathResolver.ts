import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

/**
 * Checks if a path contains glob characters.
 * Deobfuscated from IT in chunk_219.ts (used in chunk_220.ts).
 */
export function containsGlob(p: string): boolean {
    return p.includes("*") || p.includes("?") || p.includes("[") || p.includes("]");
}

/**
 * Resolves a path, handling ~, relative paths, and symlink resolution.
 * Deobfuscated from WT in chunk_220.ts.
 */
export function resolvePath(p: string): string {
    const cwd = process.cwd();
    let resolved = p;

    if (p === "~") {
        resolved = os.homedir();
    } else if (p.startsWith("~/")) {
        resolved = path.join(os.homedir(), p.slice(2));
    } else if (!path.isAbsolute(p)) {
        resolved = path.resolve(cwd, p);
    }

    // Handle globs by resolving the static prefix
    if (containsGlob(resolved)) {
        const staticPart = resolved.split(/[*?[\]]/)[0];
        if (staticPart && staticPart !== "/") {
            const dir = staticPart.endsWith("/") ? staticPart.slice(0, -1) : path.dirname(staticPart);
            try {
                const realDir = fs.realpathSync(dir);
                const globPart = resolved.slice(dir.length);
                return realDir + globPart;
            } catch {
                // Fallback to unresolved if realpath fails
            }
        }
        return resolved;
    }

    try {
        resolved = fs.realpathSync(resolved);
    } catch {
        // Fallback if file doesn't exist
    }

    return resolved;
}
