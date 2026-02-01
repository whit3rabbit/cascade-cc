
import { readdir, stat } from 'fs/promises';
import { join, resolve, relative } from 'path';
import micromatch from 'micromatch';

export interface GlobInput {
    pattern: string;
    path?: string;
}

export const GlobTool = {
    name: "Glob",
    description: "Find files matching a glob pattern.",
    async call(input: GlobInput) {
        const { pattern, path: basePath } = input;
        const cwd = basePath ? resolve(basePath) : process.cwd();

        // Safety check: ensure cwd is valid
        try {
            await stat(cwd);
        } catch {
            return {
                is_error: true,
                content: `Directory not found: ${cwd}`
            };
        }

        const matches: string[] = [];

        // Simple recursive walker
        // For production, we might want to respect .gitignore or use a proper glob library if added.
        // Since we only have micromatch, we do the walking ourselves.
        async function walk(dir: string) {
            const entries = await readdir(dir, { withFileTypes: true });
            for (const entry of entries) {
                const fullPath = join(dir, entry.name);
                const relPath = relative(cwd, fullPath);

                // Add to matches if it matches
                // Note: micromatch usually matches against relative paths
                if (micromatch(relPath, pattern, { dot: true })) {
                    matches.push(relPath);
                }

                if (entry.isDirectory()) {
                    // Optimization: don't traverse if the dir itself excludes potential matches?
                    // For simplicity, we just traverse everything unless it's obviously ignored (node_modules, .git)
                    if (entry.name === 'node_modules' || entry.name === '.git') continue;
                    await walk(fullPath);
                }
            }
        }

        try {
            await walk(cwd);
            // Limit results to avoid overflow
            const limited = matches.slice(0, 500); // arbitrary limit
            const overflow = matches.length > 500;

            return {
                is_error: false,
                content: limited.join('\n') + (overflow ? `\n... (${matches.length - 500} more matches)` : '')
            };
        } catch (error: any) {
            return {
                is_error: true,
                content: `Glob failed: ${error.message}`
            };
        }
    }
};
