import { builtinModules } from "module";

const NODE_INTERNALS = [
    ...builtinModules,
    "bootstrap_node",
    "node"
].map(name => new RegExp(`(?:\\((?:node:)?${name}(?:\\.js)?:\\d+:\\d+\\)$|^\\s*at (?:node:)?${name}(?:\\.js)?:\\d+:\\d+$)`));

NODE_INTERNALS.push(
    /\((?:node:)?internal\/[^:]+:\d+:\d+\)$/,
    /\s*at (?:node:)?internal\/[^:]+:\d+:\d+$/,
    /\/\.node-spawn-wrap-\w+-\w+\/node:\d+:\d+\)?$/
);

/**
 * Sanitizes and formats stack traces.
 * Deobfuscated from wl1 in chunk_201.ts.
 */
export class StackCleaner {
    private cwd: string;
    private internals: RegExp[];

    constructor(options: { cwd?: string; internals?: RegExp[]; ignoredPackages?: string[] } = {}) {
        this.cwd = (options.cwd || process.cwd()).replace(/\\/g, "/");
        this.internals = [...(options.internals || NODE_INTERNALS)];

        if (options.ignoredPackages?.length) {
            const pkgPatterns = options.ignoredPackages.map(pkg => pkg.replace(/[|\\{}()[\]^$+*?.-]/g, "\\$&"));
            this.internals.push(new RegExp(`[/\\\\]node_modules[/\\\\](?:${pkgPatterns.join("|")})[/\\\\][^:]+:\\d+:\\d+`));
        }
    }

    clean(stack: string | string[], indent = 0): string {
        const spacing = " ".repeat(indent);
        const lines = Array.isArray(stack) ? stack : stack.split("\n");

        let isInternal = false;
        let lastLine: string | null = null;
        const result: string[] = [];

        lines.forEach(line => {
            line = line.replace(/\\/g, "/");
            if (this.internals.some(regex => regex.test(line))) return;

            const isStackLine = /^\s*at /.test(line);
            if (isInternal) {
                line = line.trimEnd().replace(/^(\s+)at /, "$1");
            } else {
                line = line.trim();
                if (isStackLine) line = line.slice(3);
            }

            line = line.replace(`${this.cwd}/`, "");

            if (line) {
                if (isStackLine) {
                    if (lastLine) {
                        result.push(lastLine);
                        lastLine = null;
                    }
                    result.push(line);
                } else {
                    isInternal = true;
                    lastLine = line;
                }
            }
        });

        return result.map(l => `${spacing}${l}\n`).join("");
    }
}
