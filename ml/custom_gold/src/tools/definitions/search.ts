
import { z } from "zod";
import fs from "fs";
import path from "path";
import { Tool } from "./tool.js";
import { log, logError } from "../../services/logger/loggerService.js";
import { runRipgrep } from "../../utils/shared/ripgrep.js";
import { resolvePath } from "../../utils/shared/pathUtils.js";
import { formatDuration } from "../../utils/shared/formatUtils.js";

const GrepSearchInputSchema = z.object({
    pattern: z.string().describe("The regular expression pattern to search for in file contents"),
    path: z.string().optional().describe("File or directory to search in (rg PATH). Defaults to current working directory."),
    glob: z.string().optional().describe('Glob pattern to filter files (e.g. "*.js", "*.{ts,tsx}") - maps to rg --glob'),
    output_mode: z.enum(["content", "files_with_matches", "count"]).optional().describe('Output mode: "content" shows matching lines (supports -A/-B/-C context, -n line numbers, head_limit), "files_with_matches" shows file paths (supports head_limit), "count" shows match counts (supports head_limit). Defaults to "files_with_matches".'),
    "-B": z.number().optional().describe('Number of lines to show before each match (rg -B). Requires output_mode: "content", ignored otherwise.'),
    "-A": z.number().optional().describe('Number of lines to show after each match (rg -A). Requires output_mode: "content", ignored otherwise.'),
    "-C": z.number().optional().describe('Number of lines to show before and after each match (rg -C). Requires output_mode: "content", ignored otherwise.'),
    "-n": z.boolean().optional().describe('Show line numbers in output (rg -n). Requires output_mode: "content", ignored otherwise. Defaults to true.'),
    "-i": z.boolean().optional().describe("Case insensitive search (rg -i)"),
    type: z.string().optional().describe("File type to search (rg --type). Common types: js, py, rust, go, java, etc. More efficient than include for standard file types."),
    head_limit: z.number().optional().describe('Limit output to first N lines/entries, equivalent to "| head -N". Works across all output modes: content (limits output lines), files_with_matches (limits file paths), count (limits count entries). Defaults to 0 (unlimited).'),
    offset: z.number().optional().describe('Skip first N lines/entries before applying head_limit, equivalent to "| tail -n +N | head -N". Works across all output modes. Defaults to 0.'),
    multiline: z.boolean().optional().describe("Enable multiline mode where . matches newlines and patterns can span lines (rg -U --multiline-dotall). Default: false.")
});

const GrepSearchOutputSchema = z.object({
    mode: z.enum(["content", "files_with_matches", "count"]).optional(),
    numFiles: z.number(),
    filenames: z.array(z.string()),
    content: z.string().optional(),
    numLines: z.number().optional(),
    numMatches: z.number().optional(),
    appliedLimit: z.number().optional(),
    appliedOffset: z.number().optional()
});

const EXCLUDES = [".git", ".svn", ".hg", ".bzr"];

export const GrepSearchTool: Tool = {
    name: "GrepSearchTool",
    strict: true,
    input_examples: [
        { pattern: "TODO", output_mode: "files_with_matches" },
        { pattern: "function.*export", glob: "*.ts", output_mode: "content", "-n": true },
        { pattern: "error", "-i": true, type: "js" }
    ],
    description: async () => "Search for patterns in files using ripgrep.",
    userFacingName: () => "Search",
    getToolUseSummary: (input: any) => `Searching for "${input.pattern}"`,
    prompt: async () => "Search for a pattern in file contents across the codebase. Works like ripgrep.",
    isEnabled: () => true,
    inputSchema: GrepSearchInputSchema,
    outputSchema: GrepSearchOutputSchema,
    isConcurrencySafe: () => true,
    isReadOnly: () => true,
    isSearchOrReadCommand: () => ({ isSearch: true, isRead: false }),
    getPath: (input: any) => input.path ? resolvePath(input.path) : process.cwd(),

    async validateInput(input: any) {
        if (input.path) {
            const absPath = resolvePath(input.path);
            if (!fs.existsSync(absPath)) {
                return { result: false, message: `Path does not exist: ${input.path}`, errorCode: 1 };
            }
        }
        return { result: true };
    },

    async checkPermissions(input: any, context: any) {
        return true;
    },

    renderToolUseMessage: (input: any, { verbose }: any) => {
        if (!input.pattern) return null;
        let parts = [`pattern: "${input.pattern}"`];
        if (input.path) parts.push(`path: "${input.path}"`);
        if (input.glob) parts.push(`glob: "${input.glob}"`);
        if (input.type) parts.push(`type: "${input.type}"`);
        if (input.output_mode && input.output_mode !== "files_with_matches") parts.push(`output_mode: "${input.output_mode}"`);
        if (input.head_limit !== undefined) parts.push(`head_limit: ${input.head_limit}`);
        return parts.join(", ");
    },

    async call(input: any, { abortController }: any) {
        const {
            pattern,
            path: searchPath,
            glob,
            type,
            output_mode = "files_with_matches",
            "-B": before,
            "-A": after,
            "-C": contextLines,
            "-n": lineNumbers = true,
            "-i": caseInsensitive = false,
            head_limit: limit,
            offset = 0,
            multiline = false
        } = input;

        const args: string[] = ["--hidden"];
        for (const ex of EXCLUDES) args.push("--glob", `!${ex}`);
        args.push("--max-columns", "500");

        if (multiline) args.push("-U", "--multiline-dotall");
        if (caseInsensitive) args.push("-i");
        if (output_mode === "files_with_matches") args.push("-l");
        else if (output_mode === "count") args.push("-c");

        if (lineNumbers && output_mode === "content") args.push("-n");
        if (contextLines !== undefined && output_mode === "content") args.push("-C", contextLines.toString());
        else if (output_mode === "content") {
            if (before !== undefined) args.push("-B", before.toString());
            if (after !== undefined) args.push("-A", after.toString());
        }

        if (pattern.startsWith("-")) args.push("-e", pattern);
        else args.push(pattern);

        if (type) args.push("--type", type);
        if (glob) {
            const globs = glob.split(/\s+/);
            for (const g of globs) {
                if (g.includes("{") && g.includes("}")) args.push("--glob", g);
                else {
                    const parts = g.split(",").filter(Boolean);
                    for (const p of parts) args.push("--glob", p);
                }
            }
        }

        const cwd = searchPath ? resolvePath(searchPath) : process.cwd();

        try {
            const results = await runRipgrep(args, cwd, abortController?.signal);

            if (output_mode === "content") {
                const head = results.slice(offset, limit ? offset + limit : undefined);
                return {
                    data: {
                        mode: "content",
                        numFiles: 0,
                        filenames: [],
                        content: head.join("\n"),
                        numLines: head.length,
                        ...(limit !== undefined && { appliedLimit: limit }),
                        ...(offset > 0 && { appliedOffset: offset })
                    }
                };
            }

            if (output_mode === "count") {
                const head = results.slice(offset, limit ? offset + limit : undefined);
                let totalMatches = 0;
                let filesCount = 0;
                for (const line of head) {
                    const lastColon = line.lastIndexOf(":");
                    if (lastColon > 0) {
                        const count = parseInt(line.substring(lastColon + 1), 10);
                        if (!isNaN(count)) {
                            totalMatches += count;
                            filesCount++;
                        }
                    }
                }
                return {
                    data: {
                        mode: "count",
                        numFiles: filesCount,
                        filenames: [],
                        content: head.join("\n"),
                        numMatches: totalMatches,
                        ...(limit !== undefined && { appliedLimit: limit }),
                        ...(offset > 0 && { appliedOffset: offset })
                    }
                };
            }

            // files_with_matches
            // Sort by mtime as in chunk_380 (simplified here)
            const sortedResults = results.map((r: string) => path.resolve(cwd, r));
            const head = sortedResults.slice(offset, limit ? offset + limit : undefined);

            return {
                data: {
                    mode: "files_with_matches",
                    filenames: head.map((f: string) => path.relative(process.cwd(), f)),
                    numFiles: head.length,
                    ...(limit !== undefined && { appliedLimit: limit }),
                    ...(offset > 0 && { appliedOffset: offset })
                }
            };

        } catch (error: any) {
            logError("GrepSearchTool: ripgrep failed", error);
            throw error;
        }
    },

    mapToolResultToToolResultBlockParam(result: any, toolUseId: string) {
        const { mode = "files_with_matches", numFiles, filenames, content, numMatches, appliedLimit, appliedOffset } = result;
        const pagination = (appliedLimit !== undefined || appliedOffset > 0) ? `limit: ${appliedLimit}, offset: ${appliedOffset ?? 0}` : "";

        if (mode === "content") {
            let output = content || "No matches found";
            if (pagination) output += `\n\n[Showing results with pagination = ${pagination}]`;
            return { tool_use_id: toolUseId, type: "tool_result", content: output };
        }

        if (mode === "count") {
            const matchesText = numMatches === 1 ? "occurrence" : "occurrences";
            const filesText = numFiles === 1 ? "file" : "files";
            const output = (content || "No matches found") + `\n\nFound ${numMatches ?? 0} total ${matchesText} across ${numFiles ?? 0} ${filesText}.` + (pagination ? ` with pagination = ${pagination}` : "");
            return { tool_use_id: toolUseId, type: "tool_result", content: output };
        }

        if (numFiles === 0) return { tool_use_id: toolUseId, type: "tool_result", content: "No files found" };

        let output = `Found ${numFiles} file${numFiles === 1 ? "" : "s"}${pagination ? ` ${pagination}` : ""}\n${filenames.join("\n")}`;
        return { tool_use_id: toolUseId, type: "tool_result", content: output };
    }
};

const FileSearchInputSchema = z.object({
    pattern: z.string().describe("The glob pattern to match files against"),
    path: z.string().optional().describe('The directory to search in. If not specified, the current working directory will be used. IMPORTANT: Omit this field to use the default directory. DO NOT enter "undefined" or "null" - simply omit it for the default behavior. Must be a valid directory path if provided.')
});

const FileSearchOutputSchema = z.object({
    durationMs: z.number(),
    numFiles: z.number(),
    filenames: z.array(z.string()),
    truncated: z.boolean()
});

export const FileSearchTool: Tool = {
    name: "FileSearchTool",
    strict: true,
    input_examples: [{ pattern: "*.ts" }],
    description: async () => "Find files matching a glob pattern.",
    userFacingName: () => "Search",
    getToolUseSummary: (input: any) => `Searching for files matching "${input.pattern}"`,
    prompt: async () => "Find files by name or glob pattern.",
    isEnabled: () => true,
    inputSchema: FileSearchInputSchema,
    outputSchema: FileSearchOutputSchema,
    isConcurrencySafe: () => true,
    isReadOnly: () => true,
    isSearchOrReadCommand: () => ({ isSearch: true, isRead: false }),
    getPath: (input: any) => input.path ? resolvePath(input.path) : process.cwd(),

    async validateInput(input: any) {
        if (input.path) {
            const absPath = resolvePath(input.path);
            if (!fs.existsSync(absPath)) return { result: false, message: "Path does not exist", errorCode: 1 };
            if (!fs.statSync(absPath).isDirectory()) return { result: false, message: "Path is not a directory", errorCode: 2 };
        }
        return { result: true };
    },

    async checkPermissions(input: any, context: any) {
        return true;
    },

    async call(input: any, { abortController }: any) {
        const start = Date.now();
        const { pattern, path: searchPath } = input;
        const cwd = searchPath ? resolvePath(searchPath) : process.cwd();

        // Use `rg --files` as a fallback for file search if `fd` is not available
        // Or just use `rg --files` with a glob
        const args = ["--files", "--hidden", "--glob", pattern];
        for (const ex of EXCLUDES) args.push("--glob", `!${ex}`);

        try {
            const results = await runRipgrep(args, cwd, abortController?.signal);
            const truncated = results.length > 100;
            const finalFiles = results.slice(0, 100);

            return {
                data: {
                    durationMs: Date.now() - start,
                    numFiles: results.length,
                    filenames: finalFiles.map((f: string) => path.relative(process.cwd(), path.resolve(cwd, f))),
                    truncated
                }
            };
        } catch (error: any) {
            logError("FileSearchTool: failed", error);
            throw error;
        }
    },

    mapToolResultToToolResultBlockParam(result: any, toolUseId: string) {
        if (result.filenames.length === 0) return { tool_use_id: toolUseId, type: "tool_result", content: "No files found" };
        let content = result.filenames.join("\n");
        if (result.truncated) content += "\n(Results are truncated. Consider using a more specific path or pattern.)";
        return { tool_use_id: toolUseId, type: "tool_result", content };
    }
};
