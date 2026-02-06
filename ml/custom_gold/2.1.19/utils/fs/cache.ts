/**
 * File: src/utils/fs/cache.ts
 * Role: Manages instruction loading, file system scanning, and context initialization for rules.
 */

import {
    join,
    dirname,
    extname,
    resolve,
    parse as pathParse
} from "node:path";
import * as fs from 'node:fs';
import { homedir } from 'node:os';
import matter from 'gray-matter';

import { Lexer } from 'marked';

// --- Types ---

export interface FileInfo {
    path: string;
    type: string;
    content: string;
    globs?: string[];
    parent?: string;
}

export interface ScanOptions {
    rulesDir: string;
    type: string;
    processedPaths: Set<string>;
    includeExternal: boolean;
    conditionalRule?: boolean;
    visitedDirs?: Set<string>;
}

// --- Constants ---

const TEXT_EXTENSIONS = new Set(['.md', '.txt', '.js', '.ts', '.jsx', '.tsx', '.json', '.yaml', '.yml', '.html', '.css', '.scss']);
const MAX_RECURSION_DEPTH = 5;
// --- Helpers ---

/**
 * Resolves symlinks and returns the real path.
 */
function resolvePath(filePath: string): { resolvedPath: string; isSymlink: boolean } {
    try {
        const stats = fs.lstatSync(filePath);
        if (stats.isSymbolicLink()) {
            return { resolvedPath: fs.realpathSync(filePath), isSymlink: true };
        }
        return { resolvedPath: filePath, isSymlink: false };
    } catch {
        return { resolvedPath: filePath, isSymlink: false };
    }
}

/**
 * Parses frontmatter and extracts include paths if present.
 */
function parseFrontMatter(fileContent: string): { content: string; paths?: string[] } {
    try {
        const { data, content } = matter(fileContent);
        if (!data || !data.paths || !Array.isArray(data.paths)) {
            return { content };
        }
        const paths = data.paths
            .map((p: string) => (p.endsWith('/**') ? p.slice(0, -3) : p))
            .filter((p: string) => p.length > 0 && p !== "**");
        return { content, paths };
    } catch {
        return { content: fileContent };
    }
}

/**
 * Reads and parses a file, checking for text validity.
 */
function readFileAndParse(filePath: string, fileType: string): FileInfo | null {
    try {
        if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) return null;

        const extension = extname(filePath).toLowerCase();
        if (extension && !TEXT_EXTENSIONS.has(extension)) return null;

        const rawContent = fs.readFileSync(filePath, "utf-8");
        const { content, paths } = parseFrontMatter(rawContent);

        return {
            path: filePath,
            type: fileType,
            content,
            globs: paths
        };
    } catch (error: any) {
        if (error.code === 'EACCES') {
            console.warn(`[Cache] Permission denied: ${filePath}`);
        }
        return null;
    }
}

/**
 * Extracts "@include" paths from markdown content.
 */
export function extractIncludePaths(content: string, contextPath: string): string[] {
    const pathsSet = new Set<string>();
    const tokens = new Lexer({ gfm: false }).lex(content);

    function traverse(tokens: any[]) {
        for (const token of tokens) {
            if (token.type === "code" || token.type === "codespan") continue;
            if (token.type === "text") {
                const regex = /(?:^|\s)@((?:[^\s\\]|\\ )+)/g;
                let match;
                while ((match = regex.exec(token.text || "")) !== null) {
                    const matchedPath = match[1]?.replace(/\\ /g, " ");
                    if (matchedPath && matchedPath !== "/") {
                        const absPath = resolve(dirname(contextPath), matchedPath);
                        pathsSet.add(absPath);
                    }
                }
            }
            if (token.tokens) traverse(token.tokens);
            if (token.items) traverse(token.items);
        }
    }
    traverse(tokens);
    return Array.from(pathsSet);
}

/**
 * Recursively processes a file and its includes.
 */
export function processFileRecursively(
    filePath: string,
    fileType: string,
    visitedPaths: Set<string>,
    includeExternal: boolean,
    depth = 0,
    parentPath?: string
): FileInfo[] {
    if (visitedPaths.has(filePath) || depth >= MAX_RECURSION_DEPTH) return [];

    const { resolvedPath, isSymlink } = resolvePath(filePath);
    visitedPaths.add(filePath);
    if (isSymlink) visitedPaths.add(resolvedPath);

    const fileInfo = readFileAndParse(filePath, fileType);
    if (!fileInfo || !fileInfo.content.trim()) return [];

    if (parentPath) fileInfo.parent = parentPath;
    const results = [fileInfo];

    const includes = extractIncludePaths(fileInfo.content, resolvedPath);
    for (const inc of includes) {
        // Basic safety check could go here
        const nested = processFileRecursively(inc, fileType, visitedPaths, includeExternal, depth + 1, filePath);
        results.push(...nested);
    }

    return results;
}

/**
 * Scans a directory for rule files (.md).
 */
export function scanDirectoryForRules(options: ScanOptions): FileInfo[] {
    const { rulesDir, type, processedPaths, includeExternal, conditionalRule, visitedDirs = new Set() } = options;
    if (visitedDirs.has(rulesDir)) return [];

    try {
        if (!fs.existsSync(rulesDir) || !fs.statSync(rulesDir).isDirectory()) return [];

        const { resolvedPath, isSymlink } = resolvePath(rulesDir);
        visitedDirs.add(rulesDir);
        if (isSymlink) visitedDirs.add(resolvedPath);

        const results: FileInfo[] = [];
        const entries = fs.readdirSync(resolvedPath, { withFileTypes: true });

        for (const entry of entries) {
            const fullPath = join(rulesDir, entry.name);
            const { resolvedPath: resPath, isSymlink: sym } = resolvePath(fullPath);
            const stats = sym ? fs.statSync(resPath) : entry;
            const isDir = stats.isDirectory();
            const isFile = stats.isFile();

            if (isDir) {
                results.push(...scanDirectoryForRules({ ...options, rulesDir: resPath, visitedDirs }));
            } else if (isFile && entry.name.endsWith(".md")) {
                const fileInfos = processFileRecursively(resPath, type, processedPaths, includeExternal);
                results.push(...fileInfos.filter(fi => conditionalRule ? (fi.globs && fi.globs.length > 0) : (!fi.globs || fi.globs.length === 0)));
            }
        }
        return results;
    } catch {
        return [];
    }
}

/**
 * Loads all context files (Instructions, Rules, Project settings).
 */
export function loadContextFiles(): FileInfo[] {
    const results: FileInfo[] = [];
    const processedPaths = new Set<string>();
    const visitedDirs = new Set<string>();
    const includeExternal = true; // Simplified for this refinement

    // 1. Managed Instructions
    const managedDir = join(process.cwd(), '.claude', 'managed'); // Placeholder
    results.push(...scanDirectoryForRules({ rulesDir: managedDir, type: "Managed", processedPaths, includeExternal }));

    // 2. User Instructions
    const userDir = join(homedir(), '.claude', 'rules');
    results.push(...scanDirectoryForRules({ rulesDir: userDir, type: "User", processedPaths, includeExternal }));

    // 3. Project Instructions (traversing up)
    let currentDir = process.cwd();
    while (currentDir !== pathParse(currentDir).root) {
        const projRules = join(currentDir, ".claude", "rules");
        results.push(...scanDirectoryForRules({ rulesDir: projRules, type: "Project", processedPaths, includeExternal, visitedDirs }));

        const claudeMd = join(currentDir, "CLAUDE.md");
        if (fs.existsSync(claudeMd)) results.push(...processFileRecursively(claudeMd, "Project", processedPaths, includeExternal));

        currentDir = dirname(currentDir);
    }

    return results;
}
