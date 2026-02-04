/**
 * File: src/services/conversation/CategoryRuleCache.ts
 * Role: Discovers and caches instructions from various CLAUDE.md scopes.
 */

import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { join, dirname, resolve, basename, extname } from 'node:path';
import { homedir } from 'node:os';

import { getBaseConfigDir, getManagedRulesDirectory } from '../../utils/shared/runtimeAndEnv.js';
import { EnvService } from '../config/EnvService.js';

export type RuleScope = 'Managed' | 'User' | 'Project' | 'Local';

export interface Rule {
    path: string;
    type: RuleScope;
    content: string;
    globs?: string[];
    parent?: string;
}

const MAX_RECURSION_DEPTH = 5;
const TEXT_EXTENSIONS = new Set(['.md', '.txt', '.py', '.js', '.ts', '.tsx', '.jsx', '.go', '.java', '.c', '.cpp', '.h', '.hpp', '.sh', '.bash', '.json', '.yaml', '.yml', '.toml']);

export class CategoryRuleCache {
    private static instance: CategoryRuleCache;
    private rules: Rule[] = [];
    private lastLoadTime: number = 0;
    private cacheTTL: number = 5000; // 5 seconds TTL for rule discovery

    private constructor() { }

    public static getInstance(): CategoryRuleCache {
        if (!CategoryRuleCache.instance) {
            CategoryRuleCache.instance = new CategoryRuleCache();
        }
        return CategoryRuleCache.instance;
    }

    /**
     * Discovers and loads rules from all scopes.
     */
    public loadRules(force: boolean = false): Rule[] {
        const now = Date.now();
        if (!force && this.rules.length > 0 && (now - this.lastLoadTime < this.cacheTTL)) {
            return this.rules;
        }

        const discoveredRules: Rule[] = [];
        const processedPaths = new Set<string>();

        // 1. Managed Scope (App-wide defaults)
        const managedRulesDir = getManagedRulesDirectory();
        discoveredRules.push(...this.loadRulesFromDir(managedRulesDir, 'Managed', processedPaths));

        // 2. User Scope (~/.claude/rules and ~/.claude/CLAUDE.md)
        const userConfigDir = getBaseConfigDir();
        const userRulesDir = join(userConfigDir, 'rules');
        const userClaudeMd = join(userConfigDir, 'CLAUDE.md');

        discoveredRules.push(...this.loadRulesFromDir(userRulesDir, 'User', processedPaths));
        if (existsSync(userClaudeMd)) {
            discoveredRules.push(...this.processRuleFile(userClaudeMd, 'User', processedPaths));
        }

        // 3. Project & Local Scope (Scanning upwards from CWD)
        let currentDir = process.cwd();
        const projectDirs: string[] = [];

        // Find all parent directories to search for CLAUDE.md files
        let dir = currentDir;
        const root = resolve('/');
        while (dir !== root) {
            projectDirs.push(dir);
            const parent = dirname(dir);
            if (parent === dir) break;
            dir = parent;
        }
        projectDirs.push(root);

        // Process in reverse (root to CWD) to ensure closer rules take precedence or are logically grouped
        for (const dir of projectDirs.reverse()) {
            // Project Context (CLAUDE.md)
            const projectClaudeMd = join(dir, 'CLAUDE.md');
            const dotClaudeRules = join(dir, '.claude', 'rules');
            const dotClaudeClaudeMd = join(dir, '.claude', 'CLAUDE.md');

            if (existsSync(projectClaudeMd)) {
                discoveredRules.push(...this.processRuleFile(projectClaudeMd, 'Project', processedPaths));
            }
            if (existsSync(dotClaudeClaudeMd)) {
                discoveredRules.push(...this.processRuleFile(dotClaudeClaudeMd, 'Project', processedPaths));
            }
            if (existsSync(dotClaudeRules) && statSync(dotClaudeRules).isDirectory()) {
                discoveredRules.push(...this.loadRulesFromDir(dotClaudeRules, 'Project', processedPaths));
            }

            // Local overrides (CLAUDE.local.md) - typically in the project root or CWD
            const localClaudeMd = join(dir, 'CLAUDE.local.md');
            if (existsSync(localClaudeMd)) {
                discoveredRules.push(...this.processRuleFile(localClaudeMd, 'Local', processedPaths));
            }
        }

        this.rules = discoveredRules;
        this.lastLoadTime = now;
        return discoveredRules;
    }

    /**
     * Formats the instructions for the system prompt.
     */
    public getInstructions(): string {
        const rules = this.loadRules();
        if (rules.length === 0) return '';

        const ruleBlocks = rules
            .filter(r => r.content.trim().length > 0)
            .map(r => {
                let scopeDesc = '';
                switch (r.type) {
                    case 'Project': scopeDesc = ' (project instructions, checked into the codebase)'; break;
                    case 'Local': scopeDesc = " (user's private project instructions, not checked in)"; break;
                    case 'User': scopeDesc = " (user's private global instructions for all projects)"; break;
                    case 'Managed': scopeDesc = " (managed global instructions)"; break;
                }
                return `Contents of ${r.path}${scopeDesc}:\n\n${r.content.trim()}`;
            });

        if (ruleBlocks.length === 0) return '';

        return `The following are additional instructions that you must follow for this project:\n\n${ruleBlocks.join('\n\n')}`;
    }

    private loadRulesFromDir(dir: string, type: RuleScope, processedPaths: Set<string>): Rule[] {
        if (!existsSync(dir) || !statSync(dir).isDirectory()) return [];

        const results: Rule[] = [];
        try {
            const entries = readdirSync(dir, { withFileTypes: true });
            for (const entry of entries) {
                const fullPath = join(dir, entry.name);
                if (entry.isDirectory()) {
                    results.push(...this.loadRulesFromDir(fullPath, type, processedPaths));
                } else if (entry.isFile() && entry.name.endsWith('.md')) {
                    results.push(...this.processRuleFile(fullPath, type, processedPaths));
                }
            }
        } catch (error) {
            console.error(`Error reading rules from ${dir}:`, error);
        }
        return results;
    }

    private processRuleFile(path: string, type: RuleScope, processedPaths: Set<string>, depth: number = 0, parent?: string): Rule[] {
        if (processedPaths.has(path) || depth >= MAX_RECURSION_DEPTH) return [];
        processedPaths.add(path);

        if (!existsSync(path) || !statSync(path).isFile()) return [];

        const ext = extname(path).toLowerCase();
        if (ext && !TEXT_EXTENSIONS.has(ext)) {
            return [];
        }

        try {
            const rawContent = readFileSync(path, 'utf8');
            const { content, globs } = this.parseRuleContent(rawContent);

            const rule: Rule = { path, type, content, globs, parent };
            const results: Rule[] = [rule];

            // Handle @include references
            const includes = this.findIncludes(content, dirname(path));
            for (const includePath of includes) {
                results.push(...this.processRuleFile(includePath, type, processedPaths, depth + 1, path));
            }

            return results;
        } catch (error) {
            console.error(`Error processing rule file ${path}:`, error);
            return [];
        }
    }

    private parseRuleContent(content: string): { content: string; globs?: string[] } {
        // Simple frontmatter parser for `paths: [...]`
        const fmMatch = content.match(/^---\n([\s\S]*?)\n---/);
        if (!fmMatch) return { content };

        const fm = fmMatch[1];
        const pathsMatch = fm.match(/paths:\s*\[([\s\S]*?)\]/);
        const globs = pathsMatch
            ? pathsMatch[1].split(',').map(p => p.trim().replace(/['"]/g, '').replace(/\/\*\*$/, '')).filter(p => p.length > 0)
            : undefined;

        return {
            content: content.replace(fmMatch[0], '').trim(),
            globs
        };
    }

    private findIncludes(content: string, dir: string): string[] {
        const includes = new Set<string>();
        // Match @path/to/file or @./path/to/file or @~/path/to/file
        const regex = /(?:^|\s)@((?:[^\s\\]|\\ )+)/g;
        let match;
        while ((match = regex.exec(content)) !== null) {
            let p = match[1].replace(/\\ /g, ' ');
            if (!p) continue;

            let resolved: string;
            if (p.startsWith('~/')) {
                resolved = join(homedir(), p.slice(2));
            } else if (p.startsWith('./') || p.startsWith('../') || (!p.startsWith('/') && !p.startsWith('@'))) {
                resolved = resolve(dir, p);
            } else if (p.startsWith('/')) {
                resolved = p;
            } else {
                continue;
            }
            includes.add(resolved);
        }
        return Array.from(includes);
    }
}
