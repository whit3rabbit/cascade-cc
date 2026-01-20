import * as fs from 'node:fs';
import * as path from 'node:path';
import { z } from 'zod';

export const PluginManifestSchema = z.object({
    name: z.string(),
    id: z.string(),
    version: z.string().optional(),
    description: z.string().optional(),
    author: z.string().optional(),
    commands: z.array(z.string()).optional(),
    agents: z.array(z.string()).optional(),
    skills: z.array(z.string()).optional(),
    mcpServers: z.record(z.any()).optional()
}).passthrough();

export const MarketplaceManifestSchema = z.object({
    plugins: z.array(z.object({
        name: z.string(),
        id: z.string(),
        source: z.union([z.string(), z.record(z.any())])
    })),
    metadata: z.object({
        name: z.string().optional(),
        description: z.string().optional()
    }).optional()
}).passthrough();

export interface ValidationResult {
    success: boolean;
    errors: Array<{ path: string, message: string }>;
    warnings: Array<{ path: string, message: string }>;
    filePath: string;
    fileType: 'plugin' | 'marketplace' | 'unknown';
}

function getFileType(filePath: string): 'plugin' | 'marketplace' | 'unknown' {
    const filename = path.basename(filePath);
    if (filename === 'plugin.json') return 'plugin';
    if (filename === 'marketplace.json') return 'marketplace';
    const parentDir = path.basename(path.dirname(filePath));
    if (parentDir === '.claude-plugin') return 'plugin';
    return 'unknown';
}

function checkPathTraversal(val: string, location: string, errors: any[]) {
    if (val.includes('..')) {
        errors.push({
            path: location,
            message: `Path contains ".." which could be a path traversal attempt: ${val}`
        });
    }
}

export function validatePlugin(filePath: string): ValidationResult {
    const errors: any[] = [];
    const warnings: any[] = [];
    const absolutePath = path.resolve(filePath);

    if (!fs.existsSync(absolutePath)) {
        return { success: false, errors: [{ path: 'file', message: `File not found: ${absolutePath}` }], warnings, filePath: absolutePath, fileType: 'plugin' };
    }

    try {
        const content = fs.readFileSync(absolutePath, 'utf-8');
        const json = JSON.parse(content);

        // Path traversal checks (from tbA in chunk_572)
        if (json.commands) {
            (Array.isArray(json.commands) ? json.commands : [json.commands]).forEach((cmd: any, i: number) => {
                if (typeof cmd === 'string') checkPathTraversal(cmd, `commands[${i}]`, errors);
            });
        }
        if (json.agents) {
            (Array.isArray(json.agents) ? json.agents : [json.agents]).forEach((agt: any, i: number) => {
                if (typeof agt === 'string') checkPathTraversal(agt, `agents[${i}]`, errors);
            });
        }
        if (json.skills) {
            (Array.isArray(json.skills) ? json.skills : [json.skills]).forEach((sk: any, i: number) => {
                if (typeof sk === 'string') checkPathTraversal(sk, `skills[${i}]`, errors);
            });
        }

        const result = PluginManifestSchema.safeParse(json);
        if (!result.success) {
            result.error.errors.forEach(err => {
                errors.push({ path: err.path.join('.'), message: err.message });
            });
        } else {
            const data = result.data;
            if (!data.version) warnings.push({ path: 'version', message: 'No version specified. Consider adding a version following semver (e.g., "1.0.0")' });
            if (!data.description) warnings.push({ path: 'description', message: 'No description provided.' });
            if (!data.author) warnings.push({ path: 'author', message: 'No author information provided.' });
        }
    } catch (e: any) {
        errors.push({ path: 'json', message: `Failed to parse or read file: ${e.message}` });
    }

    return { success: errors.length === 0, errors, warnings, filePath: absolutePath, fileType: 'plugin' };
}

export function validateMarketplace(filePath: string): ValidationResult {
    const errors: any[] = [];
    const warnings: any[] = [];
    const absolutePath = path.resolve(filePath);

    if (!fs.existsSync(absolutePath)) {
        return { success: false, errors: [{ path: 'file', message: `File not found: ${absolutePath}` }], warnings, filePath: absolutePath, fileType: 'marketplace' };
    }

    try {
        const content = fs.readFileSync(absolutePath, 'utf-8');
        const json = JSON.parse(content);

        // Path traversal checks
        if (Array.isArray(json.plugins)) {
            json.plugins.forEach((p: any, i: number) => {
                if (p && typeof p === 'object' && p.source) {
                    if (typeof p.source === 'string') checkPathTraversal(p.source, `plugins[${i}].source`, errors);
                    else if (typeof p.source === 'object' && typeof p.source.path === 'string') checkPathTraversal(p.source.path, `plugins[${i}].source.path`, errors);
                }
            });
        }

        const result = MarketplaceManifestSchema.safeParse(json);
        if (!result.success) {
            result.error.errors.forEach(err => {
                errors.push({ path: err.path.join('.'), message: err.message });
            });
        } else {
            const data = result.data;
            if (!data.plugins || data.plugins.length === 0) warnings.push({ path: 'plugins', message: 'Marketplace has no plugins defined' });

            const names = new Set<string>();
            data.plugins.forEach((p, i) => {
                if (names.has(p.name)) {
                    errors.push({ path: `plugins[${i}].name`, message: `Duplicate plugin name "${p.name}" found in marketplace` });
                }
                names.add(p.name);
            });

            if (!data.metadata?.description) warnings.push({ path: 'metadata.description', message: 'No marketplace description provided.' });
        }
    } catch (e: any) {
        errors.push({ path: 'json', message: `Failed to parse or read file: ${e.message}` });
    }

    return { success: errors.length === 0, errors, warnings, filePath: absolutePath, fileType: 'marketplace' };
}

export function validateManifest(inputPath: string): ValidationResult {
    const absolutePath = path.resolve(inputPath);

    if (fs.existsSync(absolutePath) && fs.statSync(absolutePath).isDirectory()) {
        const mPath = path.join(absolutePath, '.claude-plugin', 'marketplace.json');
        const pPath = path.join(absolutePath, '.claude-plugin', 'plugin.json');
        if (fs.existsSync(mPath)) return validateMarketplace(mPath);
        if (fs.existsSync(pPath)) return validatePlugin(pPath);

        return {
            success: false,
            errors: [{ path: 'directory', message: 'No manifest found in directory. Expected .claude-plugin/marketplace.json or .claude-plugin/plugin.json' }],
            warnings: [],
            filePath: absolutePath,
            fileType: 'unknown'
        };
    }

    const type = getFileType(absolutePath);
    if (type === 'marketplace') return validateMarketplace(absolutePath);
    if (type === 'plugin') return validatePlugin(absolutePath);

    // If unknown, try to guess by content
    try {
        const content = fs.readFileSync(absolutePath, 'utf-8');
        const json = JSON.parse(content);
        if (json && Array.isArray(json.plugins)) return validateMarketplace(absolutePath);
    } catch { }

    return validatePlugin(absolutePath);
}
