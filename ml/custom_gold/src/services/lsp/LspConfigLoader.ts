
import * as fs from 'node:fs/promises';
import { existsSync, readFileSync, statSync } from 'node:fs';
import * as path from 'node:path';
import { z } from 'zod';
import { log } from '../logger/loggerService.js';


const LspServerConfigSchema = z.object({
    command: z.string(),
    args: z.array(z.string()).optional(),
    env: z.record(z.string(), z.string()).optional(),
    workspaceFolder: z.string().optional(),
    extensionToLanguage: z.record(z.string(), z.string()).optional(),
    maxRestarts: z.number().optional(),
    restartOnCrash: z.boolean().optional(),
    startupTimeout: z.number().optional(),
    shutdownTimeout: z.number().optional(),
    initializationOptions: z.record(z.string(), z.any()).optional()
});

export type LspServerConfig = z.infer<typeof LspServerConfigSchema>;

function resolvePluginPath(pluginPath: string, relativePath: string): string | null {
    const absolute = path.resolve(pluginPath, relativePath);
    const relativePart = path.relative(pluginPath, absolute);
    if (relativePart.startsWith('..') || path.isAbsolute(relativePart)) return null;
    return absolute;
}

export async function loadLspConfigsFromPlugin(plugin: { name: string, path: string, manifest: any, enabled: boolean }, errors: any[] = []): Promise<Record<string, LspServerConfig> | undefined> {
    if (!plugin.enabled) return;

    const configs: Record<string, LspServerConfig> = {};
    const lspJsonPath = path.join(plugin.path, ".lsp.json");

    // Try .lsp.json
    try {
        if (existsSync(lspJsonPath)) {
            const content = await fs.readFile(lspJsonPath, "utf-8");
            const parsed = JSON.parse(content);
            const validated = z.record(z.string(), LspServerConfigSchema).safeParse(parsed);
            if (validated.success) {
                Object.assign(configs, validated.data);
            } else {
                errors.push({ type: "lsp-config-invalid", plugin: plugin.name, serverName: ".lsp.json", validationError: validated.error.message });
            }
        }
    } catch (err: any) {
        if (err.code !== "ENOENT") {
            errors.push({ type: "lsp-config-invalid", plugin: plugin.name, serverName: ".lsp.json", validationError: err.message });
        }
    }

    // Try manifest.lspServers
    if (plugin.manifest.lspServers) {
        const lspServers = Array.isArray(plugin.manifest.lspServers) ? plugin.manifest.lspServers : [plugin.manifest.lspServers];
        for (const entry of lspServers) {
            if (typeof entry === "string") {
                const configPath = resolvePluginPath(plugin.path, entry);
                if (!configPath) continue;
                try {
                    const content = await fs.readFile(configPath, "utf-8");
                    const parsed = JSON.parse(content);
                    const validated = z.record(z.string(), LspServerConfigSchema).safeParse(parsed);
                    if (validated.success) {
                        Object.assign(configs, validated.data);
                    }
                } catch (err: any) {
                    errors.push({ type: "lsp-config-invalid", plugin: plugin.name, serverName: entry, validationError: err.message });
                }
            } else if (typeof entry === "object") {
                for (const [name, config] of Object.entries(entry)) {
                    const validated = LspServerConfigSchema.safeParse(config);
                    if (validated.success) {
                        configs[name] = validated.data;
                    }
                }
            }
        }
    }

    // Expand variables (${CLAUDE_PLUGIN_ROOT})
    for (const [name, config] of Object.entries(configs)) {
        configs[name] = expandConfigVariables(config, plugin.path);
    }

    return Object.keys(configs).length > 0 ? configs : undefined;
}

function expandConfigVariables(config: LspServerConfig, pluginPath: string): LspServerConfig {
    const expand = (str: string) => str.replace(/\$\{CLAUDE_PLUGIN_ROOT\}/g, pluginPath);

    return {
        ...config,
        command: expand(config.command),
        args: config.args?.map(expand),
        env: config.env ? Object.fromEntries(Object.entries(config.env).map(([k, v]) => [k, expand(v as string)])) : undefined,
        workspaceFolder: config.workspaceFolder ? expand(config.workspaceFolder) : undefined
    };
}
