/**
 * File: src/services/lsp/LspServerManager.ts
 * Role: Manages lifecycle and communication with LSP servers.
 */

import * as path from 'node:path';
import { readFileSync, existsSync, accessSync, constants } from 'node:fs';
import { LspClient } from './LspClient.js';
import { detectProjectType } from './ProjectDetection.js';
import { getSettings } from '../config/SettingsService.js';
import { McpServerManager } from '../mcp/McpServerManager.js';
import { onCleanup } from '../../utils/cleanup.js';

export interface LspServerConfig {
    name: string;
    command: string;
    args: string[];
    env?: Record<string, string>;
    extensionToLanguage: Record<string, string>;
    transport?: "stdio" | "socket"; // Defaults to stdio
    initializationOptions?: any;
    settings?: any;
    workspaceFolder?: string;
    timeout?: number;
}

export class LspServerManager {
    private servers: Map<string, LspClient> = new Map();
    private extensionToServers: Map<string, string[]> = new Map();
    private serverConfigs: Map<string, LspServerConfig> = new Map();
    private openFiles: Map<string, string> = new Map(); // uri -> serverName
    private isInitialized = false;
    private lastRequestTime: Map<string, number> = new Map();
    private idleCheckInterval: NodeJS.Timeout | null = null;
    private readonly IDLE_TIMEOUT_MS = 30 * 60 * 1000; // 30 minutes
    private static instance: LspServerManager;

    constructor() {
        LspServerManager.instance = this;
    }

    static getInstance(): LspServerManager {
        if (!LspServerManager.instance) {
            LspServerManager.instance = new LspServerManager();
        }
        return LspServerManager.instance;
    }

    // Alias for deobfuscated code usage
    getAllLspServers() {
        return this.servers;
    }

    registerServerConfig(config: LspServerConfig) {
        this.serverConfigs.set(config.name, config);
        // Also update extension map if immediate effect is needed, 
        // but typically this is called before initialize or we need to re-init.
        // For now, let's allow dynamic addition.
        const extensions = Object.keys(config.extensionToLanguage);
        for (const ext of extensions) {
            const normalizedExt = ext.toLowerCase();
            if (!this.extensionToServers.has(normalizedExt)) {
                this.extensionToServers.set(normalizedExt, []);
            }
            const list = this.extensionToServers.get(normalizedExt)!;
            if (!list.includes(config.name)) {
                list.push(config.name);
            }
        }
    }

    async initialize(): Promise<void> {
        if (this.isInitialized) return;

        // Register cleanup
        onCleanup(() => this.shutdown());

        // Start idle checker
        this.startIdleChecker();

        try {
            // In a real implementation, we would load configs from settings, 
            // project .lsp.json, and plugins. For now, we use a basic set 
            // or rely on dynamic discovery.
            // Load default configs
            const defaultConfigs = await this.getDefaultLspConfigs();
            for (const config of Object.values(defaultConfigs)) {
                if (!this.serverConfigs.has(config.name)) {
                    this.registerServerConfig(config);
                }
            }

            // In a real implementation, we would load configs from settings, 
            // project .lsp.json, and plugins. 
            // PluginManager should have registered plugin configs by now if initialized before.

            for (const [name, config] of this.serverConfigs.entries()) {
                const extensions = Object.keys(config.extensionToLanguage);
                for (const ext of extensions) {
                    const normalizedExt = ext.toLowerCase();
                    if (!this.extensionToServers.has(normalizedExt)) {
                        this.extensionToServers.set(normalizedExt, []);
                    }
                    this.extensionToServers.get(normalizedExt)!.push(name);
                }

                const client = new LspClient(name, config);
                this.servers.set(name, client);

                // Set up handlers (e.g. workspace/configuration)
                client.onRequest("workspace/configuration", (params) => {
                    return params.items.map(() => null);
                });

                // Start the client (lazy start can be implemented in getServerForFile)
                client.start().catch(err => {
                    console.error(`[LspServerManager] Failed to start LSP server ${name}:`, err);
                });
            }

            this.isInitialized = true;
        } catch (error) {
            console.error("[LspServerManager] Initialization failed:", error);
            throw error;
        }
    }

    private hasExecutable(name: string): boolean {
        const pathEnv = process.env.PATH || '';
        const paths = pathEnv.split(path.delimiter);
        const exts = process.platform === 'win32' ? ['.exe', '.cmd', '.bat', ''] : [''];

        for (const p of paths) {
            for (const ext of exts) {
                try {
                    const fullPath = path.join(p, name + ext);
                    if (existsSync(fullPath)) {
                        try {
                            accessSync(fullPath, constants.X_OK);
                            return true;
                        } catch {
                            // Not executable
                        }
                    }
                } catch {
                    // Ignore path errors
                }
            }
        }
        return false;
    }

    private async getDefaultLspConfigs(): Promise<Record<string, LspServerConfig>> {
        const configs: Record<string, LspServerConfig> = {
            "typescript-language-server": {
                name: "typescript-language-server",
                command: "typescript-language-server",
                args: ["--stdio"],
                extensionToLanguage: {
                    ".ts": "typescript",
                    ".tsx": "typescriptreact",
                    ".js": "javascript",
                    ".jsx": "javascriptreact"
                }
            },
            "pyright": {
                name: "pyright",
                command: "pyright-langserver",
                args: ["--stdio"],
                extensionToLanguage: {
                    ".py": "python"
                }
            }
        };

        if (this.hasExecutable('gopls')) {
            configs['gopls'] = {
                name: 'gopls',
                command: 'gopls',
                args: [],
                extensionToLanguage: { '.go': 'go' }
            };
        }

        if (this.hasExecutable('rust-analyzer')) {
            configs['rust-analyzer'] = {
                name: 'rust-analyzer',
                command: 'rust-analyzer',
                args: [],
                extensionToLanguage: { '.rs': 'rust' }
            };
        }

        if (this.hasExecutable('clangd')) {
            configs['clangd'] = {
                name: 'clangd',
                command: 'clangd',
                args: [],
                extensionToLanguage: { '.c': 'c', '.cpp': 'cpp', '.h': 'c', '.hpp': 'cpp' }
            };
        }

        return configs;
    }

    async shutdown(): Promise<void> {
        if (this.idleCheckInterval) {
            clearInterval(this.idleCheckInterval);
            this.idleCheckInterval = null;
        }
        const stopPromises = Array.from(this.servers.values()).map(client => client.stop());
        await Promise.allSettled(stopPromises);
        this.servers.clear();
        this.extensionToServers.clear();
        this.openFiles.clear();
        this.lastRequestTime.clear();
        this.isInitialized = false;
    }

    private startIdleChecker() {
        if (this.idleCheckInterval) return;
        this.idleCheckInterval = setInterval(() => {
            const now = Date.now();
            for (const [name, client] of this.servers.entries()) {
                if (client.state === "running") {
                    const lastTime = this.lastRequestTime.get(name) || 0;
                    if (now - lastTime > this.IDLE_TIMEOUT_MS) {
                        console.log(`[LspServerManager] Shutting down idle server: ${name}`);
                        client.stop().catch(() => { });
                    }
                }
            }
        }, 60000); // Check every minute
    }

    private updateLastRequestTime(serverName: string) {
        this.lastRequestTime.set(serverName, Date.now());
    }

    getServerForFile(filePath: string): LspClient | undefined {
        const ext = path.extname(filePath).toLowerCase();
        const serverNames = this.extensionToServers.get(ext);
        if (!serverNames || serverNames.length === 0) return undefined;

        // Return the first available server for this extension
        return this.servers.get(serverNames[0]);
    }

    async ensureServerStarted(filePath: string): Promise<LspClient | undefined> {
        const client = this.getServerForFile(filePath);
        if (!client) return undefined;

        if (client.state === "stopped") {
            await client.start();
        }
        return client;
    }

    async sendRequest(filePath: string, method: string, params: any): Promise<any> {
        const client = await this.ensureServerStarted(filePath);
        if (!client) throw new Error(`No LSP server for ${filePath}`);
        this.updateLastRequestTime(client.name);
        return client.sendRequest(method, params);
    }

    async openFile(filePath: string, content: string): Promise<void> {
        const client = await this.ensureServerStarted(filePath);
        if (!client) return;
        this.updateLastRequestTime(client.name);

        const uri = `file://${path.resolve(filePath)}`;
        if (this.openFiles.get(uri) === client.name) return;

        const ext = path.extname(filePath).toLowerCase();
        const languageId = client.config.extensionToLanguage[ext] || "plaintext";

        await client.sendNotification("textDocument/didOpen", {
            textDocument: {
                uri,
                languageId,
                version: 1,
                text: content
            }
        });
        this.openFiles.set(uri, client.name);
    }

    async changeFile(filePath: string, content: string): Promise<void> {
        const client = this.getServerForFile(filePath);
        if (!client || client.state !== "running") {
            return this.openFile(filePath, content);
        }
        this.updateLastRequestTime(client.name);

        const uri = `file://${path.resolve(filePath)}`;
        if (this.openFiles.get(uri) !== client.name) {
            return this.openFile(filePath, content);
        }

        await client.sendNotification("textDocument/didChange", {
            textDocument: {
                uri,
                version: 1
            },
            contentChanges: [{ text: content }]
        });
    }

    async saveFile(filePath: string): Promise<void> {
        const client = this.getServerForFile(filePath);
        if (!client || client.state !== "running") return;
        this.updateLastRequestTime(client.name);

        await client.sendNotification("textDocument/didSave", {
            textDocument: {
                uri: `file://${path.resolve(filePath)}`
            }
        });
    }

    async closeFile(filePath: string): Promise<void> {
        const client = this.getServerForFile(filePath);
        if (!client || client.state !== "running") return;
        this.updateLastRequestTime(client.name);

        const uri = `file://${path.resolve(filePath)}`;
        await client.sendNotification("textDocument/didClose", {
            textDocument: { uri }
        });
        this.openFiles.delete(uri);
    }

    isFileOpen(filePath: string): boolean {
        const uri = `file://${path.resolve(filePath)}`;
        return this.openFiles.has(uri);
    }

    getAllServers(): Map<string, LspClient> {
        return this.servers;
    }
}
