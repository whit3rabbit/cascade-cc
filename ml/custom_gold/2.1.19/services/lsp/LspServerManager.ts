/**
 * File: src/services/lsp/LspServerManager.ts
 * Role: Manages lifecycle and communication with LSP servers.
 */

import * as path from 'node:path';
import { readFileSync } from 'node:fs';
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
}

export class LspServerManager {
    private servers: Map<string, LspClient> = new Map();
    private extensionToServers: Map<string, string[]> = new Map();
    private openFiles: Map<string, string> = new Map(); // uri -> serverName
    private isInitialized = false;
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

    async initialize(): Promise<void> {
        if (this.isInitialized) return;

        // Register cleanup
        onCleanup(() => this.shutdown());

        try {
            // In a real implementation, we would load configs from settings, 
            // project .lsp.json, and plugins. For now, we use a basic set 
            // or rely on dynamic discovery.
            const allConfigs = await this.getAllLspConfigs();

            for (const [name, config] of Object.entries(allConfigs)) {
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

    private async getAllLspConfigs(): Promise<Record<string, LspServerConfig>> {
        // Simplified config loading logic.
        // In 2.1.19 this merges settings, enterprise, project, and plugins.
        return {
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
    }

    async shutdown(): Promise<void> {
        const stopPromises = Array.from(this.servers.values()).map(client => client.stop());
        await Promise.allSettled(stopPromises);
        this.servers.clear();
        this.extensionToServers.clear();
        this.openFiles.clear();
        this.isInitialized = false;
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
        return client.sendRequest(method, params);
    }

    async openFile(filePath: string, content: string): Promise<void> {
        const client = await this.ensureServerStarted(filePath);
        if (!client) return;

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

        await client.sendNotification("textDocument/didSave", {
            textDocument: {
                uri: `file://${path.resolve(filePath)}`
            }
        });
    }

    async closeFile(filePath: string): Promise<void> {
        const client = this.getServerForFile(filePath);
        if (!client || client.state !== "running") return;

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
