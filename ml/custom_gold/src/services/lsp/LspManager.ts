import { LspClient } from "./LspClient.js";
import { LspServerConfig, loadLspConfigsFromPlugin } from "./LspConfigLoader.js";
import * as path from "node:path";
import { log } from "../logger/loggerService.js";
import { getProjectRoot } from "../../utils/shared/pathUtils.js";
import { getInstalledPlugins } from "../mcp/McpClientManager.js";

const logger = log("lsp-manager");



const CONTENT_MODIFIED_ERROR = -32801;
const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 500;

/**
 * Manages a single LSP server process instance.
 * Based on chunk_371.ts (T52)
 */
export class LspServerInstance {
    private client: LspClient;
    private _state: "stopped" | "starting" | "running" | "stopping" | "error" = "stopped";
    private _startTime?: Date;
    private _lastError?: Error;
    private _restartCount = 0;

    constructor(public readonly name: string, public readonly config: LspServerConfig) {
        this.client = new LspClient(name);
    }

    get state() { return this._state; }
    get startTime() { return this._startTime; }
    get lastError() { return this._lastError; }
    get restartCount() { return this._restartCount; }

    async start() {
        if (this._state === "running" || this._state === "starting") return;

        try {
            this._state = "starting";
            logger.info(`Starting LSP server instance: ${this.name}`);

            await this.client.start(this.config.command, this.config.args || [], {
                env: this.config.env,
                cwd: this.config.workspaceFolder
            });

            const workspaceRoot = this.config.workspaceFolder || getProjectRoot();
            const rootUri = `file://${workspaceRoot}`;

            const initParams = {
                processId: process.pid,
                initializationOptions: this.config.initializationOptions ?? {},
                workspaceFolders: [{
                    uri: rootUri,
                    name: path.basename(workspaceRoot)
                }],
                rootPath: workspaceRoot,
                rootUri: rootUri,
                capabilities: {
                    workspace: {
                        configuration: false,
                        workspaceFolders: false
                    },
                    textDocument: {
                        synchronization: {
                            dynamicRegistration: false,
                            willSave: false,
                            willSaveWaitUntil: false,
                            didSave: true
                        },
                        publishDiagnostics: {
                            relatedInformation: true,
                            tagSupport: {
                                valueSet: [1, 2]
                            },
                            versionSupport: false,
                            codeDescriptionSupport: true,
                            dataSupport: false
                        },
                        hover: {
                            dynamicRegistration: false,
                            contentFormat: ["markdown", "plaintext"]
                        },
                        definition: {
                            dynamicRegistration: false,
                            linkSupport: true
                        },
                        references: {
                            dynamicRegistration: false
                        },
                        documentSymbol: {
                            dynamicRegistration: false,
                            hierarchicalDocumentSymbolSupport: true
                        },
                        callHierarchy: {
                            dynamicRegistration: false
                        }
                    },
                    general: {
                        positionEncodings: ["utf-16"]
                    }
                }
            };

            await this.client.initialize(initParams);
            this._state = "running";
            this._startTime = new Date();
            logger.info(`LSP server instance started: ${this.name}`);
        } catch (error: any) {
            this._state = "error";
            this._lastError = error;
            logger.error(error);
            throw error;
        }
    }

    async stop() {
        if (this._state === "stopped" || this._state === "stopping") return;
        try {
            this._state = "stopping";
            await this.client.stop();
            this._state = "stopped";
            logger.info(`LSP server instance stopped: ${this.name}`);
        } catch (error: any) {
            this._state = "error";
            this._lastError = error;
            logger.error(error);
            throw error;
        }
    }

    async restart() {
        try {
            await this.stop();
        } catch (error: any) {
            const wrappedError = new Error(`Failed to stop LSP server '${this.name}' during restart: ${error.message}`);
            logger.error(wrappedError);
        }

        this._restartCount++;
        const max = this.config.maxRestarts ?? 3;
        if (this._restartCount > max) {
            const error = new Error(`Max restart attempts (${max}) exceeded for server '${this.name}'`);
            logger.error(error);
            throw error;
        }

        try {
            await this.start();
        } catch (error: any) {
            const wrappedError = new Error(`Failed to start LSP server '${this.name}' during restart (attempt ${this._restartCount}/${max}): ${error.message}`);
            logger.error(wrappedError);
            throw wrappedError;
        }
    }

    isHealthy() {
        return this._state === "running" && this.client.isInitialized;
    }

    async sendRequest(method: string, params: any) {
        if (!this.isHealthy()) {
            const error = new Error(`Cannot send request to LSP server '${this.name}': server is ${this._state}${this._lastError ? `, last error: ${this._lastError.message}` : ""}`);
            logger.error(error);
            throw error;
        }

        let lastErr: any;
        for (let i = 0; i <= MAX_RETRIES; i++) {
            try {
                return await this.client.sendRequest(method, params);
            } catch (err: any) {
                lastErr = err;
                if (err.code === CONTENT_MODIFIED_ERROR && i < MAX_RETRIES) {
                    const delay = RETRY_DELAY_MS * Math.pow(2, i);
                    logger.info(`LSP request '${method}' to '${this.name}' got ContentModified error, retrying in ${delay}ms (attempt ${i + 1}/${MAX_RETRIES})…`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                    continue;
                }
                break;
            }
        }
        const finalError = new Error(`LSP request '${method}' failed for server '${this.name}': ${lastErr?.message ?? "unknown error"}`);
        logger.error(finalError);
        throw lastErr || finalError;
    }

    async sendNotification(method: string, params: any) {
        if (!this.isHealthy()) {
            const error = new Error(`Cannot send notification to LSP server '${this.name}': server is ${this._state}`);
            logger.error(error);
            throw error;
        }
        try {
            await this.client.sendNotification(method, params);
        } catch (err: any) {
            const error = new Error(`LSP notification '${method}' failed for server '${this.name}': ${err.message}`);
            logger.error(error);
            throw error;
        }
    }

    onNotification(method: string, handler: (params: any) => void) {
        this.client.onNotification(method, handler);
    }

    onRequest(method: string, handler: (params: any) => Promise<any>) {
        this.client.onRequest(method, handler);
    }
}

/**
 * High-level manager for multiple LSP server instances.
 * Based on chunk_371.ts (b52)
 */
export class LspManager {
    private servers = new Map<string, LspServerInstance>();
    private extensionMap = new Map<string, string[]>();
    private fileMap = new Map<string, string>(); // fileUri -> serverName

    async initialize() {
        await this.loadServers();
    }

    async loadServers() {
        logger.info("Loading LSP servers from plugins...");
        try {
            const { enabled } = await getInstalledPlugins();
            for (const plugin of enabled) {
                try {
                    const configs = await loadLspConfigsFromPlugin(plugin);
                    if (configs) {
                        for (const [name, config] of Object.entries(configs)) {
                            this.addServer(name, config);
                        }
                    }
                } catch (err) {
                    logger.error(new Error(`Failed to load LSP configs from plugin ${plugin.name}: ${err}`));
                }
            }
        } catch (err) {
            logger.error(new Error(`Failed to get installed plugins for LSP loading: ${err}`));
        }
    }

    addServer(name: string, config: LspServerConfig) {
        const instance = new LspServerInstance(name, config);
        this.servers.set(name, instance);
        if (config.extensionToLanguage) {
            for (const ext of Object.keys(config.extensionToLanguage)) {
                const normalizedExt = ext.startsWith('.') ? ext.toLowerCase() : `.${ext.toLowerCase()}`;
                const existing = this.extensionMap.get(normalizedExt) || [];
                if (!existing.includes(name)) {
                    this.extensionMap.set(normalizedExt, [...existing, name]);
                }
            }
        }
    }

    async shutdown() {
        const errors: any[] = [];
        for (const [name, server] of Array.from(this.servers.entries())) {
            if (server.state === "running") {
                try {
                    await server.stop();
                } catch (err: any) {
                    logger.error(new Error(`Failed to stop LSP server ${name}: ${err.message}`));
                    errors.push(err);
                }
            }
        }
        this.servers.clear();
        this.extensionMap.clear();
        this.fileMap.clear();

        if (errors.length > 0) {
            const aggregatedError = new Error(`Failed to stop ${errors.length} LSP server(s): ${errors.map(e => e.message).join("; ")}`);
            logger.error(aggregatedError);
            throw aggregatedError;
        }
    }

    getServerForFile(filePath: string): LspServerInstance | undefined {
        const ext = path.extname(filePath).toLowerCase();
        const serverNames = this.extensionMap.get(ext);
        if (!serverNames || serverNames.length === 0) return undefined;
        const name = serverNames[0];
        return this.servers.get(name);
    }

    async ensureServerStarted(filePath: string) {
        const server = this.getServerForFile(filePath);
        if (!server) return;
        if (server.state === "stopped") {
            try {
                await server.start();
            } catch (err: any) {
                logger.error(new Error(`Failed to start LSP server for file ${filePath}: ${err.message}`));
                throw err;
            }
        }
        return server;
    }

    async sendRequest(filePath: string, method: string, params: any) {
        const server = await this.ensureServerStarted(filePath);
        if (!server) return;
        try {
            return await server.sendRequest(method, params);
        } catch (err: any) {
            logger.error(new Error(`LSP request failed for file ${filePath}, method '${method}': ${err.message}`));
            throw err;
        }
    }

    getAllServers() {
        return this.servers;
    }

    async openFile(filePath: string, content: string) {
        const server = await this.ensureServerStarted(filePath);
        if (!server) return;

        const uri = `file://${path.resolve(filePath)}`;
        if (this.fileMap.get(uri) === server.name) {
            logger.info(`LSP: File already open, skipping didOpen for ${filePath}`);
            return;
        }

        const ext = path.extname(filePath).toLowerCase();
        const languageId = server.config.extensionToLanguage?.[ext] || "plaintext";

        try {
            await server.sendNotification("textDocument/didOpen", {
                textDocument: {
                    uri,
                    languageId,
                    version: 1,
                    text: content
                }
            });
            this.fileMap.set(uri, server.name);
            logger.info(`LSP: Sent didOpen for ${filePath} (languageId: ${languageId})`);
        } catch (err: any) {
            logger.error(new Error(`Failed to sync file open ${filePath}: ${err.message}`));
            throw err;
        }
    }

    async changeFile(filePath: string, content: string) {
        const server = this.getServerForFile(filePath);
        if (!server || server.state !== "running") {
            return this.openFile(filePath, content);
        }

        const uri = `file://${path.resolve(filePath)}`;
        if (this.fileMap.get(uri) !== server.name) {
            return this.openFile(filePath, content);
        }

        try {
            await server.sendNotification("textDocument/didChange", {
                textDocument: { uri, version: 1 },
                contentChanges: [{ text: content }]
            });
            logger.info(`LSP: Sent didChange for ${filePath}`);
        } catch (err: any) {
            logger.error(new Error(`Failed to sync file change ${filePath}: ${err.message}`));
            throw err;
        }
    }

    async saveFile(filePath: string) {
        const server = this.getServerForFile(filePath);
        if (!server || server.state !== "running") return;

        try {
            await server.sendNotification("textDocument/didSave", {
                textDocument: {
                    uri: `file://${path.resolve(filePath)}`
                }
            });
            logger.info(`LSP: Sent didSave for ${filePath}`);
        } catch (err: any) {
            logger.error(new Error(`Failed to sync file save ${filePath}: ${err.message}`));
            throw err;
        }
    }

    async closeFile(filePath: string) {
        const server = this.getServerForFile(filePath);
        if (!server || server.state !== "running") return;

        const uri = `file://${path.resolve(filePath)}`;
        try {
            await server.sendNotification("textDocument/didClose", {
                textDocument: { uri }
            });
            this.fileMap.delete(uri);
            logger.info(`LSP: Sent didClose for ${filePath}`);
        } catch (err: any) {
            logger.error(new Error(`Failed to sync file close ${filePath}: ${err.message}`));
            throw err;
        }
    }

    isFileOpen(filePath: string) {
        const uri = `file://${path.resolve(filePath)}`;
        return this.fileMap.has(uri);
    }
}
