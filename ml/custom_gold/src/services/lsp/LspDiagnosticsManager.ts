
import { LspManager } from "./LspManager.js";
import { LRUCache } from "../../utils/structs/LRUCache.js";
import { randomUUID } from "node:crypto";
import { fileURLToPath } from "node:url";
import { log } from "../logger/loggerService.js";

const logger = log("lsp-diagnostics");

export interface DiagnosticRange {
    start: { line: number; character: number };
    end: { line: number; character: number };
}

export interface Diagnostic {
    message: string;
    severity: "Error" | "Warning" | "Info" | "Hint";
    range: DiagnosticRange;
    source?: string;
    code?: string;
}

export interface FileDiagnostics {
    uri: string;
    diagnostics: Diagnostic[];
}

export interface PendingDiagnostic {
    serverName: string;
    files: FileDiagnostics[];
    timestamp: number;
    attachmentSent: boolean;
}

const MAX_DIAGNOSTICS_PER_FILE = 10;
const MAX_TOTAL_DIAGNOSTICS = 30;
const DELIVERED_DIAGNOSTICS_CACHE_SIZE = 500;

/**
 * Manages aggregation, deduplication, and volume limiting of LSP diagnostics.
 * Based on chunk_372.ts
 */
export class LspDiagnosticsManager {
    private static instance: LspDiagnosticsManager;
    private pendingDiagnostics = new Map<string, PendingDiagnostic>(); // uuid -> pending
    private deliveredDiagnostics = new LRUCache<string, Set<string>>(DELIVERED_DIAGNOSTICS_CACHE_SIZE); // uri -> set of diagnostic signatures
    private lspManager?: LspManager;

    private constructor() { }

    static getInstance(): LspDiagnosticsManager {
        if (!LspDiagnosticsManager.instance) {
            LspDiagnosticsManager.instance = new LspDiagnosticsManager();
        }
        return LspDiagnosticsManager.instance;
    }

    setLspManager(lspManager: LspManager) {
        this.lspManager = lspManager;
        this.setupHandlers();
    }

    private setupHandlers() {
        if (!this.lspManager) return;

        const servers = this.lspManager.getAllServers();
        for (const [name, server] of Array.from(servers.entries())) {
            try {
                server.onNotification("textDocument/publishDiagnostics", (params: any) => {
                    this.handlePublishDiagnostics(name, params);
                });
                logger.info(`Registered diagnostics handler for ${name}`);
            } catch (err: any) {
                logger.error(new Error(`Failed to register diagnostics handler for ${name}: ${err.message}`));
            }
        }
    }

    private handlePublishDiagnostics(serverName: string, params: any) {
        logger.info(`[PASSIVE DIAGNOSTICS] Handler invoked for ${serverName}`);
        try {
            if (!params || typeof params !== "object" || !params.uri || !params.diagnostics) {
                logger.error(`Invalid diagnostic params from ${serverName}: ${JSON.stringify(params)}`);
                return;
            }

            logger.info(`Received diagnostics from ${serverName}: ${params.diagnostics.length} diagnostic(s) for ${params.uri}`);
            const normalized = this.normalizeDiagnostics(params);

            if (normalized.length === 0 || normalized[0].diagnostics.length === 0) {
                logger.info(`Skipping empty diagnostics from ${serverName} for ${params.uri}`);
                return;
            }

            this.registerDiagnostics(serverName, normalized);
            logger.info(`LSP Diagnostics: Registered ${normalized.length} diagnostic file(s) from ${serverName} for async delivery`);
        } catch (err: any) {
            logger.error(new Error(`Unexpected error processing diagnostics from ${serverName}: ${err.message}`));
        }
    }

    private normalizeDiagnostics(params: any): FileDiagnostics[] {
        let filePath: string;
        try {
            filePath = params.uri.startsWith("file://") ? fileURLToPath(params.uri) : params.uri;
        } catch (err: any) {
            logger.error(err);
            logger.error(`Failed to convert URI to file path: ${params.uri}. Error: ${err.message}. Using original URI as fallback.`);
            filePath = params.uri;
        }

        const diagnostics: Diagnostic[] = params.diagnostics.map((d: any) => ({
            message: d.message,
            severity: this.mapSeverity(d.severity),
            range: {
                start: { line: d.range.start.line, character: d.range.start.character },
                end: { line: d.range.end.line, character: d.range.end.character }
            },
            source: d.source,
            code: d.code !== undefined && d.code !== null ? String(d.code) : undefined
        }));

        return [{ uri: filePath, diagnostics }];
    }

    private mapSeverity(severity: number): "Error" | "Warning" | "Info" | "Hint" {
        switch (severity) {
            case 1: return "Error";
            case 2: return "Warning";
            case 3: return "Info";
            case 4: return "Hint";
            default: return "Error";
        }
    }

    private getSeverityWeight(severity: string): number {
        switch (severity) {
            case "Error": return 1;
            case "Warning": return 2;
            case "Info": return 3;
            case "Hint": return 4;
            default: return 4;
        }
    }

    private registerDiagnostics(serverName: string, files: FileDiagnostics[]) {
        const id = randomUUID();
        logger.info(`LSP Diagnostics: Registering ${files.length} diagnostic file(s) from ${serverName} (ID: ${id})`);
        this.pendingDiagnostics.set(id, {
            serverName,
            files,
            timestamp: Date.now(),
            attachmentSent: false
        });
    }

    /**
     * Called by the main loop to get diagnostic updates for the user.
     * Based on chunk_372.ts (d52)
     */
    processPendingDiagnostics(): any[] {
        logger.info(`LSP Diagnostics: Checking registry - ${this.pendingDiagnostics.size} pending`);

        const toProcess: PendingDiagnostic[] = [];
        const serverNames = new Set<string>();
        const allFiles: FileDiagnostics[] = [];

        for (const pd of Array.from(this.pendingDiagnostics.values())) {
            if (!pd.attachmentSent) {
                allFiles.push(...pd.files);
                serverNames.add(pd.serverName);
                toProcess.push(pd);
            }
        }

        if (allFiles.length === 0) return [];

        let deduplicated: FileDiagnostics[];
        try {
            deduplicated = this.deduplicateDiagnostics(allFiles);
        } catch (err: any) {
            logger.error(new Error(`Failed to deduplicate LSP diagnostics: ${err.message}`));
            deduplicated = allFiles;
        }

        for (const pd of toProcess) {
            pd.attachmentSent = true;
        }

        const totalInitial = allFiles.reduce((acc, f) => acc + f.diagnostics.length, 0);
        const totalDeduplicated = deduplicated.reduce((acc, f) => acc + f.diagnostics.length, 0);

        if (totalInitial > totalDeduplicated) {
            logger.info(`LSP Diagnostics: Deduplication removed ${totalInitial - totalDeduplicated} duplicate diagnostic(s)`);
        }

        let totalDeliveredCount = 0;
        let removedByVolumeLimit = 0;

        for (const file of deduplicated) {
            // Sort by severity
            file.diagnostics.sort((a, b) => this.getSeverityWeight(a.severity) - this.getSeverityWeight(b.severity));

            // Per-file limit
            if (file.diagnostics.length > MAX_DIAGNOSTICS_PER_FILE) {
                removedByVolumeLimit += file.diagnostics.length - MAX_DIAGNOSTICS_PER_FILE;
                file.diagnostics = file.diagnostics.slice(0, MAX_DIAGNOSTICS_PER_FILE);
            }

            // Total limit
            const remainingBudget = MAX_TOTAL_DIAGNOSTICS - totalDeliveredCount;
            if (file.diagnostics.length > remainingBudget) {
                removedByVolumeLimit += file.diagnostics.length - remainingBudget;
                file.diagnostics = file.diagnostics.slice(0, Math.max(0, remainingBudget));
            }
            totalDeliveredCount += file.diagnostics.length;
        }

        deduplicated = deduplicated.filter(f => f.diagnostics.length > 0);

        if (removedByVolumeLimit > 0) {
            logger.info(`LSP Diagnostics: Volume limiting removed ${removedByVolumeLimit} diagnostic(s) (max ${MAX_DIAGNOSTICS_PER_FILE}/file, ${MAX_TOTAL_DIAGNOSTICS} total)`);
        }

        // Track delivered
        for (const file of deduplicated) {
            let deliveredSet = this.deliveredDiagnostics.get(file.uri);
            if (!deliveredSet) {
                deliveredSet = new Set<string>();
                this.deliveredDiagnostics.set(file.uri, deliveredSet);
            }
            for (const d of file.diagnostics) {
                try {
                    deliveredSet.add(this.getDiagnosticSignature(d));
                } catch (err: any) {
                    logger.error(new Error(`Failed to track delivered diagnostic in ${file.uri}: ${err.message}`));
                }
            }
        }

        const finalCount = deduplicated.reduce((acc, f) => acc + f.diagnostics.length, 0);
        if (finalCount === 0) {
            logger.info("LSP Diagnostics: No new diagnostics to deliver (all filtered by deduplication)");
            return [];
        }

        logger.info(`LSP Diagnostics: Delivering ${deduplicated.length} file(s) with ${finalCount} diagnostic(s) from ${serverNames.size} server(s)`);
        return [{
            serverName: Array.from(serverNames).join(", "),
            files: deduplicated
        }];
    }

    private deduplicateDiagnostics(files: FileDiagnostics[]): FileDiagnostics[] {
        const fileMap = new Map<string, Set<string>>();
        const result: FileDiagnostics[] = [];

        for (const file of files) {
            if (!fileMap.has(file.uri)) {
                fileMap.set(file.uri, new Set<string>());
                result.push({ uri: file.uri, diagnostics: [] });
            }

            const currentFileSet = fileMap.get(file.uri)!;
            const resFile = result.find(f => f.uri === file.uri)!;
            const deliveredSet = this.deliveredDiagnostics.get(file.uri) || new Set<string>();

            for (const d of file.diagnostics) {
                try {
                    const signature = this.getDiagnosticSignature(d);
                    if (currentFileSet.has(signature) || deliveredSet.has(signature)) {
                        continue;
                    }
                    currentFileSet.add(signature);
                    resFile.diagnostics.push(d);
                } catch (err: any) {
                    logger.error(new Error(`Failed to deduplicate diagnostic in ${file.uri}: ${err.message}`));
                    resFile.diagnostics.push(d);
                }
            }
        }
        return result.filter(f => f.diagnostics.length > 0);
    }

    private getDiagnosticSignature(diag: Diagnostic): string {
        return JSON.stringify({
            message: diag.message,
            severity: diag.severity,
            range: diag.range,
            source: diag.source || null,
            code: diag.code || null
        });
    }

    clearPendingDiagnostics() {
        logger.info(`LSP Diagnostics: Clearing ${this.pendingDiagnostics.size} pending diagnostic(s)`);
        this.pendingDiagnostics.clear();
    }

    clearDeliveredDiagnostics(uri: string) {
        if (this.deliveredDiagnostics.has(uri)) {
            logger.info(`LSP Diagnostics: Clearing delivered diagnostics for ${uri}`);
            this.deliveredDiagnostics.delete(uri);
        }
    }
}
