
import { callMcpTool } from "../mcp/McpToolExecution.js";
import { figures } from "../../vendor/terminalFigures.js";

export class DiagnosticsError extends Error {}

function getConnectedIdeClient(connections: any): any | undefined {
    if (!Array.isArray(connections)) return undefined;
    return connections.find(
        (connection) => connection?.type === "connected" && connection?.name === "ide"
    );
}

export interface Diagnostic {
    message: string;
    severity: string;
    source?: string;
    code?: string | number;
    range: {
        start: { line: number; character: number };
        end: { line: number; character: number };
    };
    uri: string;
}

export class DiagnosticsManager {
    static instance: DiagnosticsManager;
    private baseline = new Map<string, Diagnostic[]>();
    private initialized = false;
    private mcpClient: any;
    private lastProcessedTimestamps = new Map<string, number>();
    private rightFileDiagnosticsState = new Map<string, Diagnostic[]>();

    static getInstance(): DiagnosticsManager {
        if (!DiagnosticsManager.instance) {
            DiagnosticsManager.instance = new DiagnosticsManager();
        }
        return DiagnosticsManager.instance;
    }

    initialize(client: any): void {
        if (this.initialized) return;
        this.mcpClient = client;
        this.initialized = true;
    }

    async shutdown(): Promise<void> {
        this.initialized = false;
        this.baseline.clear();
    }

    reset(): void {
        this.baseline.clear();
        this.rightFileDiagnosticsState.clear();
    }

    normalizeFileUri(uri: string): string {
        const prefixes = ["file://", "_claude_fs_right:", "_claude_fs_left:"];
        for (const prefix of prefixes) {
            if (uri.startsWith(prefix)) return uri.slice(prefix.length);
        }
        return uri;
    }

    async ensureFileOpened(path: string): Promise<void> {
        if (!this.initialized || !this.mcpClient || this.mcpClient.type !== "connected") return;
        try {
            await callMcpTool(this.mcpClient, "openFile", {
                filePath: path,
                preview: false,
                startText: "",
                endText: "",
                selectToEndOfLine: false,
                makeFrontmost: false
            });
        } catch (error) {
            console.warn(error);
        }
    }

    async beforeFileEdited(path: string): Promise<void> {
        if (!this.initialized || !this.mcpClient || this.mcpClient.type !== "connected") return;
        const timestamp = Date.now();
        try {
            const result = await callMcpTool(this.mcpClient, "getDiagnostics", { uri: `file://${path}` });
            const diagnosticsResult = this.parseDiagnosticResult(result)[0];

            if (diagnosticsResult) {
                if (path !== this.normalizeFileUri(diagnosticsResult.uri)) {
                    console.warn(
                        new DiagnosticsError(
                            `Diagnostics file path mismatch: expected ${path}, got ${diagnosticsResult.uri})`
                        )
                    );
                    return;
                }
                this.baseline.set(path, diagnosticsResult.diagnostics);
                this.lastProcessedTimestamps.set(path, timestamp);
            } else {
                this.baseline.set(path, []);
                this.lastProcessedTimestamps.set(path, timestamp);
            }
        } catch { }
    }

    async getNewDiagnostics(): Promise<{ uri: string; diagnostics: Diagnostic[] }[]> {
        if (!this.initialized || !this.mcpClient || this.mcpClient.type !== "connected") return [];
        let allDiagnostics: any[] = [];

        try {
            const result = await callMcpTool(this.mcpClient, "getDiagnostics", {});
            allDiagnostics = this.parseDiagnosticResult(result);
        } catch {
            return [];
        }

        const projectDiagnostics = allDiagnostics
            .filter(d => this.baseline.has(this.normalizeFileUri(d.uri)))
            .filter(d => d.uri.startsWith("file://"));

        const rightFileDiagnostics = new Map<string, any>();
        allDiagnostics
            .filter(d => this.baseline.has(this.normalizeFileUri(d.uri)))
            .filter(d => d.uri.startsWith("_claude_fs_right:"))
            .forEach(d => {
                rightFileDiagnostics.set(this.normalizeFileUri(d.uri), d);
            });

        const newDiagnostics: { uri: string; diagnostics: Diagnostic[] }[] = [];

        for (const diagObj of projectDiagnostics) {
            const normalizedUri = this.normalizeFileUri(diagObj.uri);
            const baselineDiags = this.baseline.get(normalizedUri) || [];
            const rightDiagsObj = rightFileDiagnostics.get(normalizedUri);

            let currentDiagsObj = diagObj;

            if (rightDiagsObj) {
                const lastRightDiags = this.rightFileDiagnosticsState.get(normalizedUri);
                if (!lastRightDiags || !this.areDiagnosticArraysEqual(lastRightDiags, rightDiagsObj.diagnostics)) {
                    currentDiagsObj = rightDiagsObj;
                }
                this.rightFileDiagnosticsState.set(normalizedUri, rightDiagsObj.diagnostics);
            }

            const diffDiags = currentDiagsObj.diagnostics.filter((d: Diagnostic) =>
                !baselineDiags.some((b: Diagnostic) => this.areDiagnosticsEqual(d, b))
            );

            if (diffDiags.length > 0) {
                newDiagnostics.push({
                    uri: diagObj.uri,
                    diagnostics: diffDiags
                });
            }
            this.baseline.set(normalizedUri, currentDiagsObj.diagnostics);
        }
        return newDiagnostics;
    }

    parseDiagnosticResult(result: any): any[] {
        if (Array.isArray(result)) {
            const textPart = result.find(r => r.type === "text");
            if (textPart && "text" in textPart) {
                return JSON.parse(textPart.text);
            }
        }
        return [];
    }

    areDiagnosticsEqual(a: Diagnostic, b: Diagnostic): boolean {
        return a.message === b.message &&
            a.severity === b.severity &&
            a.source === b.source &&
            a.code === b.code &&
            a.range.start.line === b.range.start.line &&
            a.range.start.character === b.range.start.character &&
            a.range.end.line === b.range.end.line &&
            a.range.end.character === b.range.end.character;
    }

    areDiagnosticArraysEqual(a: Diagnostic[], b: Diagnostic[]): boolean {
        if (a.length !== b.length) return false;
        return a.every(d1 => b.some(d2 => this.areDiagnosticsEqual(d1, d2))) &&
            b.every(d1 => a.some(d2 => this.areDiagnosticsEqual(d1, d2)));
    }

    isLinterDiagnostic(diagnostic: Diagnostic): boolean {
        const linterSources = ["eslint", "eslint-plugin", "tslint", "prettier", "stylelint", "jshint", "standardjs", "xo", "rome", "biome", "deno-lint", "rubocop", "pylint", "flake8", "black", "ruff", "clippy", "rustfmt", "golangci-lint", "gofmt", "swiftlint", "detekt", "ktlint", "checkstyle", "pmd", "sonarqube", "sonarjs"];
        if (!diagnostic.source) return false;
        const source = diagnostic.source.toLowerCase();
        return linterSources.some(s => source.includes(s));
    }

    async handleQueryStart(connections: any[]): Promise<void> {
        if (!this.initialized) {
            const client = getConnectedIdeClient(connections);
            if (client) this.initialize(client);
        } else {
            this.reset();
        }
    }

    static formatDiagnosticsSummary(diagnosticsGroups: { uri: string; diagnostics: Diagnostic[] }[]): string {
        return diagnosticsGroups.map(group => {
            const filename = group.uri.split("/").pop() || group.uri;
            const messages = group.diagnostics.map(d => {
                return `  ${DiagnosticsManager.getSeveritySymbol(d.severity)} [Line ${d.range.start.line + 1}:${d.range.start.character + 1}] ${d.message}${d.code ? ` [${d.code}]` : ""}${d.source ? ` (${d.source})` : ""}`;
            }).join("\n");
            return `${filename}:\n${messages}`;
        }).join("\n\n");
    }

    static getSeveritySymbol(severity: string): string {
        const symbols: Record<string, string> = {
            Error: figures.cross,
            Warning: figures.warning,
            Info: figures.info,
            Hint: figures.star
        };
        return symbols[severity] || "•";
    }
}
