import { existsSync, mkdirSync, readFileSync, readdirSync, unlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { randomUUID } from "node:crypto";
import { getTelemetryDir } from "./telemetryEvents.js";
import { getCliUserAgent } from "../claude/anthropicApiClient.js";
import { isOauthActive } from "../auth/oauthManager.js";
import { getAuthHeaders } from "../claude/anthropicApiClient.js";

const BATCH_FILE_PREFIX = "1p_failed_events.";
const SESSION_UUID = randomUUID();

export enum ExportResultCode {
    SUCCESS = 0,
    FAILED = 1
}

/**
 * Class for batching and exporting telemetry events.
 * Deobfuscated from yp1 in chunk_188.ts.
 */
export class TelemetryExporter {
    private endpoint: string;
    private timeout: number;
    private maxQueuedEvents: number;
    private maxBatchSize: number;
    private batchDelayMs: number;
    private baseBackoffDelayMs: number;
    private maxBackoffDelayMs: number;

    private pendingExports: Promise<any>[] = [];
    private isShutdown = false;
    private backoffRetryTimer: NodeJS.Timeout | null = null;
    private backoffAttempt = 0;
    private isRetrying = false;

    constructor(options: any = {}) {
        const baseUrl = process.env.ANTHROPIC_BASE_URL || "https://api.anthropic.com";
        this.endpoint = `${baseUrl}/api/event_logging/batch`;
        this.timeout = options.timeout || 10000;
        this.maxQueuedEvents = options.maxQueuedEvents || 8192;
        this.maxBatchSize = options.maxBatchSize || 200;
        this.batchDelayMs = options.batchDelayMs || 100;
        this.baseBackoffDelayMs = options.baseBackoffDelayMs || 500;
        this.maxBackoffDelayMs = options.maxBackoffDelayMs || 30000;

        this.retryPreviousBatches();
    }

    private getCurrentBatchFilePath(): string {
        const dateStr = new Date().toISOString().split("T")[0];
        return join(getTelemetryDir(), `${BATCH_FILE_PREFIX}${dateStr}.${SESSION_UUID}.json`);
    }

    private loadEventsFromFile(path: string): any[] {
        try {
            if (!existsSync(path)) return [];
            return JSON.parse(readFileSync(path, "utf8"));
        } catch {
            return [];
        }
    }

    private saveEventsToFile(path: string, events: any[]): void {
        try {
            if (events.length === 0) {
                if (existsSync(path)) unlinkSync(path);
            } else {
                const dir = getTelemetryDir();
                if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
                writeFileSync(path, JSON.stringify(events), "utf8");
            }
        } catch (err) {
            console.error("Failed to save telemetry events to file", err);
        }
    }

    private retryPreviousBatches(): void {
        try {
            const dir = getTelemetryDir();
            if (!existsSync(dir)) return;

            const dateStr = new Date().toISOString().split("T")[0];
            const prefix = `${BATCH_FILE_PREFIX}${dateStr}.`;

            const files = readdirSync(dir)
                .filter(f => f.startsWith(prefix) && f.endsWith(".json") && !f.includes(SESSION_UUID));

            for (const file of files) {
                this.retryFileInBackground(join(dir, file));
            }
        } catch (err) {
            console.error(err);
        }
    }

    private async retryFileInBackground(path: string): Promise<void> {
        const events = this.loadEventsFromFile(path);
        if (events.length === 0) {
            if (existsSync(path)) unlinkSync(path);
            return;
        }
        const failed = await this.sendEventsInBatches(events);
        if (failed.length === 0) {
            if (existsSync(path)) unlinkSync(path);
        }
    }

    async export(logs: any[], callback: (result: any) => void): Promise<void> {
        if (this.isShutdown) {
            callback({ code: ExportResultCode.FAILED, error: new Error("Exporter shutdown") });
            return;
        }

        const exportPromise = this.doExport(logs, callback);
        this.pendingExports.push(exportPromise);
        exportPromise.finally(() => {
            const idx = this.pendingExports.indexOf(exportPromise);
            if (idx > -1) this.pendingExports.splice(idx, 1);
        });
    }

    private async doExport(logs: any[], callback: (result: any) => void): Promise<void> {
        try {
            const events = logs.map(l => l.body); // Simplified transformation
            if (events.length === 0) {
                callback({ code: ExportResultCode.SUCCESS });
                return;
            }

            const failed = await this.sendEventsInBatches(events);
            if (failed.length > 0) {
                this.queueFailedEvents(failed);
                this.scheduleBackoffRetry();
                callback({ code: ExportResultCode.FAILED, error: new Error(`Failed to export ${failed.length} events`) });
                return;
            }

            this.resetBackoff();
            callback({ code: ExportResultCode.SUCCESS });
        } catch (err) {
            callback({ code: ExportResultCode.FAILED, error: err });
        }
    }

    private async sendEventsInBatches(events: any[]): Promise<any[]> {
        const failed: any[] = [];
        for (let i = 0; i < events.length; i += this.maxBatchSize) {
            const batch = events.slice(i, i + this.maxBatchSize);
            try {
                await this.postBatch(batch);
            } catch {
                failed.push(...batch);
            }
            if (i + this.maxBatchSize < events.length && this.batchDelayMs > 0) {
                await new Promise(r => setTimeout(r, this.batchDelayMs));
            }
        }
        return failed;
    }

    private async postBatch(events: any[]): Promise<void> {
        const headers: Record<string, string> = {
            "Content-Type": "application/json",
            "User-Agent": getCliUserAgent(),
            "x-service-name": "claude-code"
        };

        // simplified auth check
        const auth = getAuthHeaders();
        if (!auth.error) {
            Object.assign(headers, auth.headers);
        }

        const response = await fetch(this.endpoint, {
            method: "POST",
            body: JSON.stringify({ events }),
            headers,
            signal: AbortSignal.timeout(this.timeout)
        });

        if (!response.ok) throw new Error(`HTTP ${response.status}`);
    }

    private queueFailedEvents(events: any[]): void {
        const path = this.getCurrentBatchFilePath();
        const existing = this.loadEventsFromFile(path);
        const combined = [...existing, ...events].slice(-this.maxQueuedEvents);
        this.saveEventsToFile(path, combined);
    }

    private scheduleBackoffRetry(): void {
        if (this.backoffRetryTimer || this.isRetrying || this.isShutdown) return;
        const delay = Math.min(this.baseBackoffDelayMs * Math.pow(2, this.backoffAttempt), this.maxBackoffDelayMs);
        this.backoffRetryTimer = setTimeout(() => {
            this.backoffRetryTimer = null;
            this.retryFailedEvents();
        }, delay);
    }

    private async retryFailedEvents(): Promise<void> {
        const path = this.getCurrentBatchFilePath();
        while (!this.isShutdown) {
            const events = this.loadEventsFromFile(path);
            if (events.length === 0) break;

            this.isRetrying = true;
            this.backoffAttempt++;
            if (existsSync(path)) unlinkSync(path);

            const failed = await this.sendEventsInBatches(events);
            this.isRetrying = false;

            if (failed.length > 0) {
                this.saveEventsToFile(path, failed);
                this.scheduleBackoffRetry();
                return;
            }
            this.resetBackoff();
        }
    }

    private resetBackoff(): void {
        this.backoffAttempt = 0;
        if (this.backoffRetryTimer) {
            clearTimeout(this.backoffRetryTimer);
            this.backoffRetryTimer = null;
        }
    }

    async shutdown(): Promise<void> {
        this.isShutdown = true;
        this.resetBackoff();
        await Promise.all(this.pendingExports);
    }
}
