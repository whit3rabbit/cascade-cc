/**
 * File: src/services/logging/LogManager.ts
 * Role: Provides buffered file writing for logs (JSONL format) and error reporting.
 */

import { appendFileSync, mkdirSync, existsSync, readdirSync, unlinkSync, statSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { getBaseConfigDir } from "../../utils/shared/runtimeAndEnv.js";

/**
 * Interface for a buffered file writer.
 */
export interface BufferedFileWriter {
    write(data: string): void;
    flush(): void;
    dispose(): void;
}

/**
 * Options for creating a buffered file writer.
 */
export interface BufferedFileWriterOptions {
    filePath: string;
    flushIntervalMs?: number;
    maxBufferSize?: number; // Kept for backward compat but unused if not specified
    maxBufferLength?: number; // Size in bytes
}

/**
 * Formats a date for session-based logging (YYYY-MM-DD_HH-mm-ss).
 */
export function formatSessionTimestamp(date: Date): string {
    const pad = (n: number) => n.toString().padStart(2, '0');
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}_${pad(date.getHours())}-${pad(date.getMinutes())}-${pad(date.getSeconds())}`;
}

let mainLogWriter: BufferedFileWriter | null = null;
let currentSessionLogFile: string | null = null;

/**
 * Creates a buffered file writer to reduce disk I/O.
 * Batches log entries and flushes them every 5 seconds or when the buffer exceeds 10KB.
 */
export function createBufferedFileWriter(options: BufferedFileWriterOptions): BufferedFileWriter {
    const {
        filePath,
        flushIntervalMs = 5000,
        maxBufferLength = 10 * 1024 // 10KB
    } = options;

    let buffer: string[] = [];
    let currentBufferLength = 0;
    let timeout: NodeJS.Timeout | null = null;

    function flush() {
        if (timeout) {
            clearTimeout(timeout);
            timeout = null;
        }
        if (buffer.length === 0) return;

        try {
            const dir = dirname(filePath);
            if (!existsSync(dir)) {
                mkdirSync(dir, { recursive: true });
            }
            appendFileSync(filePath, buffer.join(""));
            buffer = [];
            currentBufferLength = 0;
        } catch {
            // Silently fail if logging fails to avoid crashing the app
        }
    }

    function scheduleFlush() {
        if (!timeout) {
            timeout = setTimeout(flush, flushIntervalMs);
        }
    }

    return {
        write(data: string) {
            buffer.push(data);
            currentBufferLength += Buffer.byteLength(data, 'utf8');

            if (currentBufferLength >= maxBufferLength) {
                flush();
            } else {
                scheduleFlush();
            }
        },
        flush,
        dispose() {
            flush();
        }
    };
}

/**
 * Service for writing logs to disk with optional buffering.
 */
export const LogManager = {
    /**
     * Appends a log entry to a file, creating directories if needed.
     * This version is for backward compatibility or direct file logging.
     */
    logToFile(filePath: string, data: Record<string, any>): void {
        try {
            const entry = JSON.stringify({
                ...data,
                timestamp: new Date().toISOString()
            }) + "\n";

            appendFileSync(filePath, entry);
        } catch {
            try {
                mkdirSync(dirname(filePath), { recursive: true });
                appendFileSync(filePath, JSON.stringify(data) + "\n");
            } catch {
                // Silently fail if logging fails
            }
        }
    },

    /**
     * Specialized logger for MCP server output/errors.
     */
    logMcp(serverName: string, message: string, type: "info" | "error" = "info"): void {
        console.log(`[MCP:${serverName}] ${type.toUpperCase()}: ${message}`);
    },

    /**
     * Logs to the current session log file using the buffered writer.
     */
    logSession(data: any): void {
        if (!mainLogWriter) {
            const logDir = join(getBaseConfigDir(), "logs");
            this.logToFile(join(logDir, "early_boot.log"), data);
            return;
        }

        const logEntry = JSON.stringify({
            timestamp: new Date().toISOString(),
            ...data
        }) + "\n";

        mainLogWriter.write(logEntry);
    }
};

/**
 * Initializes the logging system, setting up directories and log rotation.
 */
export async function initializeLogging(): Promise<void> {
    const logDir = join(getBaseConfigDir(), "logs");
    if (!existsSync(logDir)) {
        mkdirSync(logDir, { recursive: true });
    }

    // Generate session-based log file
    const timestamp = formatSessionTimestamp(new Date());
    currentSessionLogFile = join(logDir, `session_${timestamp}.log`);

    // Initialize main log writer
    mainLogWriter = createBufferedFileWriter({
        filePath: currentSessionLogFile,
        flushIntervalMs: 5000,
        maxBufferLength: 10 * 1024 // 10KB
    });

    // Log rotation: Keep last 10 session logs to prevent bloating
    try {
        const files = readdirSync(logDir)
            .filter(f => f.startsWith("session_") && f.endsWith(".log"))
            .map(f => ({ name: f, path: join(logDir, f), mtime: statSync(join(logDir, f)).mtimeMs }))
            .sort((a, b) => b.mtime - a.mtime);

        if (files.length > 10) {
            files.slice(10).forEach(f => {
                try {
                    unlinkSync(f.path);
                } catch {
                    // Ignore errors during cleanup
                }
            });
        }
    } catch {
        // Log rotation failure should not block startup
    }
}
