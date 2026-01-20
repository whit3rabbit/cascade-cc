import * as fs from "fs";
import * as path from "path";
import os from "os";
import { getSessionId } from "../services/session/sessionStore.js";

let debugWriter: { write: (msg: string) => void; flush: () => void; dispose: () => void } | null = null;
let profilePath = "";

function createBufferedStatsWriter({ writeFn, flushIntervalMs = 1000, maxBufferSize = 100, immediateMode = false }: any) {
    let buffer: string[] = [];
    let timeout: NodeJS.Timeout | null = null;

    function clear() {
        if (timeout) {
            clearTimeout(timeout);
            timeout = null;
        }
    }

    function flush() {
        if (buffer.length === 0) return;
        writeFn(buffer.join(""));
        buffer = [];
        clear();
    }

    function schedule() {
        if (!timeout) {
            timeout = setTimeout(flush, flushIntervalMs);
        }
    }

    return {
        write(msg: string) {
            if (immediateMode) {
                writeFn(msg);
                return;
            }
            buffer.push(msg);
            schedule();
            if (buffer.length >= maxBufferSize) {
                flush();
            }
        },
        flush,
        dispose() {
            flush();
        }
    };
}

export function isDebugEnabled(): boolean {
    return process.env.DEBUG !== undefined || process.argv.includes("--debug") || process.argv.includes("-d");
}

export function getDebugLogDir(): string {
    return process.env.CLAUDE_CODE_DEBUG_LOGS_DIR ?? path.join(os.tmpdir(), "claude-code-debug");
}

function getDebugLogPath(): string {
    return path.join(getDebugLogDir(), `${getSessionId()}.txt`);
}

function ensureDebugWriter() {
    if (!debugWriter) {
        const logPath = getDebugLogPath();
        const logDir = path.dirname(logPath);

        try {
            if (!fs.existsSync(logDir)) {
                fs.mkdirSync(logDir, { recursive: true });
            }
        } catch (e) { }

        debugWriter = createBufferedStatsWriter({
            writeFn: (msg: string) => {
                try {
                    fs.appendFileSync(logPath, msg);
                    // Also maintain 'latest' symlink if possible
                    try {
                        const latestLink = path.join(logDir, "latest");
                        if (fs.existsSync(latestLink)) {
                            fs.unlinkSync(latestLink);
                        }
                        fs.symlinkSync(logPath, latestLink);
                    } catch { }
                } catch { }
            },
            immediateMode: false // Could be dynamic
        });

        // Ensure cleanup
        process.on("exit", () => debugWriter?.dispose());
    }
    return debugWriter;
}

export function debugLog(msg: string, level: string = "debug") {
    if (!isDebugEnabled()) return;

    if (typeof msg !== "string") {
        msg = JSON.stringify(msg);
    }

    const timestamp = new Date().toISOString();
    const formatted = `${timestamp} [${level.toUpperCase()}] ${msg.trim()}\n`;

    if (process.argv.includes("--debug-to-stderr")) {
        process.stderr.write(formatted);
        return;
    }

    ensureDebugWriter()?.write(formatted);
}
