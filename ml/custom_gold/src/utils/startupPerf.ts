import { performance } from "perf_hooks";
import * as fs from "fs";
import * as path from "path";
import os from "os";
import { getSessionId } from "../services/session/sessionStore.js";
import { debugLog } from "./debugLog.js";

const memorySnapshots = new Map<string, NodeJS.MemoryUsage>();
let profilingEnabled = process.env.CLAUDE_CODE_PROFILE_STARTUP === "1";

export function markStartupCheckpoint(name: string) {
    if (!profilingEnabled) return;
    performance.mark(name);
    memorySnapshots.set(name, process.memoryUsage());
}

function formatMb(bytes: number) {
    return (bytes / 1024 / 1024).toFixed(2);
}

function formatDuration(ms: number) {
    return ms.toFixed(3);
}

export function generateStartupReport(): string {
    if (!profilingEnabled) return "Startup profiling not enabled";

    const marks = performance.getEntriesByType("mark");
    if (marks.length === 0) return "No profiling checkpoints recorded";

    const lines: string[] = [];
    lines.push("=".repeat(80));
    lines.push("STARTUP PROFILING REPORT");
    lines.push("=".repeat(80));
    lines.push("");

    let previousTime = 0;

    for (const mark of marks) {
        const timestamp = formatDuration(mark.startTime);
        const diff = formatDuration(mark.startTime - previousTime);
        const mem = memorySnapshots.get(mark.name);
        const memStr = mem ? ` | RSS: ${formatMb(mem.rss)}MB, Heap: ${formatMb(mem.heapUsed)}MB` : "";

        lines.push(`[+${timestamp.padStart(8)}ms] (+${diff.padStart(7)}ms) ${mark.name}${memStr}`);
        previousTime = mark.startTime;
    }

    const lastMark = marks[marks.length - 1];
    const totalTime = formatDuration(lastMark?.startTime ?? 0);

    lines.push("");
    lines.push(`Total startup time: ${totalTime}ms`);
    lines.push("=".repeat(80));

    return lines.join("\n");
}

export function saveStartupReport() {
    if (!profilingEnabled) return;

    try {
        const report = generateStartupReport();
        const reportDir = path.join(os.tmpdir(), "claude-code-profile");
        const reportPath = path.join(reportDir, `${getSessionId()}.txt`);

        if (!fs.existsSync(reportDir)) {
            fs.mkdirSync(reportDir, { recursive: true });
        }

        fs.writeFileSync(reportPath, report, { encoding: "utf8" });
        debugLog("Startup profiling report saved to " + reportPath);
        debugLog(report);
    } catch (e) {
        // Ignore errors during reporting
    }
}
