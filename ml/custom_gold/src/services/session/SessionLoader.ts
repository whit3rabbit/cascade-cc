import * as fs from "node:fs";
import * as path from "node:path";
import { getProjectRoot } from "../../utils/shared/pathUtils.js";

export interface SessionData {
    messages: any[];
    fileHistorySnapshots: any[];
    sessionId: string;
}

export function loadSession(sessionId?: string): SessionData | null {
    const root = getProjectRoot();
    const sessionsDir = path.join(root, ".claude", "sessions");

    if (!fs.existsSync(sessionsDir)) return null;

    let targetFile: string | null = null;

    if (sessionId) {
        const filePath = path.join(sessionsDir, `${sessionId}.jsonl`);
        if (fs.existsSync(filePath)) {
            targetFile = filePath;
        }
    } else {
        // Find the most recent session
        const files = fs.readdirSync(sessionsDir)
            .filter(f => f.endsWith(".jsonl"))
            .map(f => ({ name: f, time: fs.statSync(path.join(sessionsDir, f)).mtime.getTime() }))
            .sort((a, b) => b.time - a.time);

        if (files.length > 0) {
            targetFile = path.join(sessionsDir, files[0].name);
            sessionId = files[0].name.replace(".jsonl", "");
        }
    }

    if (!targetFile || !sessionId) return null;

    try {
        const content = fs.readFileSync(targetFile, "utf-8");
        const lines = content.split("\n").filter(l => l.trim().length > 0);
        const messages: any[] = [];
        const snapshots: any[] = [];

        for (const line of lines) {
            try {
                const entry = JSON.parse(line);
                if (entry.type === "user" || entry.type === "assistant" || entry.type === "metadata" || entry.type === "attachment" || entry.type === "tool_result") {
                    messages.push(entry);
                }
                if (entry.type === "snapshot") {
                    snapshots.push(entry);
                }
            } catch {
                // Ignore corrupt lines
            }
        }

        return {
            sessionId,
            messages,
            fileHistorySnapshots: snapshots
        };
    } catch (err) {
        console.error(`Failed to load session ${sessionId}:`, err);
        return null;
    }
}
