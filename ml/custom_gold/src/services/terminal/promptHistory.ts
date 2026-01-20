import * as fs from "node:fs";
import * as path from "node:path";
import * as readline from "node:readline";
// @ts-ignore
import lockfile from "proper-lockfile";
import { getConfigDir } from "../../utils/settings/runtimeSettingsAndAuth.js";

/**
 * Persistent prompt history implementation.
 * Deobfuscated from IA1, GFB, and related in chunk_216.ts.
 */

const HISTORY_LIMIT = 100;
const MAX_PASTED_CONTENT_SIZE = 1024;
const HISTORY_FILE = "history.jsonl";

let memoryHistory: any[] = [];
let isWriting = false;
let pendingSave: Promise<void> | null = null;
let saveTimer: NodeJS.Timeout | null = null;

function getHistoryPath(): string {
    return path.join(getConfigDir(), HISTORY_FILE);
}

/**
 * Parses [Pasted text #N] markers from a prompt.
 */
export function extractPastedRefs(text: string): Array<{ id: number; match: string }> {
    const regex = /\[(Pasted text|Image|\.\.\.Truncated text) #(\d+)(?: \+\d+ lines)?(\.)*\]/g;
    return Array.from(text.matchAll(regex)).map(match => ({
        id: parseInt(match[2] || "0", 10),
        match: match[0]
    })).filter(ref => ref.id > 0);
}

/**
 * Async generator to read history from disk (newest first).
 */
export async function* readHistoryStream(): AsyncGenerator<any> {
    // Yield memory history first
    for (let i = memoryHistory.length - 1; i >= 0; i--) {
        yield memoryHistory[i];
    }

    const historyPath = getHistoryPath();
    if (!fs.existsSync(historyPath)) return;

    const fileStream = fs.createReadStream(historyPath);
    const rl = readline.createInterface({
        input: fileStream,
        crlfDelay: Infinity
    });

    // Since it's JSONL, we'd ideally read backwards, 
    // but for simplicity we collect and reverse or just read forwards.
    // The obfuscated logic used a reverse loop over lines.
    const lines: string[] = [];
    for await (const line of rl) {
        lines.push(line);
    }

    for (let i = lines.length - 1; i >= 0; i--) {
        try {
            yield JSON.parse(lines[i]);
        } catch (e) {
            // Ignore corrupted lines
        }
    }
}

/**
 * Flushes memory history to disk.
 */
async function flushHistory() {
    if (memoryHistory.length === 0) return;

    const historyPath = getHistoryPath();
    let release: (() => Promise<void>) | null = null;

    try {
        if (!fs.existsSync(historyPath)) {
            fs.writeFileSync(historyPath, "", { mode: 0o600 });
        }

        release = await lockfile.lock(historyPath, {
            stale: 10000,
            retries: { retries: 3, minTimeout: 50 }
        });

        const lines = memoryHistory.map(entry => JSON.stringify(entry) + "\n");
        memoryHistory = [];
        fs.appendFileSync(historyPath, lines.join(""), { mode: 0o600 });

    } catch (e) {
        console.warn(`Failed to write prompt history: ${e}`);
    } finally {
        if (release) await release();
    }
}

/**
 * Queues a save operation.
 */
async function scheduleSave() {
    if (isWriting || memoryHistory.length === 0) return;
    isWriting = true;
    try {
        await flushHistory();
    } finally {
        isWriting = false;
        if (memoryHistory.length > 0) {
            // If new entries arrived during save, schedule another
            setTimeout(() => scheduleSave(), 500);
        }
    }
}

/**
 * Adds a new entry to history and triggers background save.
 */
export function addHistoryEntry(entry: string | any) {
    if (process.env.CLAUDE_CODE_SKIP_PROMPT_HISTORY === "true") return;

    const data = typeof entry === "string" ? { display: entry, pastedContents: {} } : entry;

    // Filter out large pasted contents before saving
    const filteredPasted: Record<number, any> = {};
    if (data.pastedContents) {
        for (const [id, content] of Object.entries(data.pastedContents)) {
            const c = content as any;
            if (c.type !== "image" && (c.content?.length || 0) <= MAX_PASTED_CONTENT_SIZE) {
                filteredPasted[Number(id)] = c;
            }
        }
    }

    const historyEntry = {
        ...data,
        pastedContents: filteredPasted,
        timestamp: Date.now(),
        project: process.cwd(), // Simplified for now
        sessionId: "current-session" // Simplified for now
    };

    memoryHistory.push(historyEntry);
    pendingSave = scheduleSave();
}
