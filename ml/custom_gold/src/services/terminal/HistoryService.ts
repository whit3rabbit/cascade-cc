
import { readHistoryStream, addHistoryEntry } from './promptHistory.js';

export interface HistoryEntry {
    display: string;
    timestamp: number;
    project: string;
    sessionId: string;
    pastedContents?: Record<number, any>;
}

/**
 * Fetches command history for a specific project.
 * Deobfuscated from BFB in chunk_216.ts.
 */
export async function getProjectHistory(projectPath: string, limit: number = 100): Promise<HistoryEntry[]> {
    const entries: HistoryEntry[] = [];
    for await (const entry of readHistoryStream()) {
        if (entry.project === projectPath) {
            entries.push(entry);
            if (entries.length >= limit) break;
        }
    }
    return entries;
}

/**
 * Saves a new entry to command history.
 * Deobfuscated from I0A in chunk_216.ts.
 */
export function saveToHistory(text: string) {
    addHistoryEntry(text);
}
