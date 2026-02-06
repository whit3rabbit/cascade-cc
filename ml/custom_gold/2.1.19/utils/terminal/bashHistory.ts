/**
 * File: src/utils/terminal/bashHistory.ts
 * Role: Provides utilities for bash-style history autocomplete.
 *
 * Note: This uses Claude Code's prompt history (history.jsonl) to mirror
 * the gold reference behavior, and filters entries starting with "!".
 */

import { getProjectHistory } from '../../services/terminal/HistoryService.js';

const HISTORY_CACHE_MS = 60_000;
const MAX_BASH_HISTORY_ENTRIES = 50;

let cachedHistory: string[] | null = null;
let lastHistoryFetch = 0;

/**
 * Retrieves a list of bash-style commands from Claude Code prompt history.
 * Commands are extracted from entries that start with "!".
 */
export async function getBashHistory(): Promise<string[]> {
    const now = Date.now();
    if (cachedHistory && now - lastHistoryFetch < HISTORY_CACHE_MS) {
        return cachedHistory;
    }

    const entries = await getProjectHistory(process.cwd());
    const commands: string[] = [];
    const seen = new Set<string>();

    for (const entry of entries) {
        const display = (entry.display || "").trim();
        if (!display.startsWith('!')) continue;

        const command = display.slice(1).trim();
        if (!command || seen.has(command)) continue;

        seen.add(command);
        commands.push(command);

        if (commands.length >= MAX_BASH_HISTORY_ENTRIES) {
            break;
        }
    }

    cachedHistory = commands;
    lastHistoryFetch = now;
    return commands;
}

/**
 * Returns the first history completion that starts with the given prefix.
 */
export async function getBashHistoryCompletion(prefix: string): Promise<{ fullCommand: string; suffix: string } | null> {
    if (!prefix || prefix.length < 2 || !prefix.trim()) {
        return null;
    }

    const history = await getBashHistory();
    for (const cmd of history) {
        if (cmd.startsWith(prefix) && cmd !== prefix) {
            return {
                fullCommand: cmd,
                suffix: cmd.slice(prefix.length)
            };
        }
    }
    return null;
}
