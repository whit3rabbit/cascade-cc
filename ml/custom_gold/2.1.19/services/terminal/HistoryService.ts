/**
 * File: src/services/terminal/HistoryService.ts
 * Role: Manages project-specific command history persistence and retrieval.
 */

import { join } from 'node:path';
import { readFileSync, appendFileSync, existsSync } from 'node:fs';
import { getBaseConfigDir } from '../../utils/shared/runtimeAndEnv.js';

const HISTORY_FILE = 'history.jsonl';
const MAX_ENTRIES_PER_PROJECT = 100;

export interface HistoryEntry {
    display: string;
    project: string;
    timestamp?: number;
    [key: string]: any;
}

/**
 * Retrieves the command history for a specific project.
 * 
 * @param {string} projectPath - The absolute path to the project.
 * @returns {Promise<Array>} A list of history entries.
 */
export async function getProjectHistory(projectPath: string): Promise<HistoryEntry[]> {
    const historyPath = join(getBaseConfigDir(), HISTORY_FILE);
    if (!existsSync(historyPath)) return [];

    try {
        const content = readFileSync(historyPath, 'utf8');
        const lines = content.split('\n').filter(Boolean);

        const entries = lines
            .map(line => {
                try {
                    return JSON.parse(line);
                } catch {
                    return null;
                }
            })
            .filter((entry): entry is HistoryEntry => entry && entry.project === projectPath);

        return entries.slice(-MAX_ENTRIES_PER_PROJECT).reverse();
    } catch (error) {
        console.error("Failed to read history:", error);
        return [];
    }
}

import { EnvService } from '../config/EnvService.js';

/**
 * Adds a new entry to the command history.
 * 
 * @param {Object} entry - The history entry to add.
 * @param {string} entry.display - The command text.
 * @param {string} entry.project - The project path.
 */
export async function addToPromptHistory(entry: HistoryEntry): Promise<void> {
    if (EnvService.isTruthy("CLAUDE_CODE_SKIP_PROMPT_HISTORY")) return;

    const historyPath = join(getBaseConfigDir(), HISTORY_FILE);
    const data: HistoryEntry = {
        ...entry,
        timestamp: Date.now()
    };

    try {
        appendFileSync(historyPath, JSON.stringify(data) + '\n', { mode: 0o600 });
    } catch (error) {
        console.error("Failed to write to history:", error);
    }
}
