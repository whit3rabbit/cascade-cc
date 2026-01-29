/**
 * File: src/utils/terminal/tmuxUtils.ts
 * Role: Utilities for managing tmux sessions.
 */

import { executeBashCommand } from "../shared/bashUtils.js";

/**
 * Checks if a tmux session exists.
 */
export async function hasTmuxSession(sessionName: string): Promise<boolean> {
    try {
        const res = await executeBashCommand(`tmux has-session -t "${sessionName}"`);
        return res.exitCode === 0;
    } catch {
        return false;
    }
}

/**
 * Ensures a tmux session exists, creating it if necessary.
 */
export async function ensureTmuxSession(sessionName: string): Promise<void> {
    if (await hasTmuxSession(sessionName)) return;

    const res = await executeBashCommand(`tmux new-session -d -s "${sessionName}"`);
    if (res.exitCode !== 0) {
        throw new Error(`Failed to create tmux session '${sessionName}': ${res.stderr || "Unknown error"}`);
    }
}
