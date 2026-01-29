/**
 * File: src/services/terminal/AppleTerminalService.ts
 * Role: Manages backup and restoration of Apple Terminal (com.apple.Terminal) settings.
 */

import { homedir } from "os";
import { join } from "path";
import * as fs from 'fs';
import { runCommand } from '../../utils/process/commandRunner.js';

const PLIST_PATH = join(homedir(), "Library", "Preferences", "com.apple.Terminal.plist");
const BACKUP_PATH = `${PLIST_PATH}.bak`;

// Global state stubs (should be integrated with a proper Store)
let setupInProgress = false;
let backupPath: string | null = null;

export interface TerminalSetupState {
    inProgress: boolean;
    backupPath: string | null;
}

export function getAppleTerminalSetupState(): TerminalSetupState {
    return {
        inProgress: setupInProgress,
        backupPath: backupPath
    };
}

export function clearAppleTerminalSetupInProgress(): void {
    setupInProgress = false;
    backupPath = null;
}

/**
 * Backs up the current Apple Terminal settings using the 'defaults' command.
 */
export async function backupAppleTerminalSettings(): Promise<string | null> {
    try {
        const { code } = await runCommand("defaults", ["export", "com.apple.Terminal", PLIST_PATH]);
        if (code !== 0) return null;

        if (fs.existsSync(PLIST_PATH)) {
            await runCommand("defaults", ["export", "com.apple.Terminal", BACKUP_PATH]);
            setupInProgress = true;
            backupPath = BACKUP_PATH;
            return BACKUP_PATH;
        }
        return null;
    } catch (error) {
        console.error("Failed to backup Apple Terminal settings:", error);
        return null;
    }
}

export interface RestoreResult {
    status: "restored" | "failed" | "no_backup";
    backupPath?: string;
}

/**
 * Restores Apple Terminal settings from backup.
 */
export async function restoreAppleTerminalSettings(): Promise<RestoreResult> {
    const { inProgress, backupPath: path } = getAppleTerminalSetupState();

    if (!inProgress || !path || !fs.existsSync(path)) {
        clearAppleTerminalSetupInProgress();
        return { status: "no_backup" };
    }

    try {
        const { code } = await runCommand("defaults", ["import", "com.apple.Terminal", path]);
        if (code !== 0) return { status: "failed", backupPath: path };

        // Kill cfprefsd to force reload of defaults
        await runCommand("killall", ["cfprefsd"]);
        clearAppleTerminalSetupInProgress();
        return { status: "restored" };
    } catch (error) {
        console.error("Failed to restore Apple Terminal settings:", error);
        clearAppleTerminalSetupInProgress();
        return { status: "failed", backupPath: path };
    }
}
