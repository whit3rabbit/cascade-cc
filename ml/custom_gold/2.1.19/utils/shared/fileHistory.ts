/**
 * File: src/utils/shared/fileHistory.ts
 * Role: Manages file checkpoints and history tracking for undo/rewind functionality.
 */

import { createHash } from "node:crypto";
import { join, dirname, isAbsolute } from "node:path";
import fs, { chmodSync, existsSync, mkdirSync, readFileSync, statSync, unlinkSync, writeFileSync } from "node:fs";
import { getBaseConfigDir } from "./runtimeAndEnv.js";

// --- Types ---

export interface BackupInfo {
    backupFileName: string | null; // null if file was deleted
    version: number;
    backupTime: Date;
}

export interface Snapshot {
    messageId: string;
    trackedFileBackups: Record<string, BackupInfo>;
    timestamp: Date;
}

export interface HistoryState {
    snapshots: Snapshot[];
    trackedFiles: Set<string>;
    trackedFilePaths: Record<string, string>; // Maps fileKey to absolutePath
}

export interface DiffStats {
    filesChanged: string[];
    insertions: number;
    deletions: number;
}

// --- Constants & Helpers ---

import { EnvService } from "../../services/config/EnvService.js";

/**
 * Checks if file checkpointing is enabled via settings or environment.
 */
export function isFileCheckpointingEnabled(): boolean {
    if (EnvService.isTruthy("CLAUDE_CODE_DISABLE_FILE_CHECKPOINTING")) return false;
    // Additional logic for agent memory paths or specific settings would go here.
    return true;
}

/**
 * Checks if SDK-level file checkpointing is enabled.
 */
export function isSdkFileCheckpointingEnabled(): boolean {
    return EnvService.isTruthy("CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING") &&
        !EnvService.isTruthy("CLAUDE_CODE_DISABLE_FILE_CHECKPOINTING");
}

/**
 * Returns a unique key for a file path.
 */
export function getFileKey(filePath: string): string {
    return createHash('sha256').update(filePath).digest('hex').slice(0, 16);
}

/**
 * Returns the storage path for file history backups.
 */
export function getBackupStoragePath(): string {
    return join(getBaseConfigDir(), "file-history");
}

/**
 * Resolves a backup file key to a full absolute path.
 */
export function resolveBackupFilePath(fileKey: string): string {
    return join(getBackupStoragePath(), fileKey);
}

/**
 * Logic to track file modification during an operation.
 */
export async function trackFileModification(
    state: HistoryState,
    filePath: string,
    updateState: (newState: HistoryState) => void
): Promise<void> {
    if (!isFileCheckpointingEnabled()) return;

    const fileKey = getFileKey(filePath);
    if (state.trackedFiles.has(fileKey)) return;

    const absolutePath = isAbsolute(filePath) ? filePath : join(process.cwd(), filePath);
    if (!existsSync(absolutePath)) return;

    try {
        const lastSnapshot = state.snapshots.at(-1);
        if (!lastSnapshot) {
            console.warn("[FileHistory] No active snapshot to attach modification to.");
            return;
        }

        if (lastSnapshot.trackedFileBackups[fileKey]) return;

        // Add to tracked files
        const newTrackedFiles = new Set(state.trackedFiles);
        newTrackedFiles.add(fileKey);

        // Create backup
        const backupInfo = createBackupFileInfo(absolutePath, 1);

        // Update last snapshot with this backup
        const updatedSnapshot = {
            ...lastSnapshot,
            trackedFileBackups: {
                ...lastSnapshot.trackedFileBackups,
                [fileKey]: backupInfo
            }
        };

        updateState({
            ...state,
            snapshots: [...state.snapshots.slice(0, -1), updatedSnapshot],
            trackedFiles: newTrackedFiles,
            trackedFilePaths: {
                ...state.trackedFilePaths,
                [fileKey]: absolutePath
            }
        });

    } catch (error) {
        console.error(`[FileHistory] Failed to track modification for ${filePath}:`, error);
    }
}

/**
 * Creates a new snapshot for a given message/turn.
 */
export function createSnapshot(state: HistoryState, messageId: string): Snapshot {
    const fileBackups: Record<string, BackupInfo> = {};
    const lastSnapshot = state.snapshots.at(-1);

    for (const fileKey of state.trackedFiles) {
        // In a real implementation, we'd map fileKey back to absolute path.
        // For now we assume state or context provides this.
        // This is a simplified reconstruction.
    }

    return {
        messageId,
        trackedFileBackups: fileBackups,
        timestamp: new Date()
    };
}

/**
 * Backs up a file and returns its backup info.
 */
export function createBackupFileInfo(absolutePath: string, version: number): BackupInfo {
    const fileKey = getFileKey(absolutePath);
    const backupFileName = `${fileKey}@v${version}`;
    const backupPath = resolveBackupFilePath(backupFileName);

    const dirPath = dirname(backupPath);
    if (!existsSync(dirPath)) {
        mkdirSync(dirPath, { recursive: true });
    }

    const content = readFileSync(absolutePath);
    writeFileSync(backupPath, content);

    const stats = statSync(absolutePath);
    chmodSync(backupPath, stats.mode);

    return {
        backupFileName,
        version,
        backupTime: new Date()
    };
}

/**
 * Restores a file from a backup.
 */
export function restoreFile(absolutePath: string, backupFileName: string | null): void {
    if (backupFileName === null) {
        if (existsSync(absolutePath)) {
            unlinkSync(absolutePath);
        }
        return;
    }

    const backupPath = resolveBackupFilePath(backupFileName);
    if (!existsSync(backupPath)) {
        throw new Error(`[FileHistory] Backup file not found: ${backupPath}`);
    }

    const content = readFileSync(backupPath);
    const dirPath = dirname(absolutePath);
    if (!existsSync(dirPath)) {
        mkdirSync(dirPath, { recursive: true });
    }

    writeFileSync(absolutePath, content);

    const stats = statSync(backupPath);
    chmodSync(absolutePath, stats.mode);
}
