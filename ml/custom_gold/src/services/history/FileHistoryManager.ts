
import fs from "fs";
import path from "path";
import { createHash } from "crypto";
// import { getProjectRoot } from "../../utils/project/projectRoot"; 
// import { getTaskId } from "../../utils/task/taskId";
// import { diffStats } from "../../utils/shared/diffUtils";
import { log, logError } from "../logger/loggerService.js";

const FILE_HISTORY_DIR_NAME = ".file-history";

interface Backup {
    backupFileName: string | null;
    version: number;
    backupTime: Date;
}

interface Snapshot {
    messageId?: string;
    sessionId?: string;
    trackedFileBackups: Record<string, Backup>;
    trackedFiles?: Set<string>;
    timestamp?: Date;
}

interface FileHistoryState {
    snapshots: Snapshot[];
    trackedFiles: Set<string>;
    messages?: { sessionId?: string }[]; // Mocking structure from chunk_373
    fileHistorySnapshots?: Snapshot[]; // Mocking structure
}

export class FileHistoryManager {
    static instance: FileHistoryManager;

    private getProjectRoot(): string {
        return process.cwd();
    }

    private getHistoryDir(taskId: string): string {
        const cacheDir = path.join(process.cwd(), ".claude_cache"); // Placeholder
        return path.join(cacheDir, FILE_HISTORY_DIR_NAME, taskId);
    }

    private getBackupPath(taskId: string, backupFileName: string): string {
        return path.join(this.getHistoryDir(taskId), backupFileName);
    }

    trackFileModification(state: FileHistoryState, filePath: string, taskId: string): FileHistoryState {
        try {
            const lastSnapshot = state.snapshots[state.snapshots.length - 1];
            if (!lastSnapshot) throw new Error("Missing most recent snapshot");

            const relPath = path.isAbsolute(filePath) ? path.relative(this.getProjectRoot(), filePath) : filePath;

            if (lastSnapshot.trackedFileBackups[relPath]) return state;

            const newTrackedFiles = new Set(state.trackedFiles);
            newTrackedFiles.add(relPath);

            const isNewFile = !fs.existsSync(filePath);
            const backup = isNewFile ? this.createBackup(null, 1, taskId) : this.createBackup(filePath, 1, taskId);

            const newSnapshot = { ...lastSnapshot };
            newSnapshot.trackedFileBackups = { ...lastSnapshot.trackedFileBackups };
            newSnapshot.trackedFileBackups[relPath] = backup;

            return {
                ...state,
                snapshots: [...state.snapshots.slice(0, -1), newSnapshot],
                trackedFiles: newTrackedFiles
            };
        } catch (error) {
            return state;
        }
    }

    createSnapshot(state: FileHistoryState, messageId: string, taskId: string): FileHistoryState {
        try {
            const backups: Record<string, Backup> = {};
            const lastSnapshot = state.snapshots[state.snapshots.length - 1];

            if (lastSnapshot) {
                for (const file of state.trackedFiles) {
                    const absPath = path.resolve(this.getProjectRoot(), file);
                    if (!fs.existsSync(absPath)) {
                        const lastBackup = lastSnapshot.trackedFileBackups[file];
                        const version = lastBackup ? lastBackup.version + 1 : 1;
                        backups[file] = {
                            backupFileName: null,
                            version,
                            backupTime: new Date()
                        };
                    } else {
                        const lastBackup = lastSnapshot.trackedFileBackups[file];
                        const version = lastBackup ? lastBackup.version + 1 : 1;
                        backups[file] = this.createBackup(absPath, version, taskId);
                    }
                }
            }

            const newSnapshot: Snapshot = {
                messageId,
                sessionId: taskId,
                trackedFileBackups: backups,
                timestamp: new Date()
            };

            return {
                ...state,
                snapshots: [...state.snapshots, newSnapshot]
            };
        } catch (error) {
            return state;
        }
    }

    private createBackup(filePath: string | null, version: number, taskId: string): Backup {
        let backupFileName: string | null = null;
        if (filePath !== null) {
            const hash = createHash("sha256").update(filePath).digest("hex").slice(0, 16);
            backupFileName = `${hash}@v${version}`;

            const historyDir = this.getHistoryDir(taskId);
            if (!fs.existsSync(historyDir)) fs.mkdirSync(historyDir, { recursive: true });

            const backupPath = path.join(historyDir, backupFileName);
            fs.copyFileSync(filePath, backupPath);
        }
        return {
            backupFileName,
            version,
            backupTime: new Date()
        };
    }

    async restoreFileHistory(state: FileHistoryState, currentTaskId: string) {
        // Logic from V71
        const snapshots = state.fileHistorySnapshots;
        if (!snapshots || !state.messages || state.messages.length === 0) return;

        const previousSessionId = state.messages[state.messages.length - 1]?.sessionId;
        if (!previousSessionId) {
            // logError("FileHistory: Failed to copy backups on restore (no previous session id)");
            return;
        }

        if (previousSessionId === currentTaskId) {
            // log(`FileHistory: No need to copy file history for resuming with same session id: ${currentTaskId}`);
            return;
        }

        try {
            for (const snapshot of snapshots) {
                let error = false;
                for (const [file, backup] of Object.entries(snapshot.trackedFileBackups)) {
                    if (!backup.backupFileName) continue;

                    const historyProjectDir = this.getHistoryDir(previousSessionId);
                    const currentHistoryDir = this.getHistoryDir(currentTaskId);

                    const srcPath = path.join(historyProjectDir, backup.backupFileName);
                    const destPath = path.join(currentHistoryDir, backup.backupFileName);

                    if (!fs.existsSync(currentHistoryDir)) fs.mkdirSync(currentHistoryDir, { recursive: true });
                    if (fs.existsSync(destPath)) continue;

                    if (!fs.existsSync(srcPath)) {
                        // logError(`FileHistory: Failed to copy backup ${backup.backupFileName} on restore (backup file does not exist in ${previousSessionId})`);
                        error = true;
                        break;
                    }

                    try {
                        fs.linkSync(srcPath, destPath);
                    } catch {
                        try {
                            fs.copyFileSync(srcPath, destPath);
                        } catch {
                            error = true;
                            // logError("FileHistory: Error copying over backup from previous session");
                        }
                    }
                    // log(`FileHistory: Copied backup ${backup.backupFileName} from session ${previousSessionId} to ${currentTaskId}`);
                }
            }
        } catch (error) {
            // logError(error);
        }
    }
}
