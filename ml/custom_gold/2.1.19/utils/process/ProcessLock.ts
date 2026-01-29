/**
 * File: src/utils/process/ProcessLock.ts
 * Role: Manages process-level locking to prevent multiple instances or ensure version compatibility.
 */

import * as fs from "node:fs";
import { join as pathJoin, basename as pathBasename } from "node:path";

const LOCK_EXPIRATION_MS = 7200000; // 2 hours

export interface LockData {
    pid: number;
    version: string;
    execPath: string;
    acquiredAt: number;
}

/**
 * Checks if a lock file exists.
 */
export function isValidLockFile(lockFilePath: string): boolean {
    try {
        fs.statSync(lockFilePath);
        return true;
    } catch {
        return false;
    }
}

/**
 * Checks if a process with the given PID is currently running.
 */
export function isProcessRunning(pid: number): boolean {
    if (pid <= 1) return false;
    try {
        process.kill(pid, 0);
        return true;
    } catch {
        return false;
    }
}

/**
 * Reads and parses the lock file data.
 */
function getProcessInfoFromLock(lockFilePath: string): LockData | null {
    try {
        if (!fs.existsSync(lockFilePath)) return null;

        const fileContent = fs.readFileSync(lockFilePath, 'utf8');
        if (!fileContent || fileContent.trim() === "") return null;

        const parsedContent: LockData = JSON.parse(fileContent);
        if (typeof parsedContent.pid !== "number" || !parsedContent.version || !parsedContent.execPath) {
            return null;
        }
        return parsedContent;
    } catch {
        return null;
    }
}

/**
 * Determines if a lock is stale (process not running or expiration reached).
 */
export function isLockStale(lockFilePath: string): boolean {
    const processInfo = getProcessInfoFromLock(lockFilePath);
    if (!processInfo) return true; // No info means it's effectively stale or invalid

    const { pid } = processInfo;
    if (!isProcessRunning(pid)) return true;

    try {
        const stat = fs.statSync(lockFilePath);
        if (Date.now() - stat.mtimeMs > LOCK_EXPIRATION_MS) {
            // Re-verify process is still dead after expiration check
            if (!isProcessRunning(pid)) return true;
        }
    } catch {
        return true;
    }

    return false;
}

/**
 * Atomically writes a lock file using a temporary file.
 */
function acquireLockFile(lockFilePath: string, lockData: LockData): void {
    const tmpLockFilePath = `${lockFilePath}.tmp.${process.pid}.${Date.now()}`;
    try {
        fs.writeFileSync(tmpLockFilePath, JSON.stringify(lockData, null, 2), 'utf8');
        fs.renameSync(tmpLockFilePath, lockFilePath);
    } catch (e) {
        try {
            if (fs.existsSync(tmpLockFilePath)) fs.unlinkSync(tmpLockFilePath);
        } catch {
            // Ignore cleanup failures
        }
        throw e;
    }
}

/**
 * Attempts to acquire a named lock.
 * 
 * @param lockName - The identity/version string for the lock.
 * @param lockDir - The directory where the lock file should be kept.
 * @returns {Promise<(() => void) | null>} A release function if acquired, otherwise null.
 */
export async function tryAcquireLock(lockName: string, lockDir: string): Promise<(() => void) | null> {
    const lockFilePath = pathJoin(lockDir, `${lockName}.lock`);
    const identity = pathBasename(lockName);

    // If there's an existing lock that isn't stale, we can't acquire it.
    if (fs.existsSync(lockFilePath) && !isLockStale(lockFilePath)) {
        console.log(`[ProcessLock] Cannot acquire lock for ${identity} - held by another active process`);
        return null;
    }

    const lockData: LockData = {
        pid: process.pid,
        version: identity,
        execPath: process.execPath,
        acquiredAt: Date.now()
    };

    try {
        acquireLockFile(lockFilePath, lockData);

        // Double check that we actually won the race
        const currentLockData = getProcessInfoFromLock(lockFilePath);
        if (currentLockData?.pid !== process.pid) {
            return null;
        }

        return () => {
            try {
                const check = getProcessInfoFromLock(lockFilePath);
                if (check?.pid === process.pid) {
                    fs.unlinkSync(lockFilePath);
                }
            } catch (err: any) {
                console.error(`[ProcessLock] Failed to release lock for ${identity}: ${err.message}`);
            }
        };
    } catch (err: any) {
        console.error(`[ProcessLock] Error during lock acquisition for ${identity}: ${err.message}`);
        return null;
    }
}
