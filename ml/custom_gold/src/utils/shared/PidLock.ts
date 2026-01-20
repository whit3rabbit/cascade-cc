
import { existsSync, readFileSync, writeFileSync, renameSync, unlinkSync, statSync, readdirSync } from 'fs';
import { join, basename } from 'path';

export interface LockInfo {
    pid: number;
    version: string;
    execPath: string;
    acquiredAt: number;
}

function isProcessRunning(pid: number): boolean {
    if (pid <= 1) return false;
    try {
        process.kill(pid, 0);
        return true;
    } catch {
        return false;
    }
}

function readLockFile(lockPath: string): LockInfo | null {
    try {
        if (!existsSync(lockPath)) return null;
        const content = readFileSync(lockPath, 'utf8');
        if (!content.trim()) return null;
        const data = JSON.parse(content);
        if (typeof data.pid !== 'number' || !data.version || !data.execPath) return null;
        return data as LockInfo;
    } catch {
        return null;
    }
}

export function isLockStale(lockPath: string): boolean {
    const lock = readLockFile(lockPath);
    if (!lock) return false;

    if (!isProcessRunning(lock.pid)) return true;

    // Logic from chunk_490:663 - checking if PID is actually Claude
    // ...

    return false;
}

export async function acquireLock(filePath: string, lockPath: string): Promise<(() => void) | null> {
    const fileName = basename(filePath);

    if (readLockFile(lockPath) && !isLockStale(lockPath)) {
        return null;
    }

    const lockInfo: LockInfo = {
        pid: process.pid,
        version: fileName,
        execPath: process.execPath,
        acquiredAt: Date.now()
    };

    const tmpPath = `${lockPath}.tmp.${process.pid}.${Date.now()}`;
    try {
        writeFileSync(tmpPath, JSON.stringify(lockInfo, null, 2), 'utf8');
        renameSync(tmpPath, lockPath);

        // Verify we got it
        const verify = readLockFile(lockPath);
        if (verify?.pid !== process.pid) return null;

        return () => {
            try {
                const current = readLockFile(lockPath);
                if (current?.pid === process.pid) {
                    unlinkSync(lockPath);
                }
            } catch { }
        };
    } catch (err) {
        try { if (existsSync(tmpPath)) unlinkSync(tmpPath); } catch { }
        return null;
    }
}
