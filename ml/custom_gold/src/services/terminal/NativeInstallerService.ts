
import { existsSync, mkdirSync, writeFileSync, copyFileSync, renameSync, unlinkSync, statSync, readdirSync, readlinkSync, symlinkSync, chmodSync } from 'fs';
import { join, dirname, resolve, basename, delimiter } from 'path';
import { homedir } from 'os';
import { getStateHome, getCacheHome, getDataHome, getLocalBin } from '../../utils/shared/XDGUtils.js';
import { acquireLock, isLockStale } from '../../utils/shared/PidLock.js';

export interface InstallationPaths {
    versions: string;
    staging: string;
    locks: string;
    executable: string;
}

export function getClaudePaths(): InstallationPaths {
    const binName = process.platform === 'win32' ? 'claude.exe' : 'claude';
    return {
        versions: join(getDataHome(), 'claude', 'versions'),
        staging: join(getCacheHome(), 'claude', 'staging'),
        locks: join(getStateHome(), 'claude', 'locks'),
        executable: join(getLocalBin(), binName)
    };
}

export function ensureDirectoriesExist() {
    const paths = getClaudePaths();
    [paths.versions, paths.staging, paths.locks].forEach(p => {
        if (!existsSync(p)) mkdirSync(p, { recursive: true });
    });
    const binDir = dirname(paths.executable);
    if (!existsSync(binDir)) mkdirSync(binDir, { recursive: true });
}

export function atomicallyInstallBinary(sourcePath: string, destPath: string) {
    const destDir = dirname(destPath);
    if (!existsSync(destDir)) mkdirSync(destDir, { recursive: true });

    const tmpPath = `${destPath}.tmp.${process.pid}.${Date.now()}`;
    try {
        copyFileSync(sourcePath, tmpPath);
        chmodSync(tmpPath, 0o755);
        renameSync(tmpPath, destPath);
    } catch (err) {
        try { if (existsSync(tmpPath)) unlinkSync(tmpPath); } catch { }
        throw err;
    }
}

export async function performSelfUpdate(targetVersion: string, force: boolean = false) {
    // Logic from Vo5
    console.log(`Checking for update to version ${targetVersion}...`);

    // ... download logic ...

    return { success: true };
}

export async function checkInstallationHealth() {
    // Logic from jb (doctor command)
    const paths = getClaudePaths();
    const errors: any[] = [];

    if (!existsSync(paths.executable)) {
        errors.push({
            message: `Claude command not found at ${paths.executable}`,
            userActionRequired: true,
            type: "error"
        });
    }

    // Check if PATH includes the bin directory
    const pathDirs = (process.env.PATH || "").split(delimiter);
    const binDir = resolve(dirname(paths.executable));
    if (!pathDirs.some(p => resolve(p) === binDir)) {
        errors.push({
            message: `Native installation exists but ${binDir} is not in your PATH.`,
            userActionRequired: true,
            type: "path"
        });
    }

    return errors;
}
