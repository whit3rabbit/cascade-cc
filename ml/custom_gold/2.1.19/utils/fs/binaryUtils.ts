/**
 * File: src/utils/fs/binaryUtils.ts
 * Role: Utilities for atomic installation and management of binaries.
 */

import { stat, access, mkdir, copyFile, chmod, rename, unlink } from "node:fs/promises";
import { constants } from "node:fs";
import { dirname, join } from "node:path";
import { getClaudePaths } from "../shared/runtimeAndEnv.js";

/**
 * Checks if a file exists and is executable.
 */
export async function isFileExecutable(filePath: string): Promise<boolean> {
    try {
        const stats = await stat(filePath);
        if (!stats.isFile() || stats.size === 0) {
            return false;
        }
        await access(filePath, constants.X_OK);
        return true;
    } catch {
        return false;
    }
}

/**
 * Atomically installs a binary by copying to a temp file and renaming.
 */
export async function atomicallyInstallBinary(sourcePath: string, destinationPath: string): Promise<void> {
    await mkdir(dirname(destinationPath), { recursive: true });

    const tempFilePath = `${destinationPath}.tmp.${process.pid}.${Date.now()}`;

    try {
        await copyFile(sourcePath, tempFilePath);
        await chmod(tempFilePath, 0o755);
        await rename(tempFilePath, destinationPath);
        console.info(`[Binary] Atomically installed binary to ${destinationPath}`);
    } catch (error) {
        await unlink(tempFilePath).catch(() => { });
        throw error;
    }
}

/**
 * Prepares directories for a specific version.
 */
export async function prepareVersionDirectories(version: string) {
    const paths = getClaudePaths();
    await mkdir(paths.versions, { recursive: true });
    await mkdir(paths.staging, { recursive: true });
    await mkdir(paths.locks, { recursive: true });

    const versionInstallPath = join(paths.versions, version);
    return {
        stagingPath: join(paths.staging, version),
        installPath: versionInstallPath,
    };
}
