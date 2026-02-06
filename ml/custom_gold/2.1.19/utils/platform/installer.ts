/**
 * File: src/utils/platform/installer.ts
 * Role: Manages native installation checks and executable symlink maintenance.
 */

import { mkdir, rm, symlink, lstat } from 'node:fs/promises';
import { dirname, resolve, delimiter } from 'node:path';
import { existsSync } from 'node:fs';
import { getPlatform } from './detector.js';

/**
 * Updates a symlink (or copy on Windows) to point to the current executable.
 * 
 * @param executablePath - The destination path for the link or binary.
 * @param targetBinary - The actual binary path to link to.
 * @returns {Promise<boolean>} True if the update was successful.
 */
export async function updateExecutableLink(executablePath: string, targetBinary: string): Promise<boolean> {
    const isWin = getPlatform() === 'windows';
    const targetDir = dirname(executablePath);

    try {
        if (!existsSync(targetDir)) {
            await mkdir(targetDir, { recursive: true });
        }

        if (isWin) {
            // Windows: In a real implementation, we might copy the binary.
            // For now, we stub the actual copy logic.
            if (existsSync(executablePath)) {
                // Handle locked files or old versions if necessary
            }
            return true;
        }

        // Unix-like systems: Use symbolic links for the binary.
        if (existsSync(executablePath)) {
            try {
                const stats = await lstat(executablePath);
                if (stats.isSymbolicLink() || stats.isFile()) {
                    await rm(executablePath, { force: true });
                }
            } catch {
                // Ignore if file doesn't exist
            }
        }

        await symlink(targetBinary, executablePath);
        return true;
    } catch (err) {
        console.error(`[Installer] Failed to update executable link at ${executablePath}:`, err);
        return false;
    }
}

/**
 * Checks if the system PATH includes the expected binary location.
 * 
 * @param expectedPath - The directory path to check for in the system PATH.
 * @returns {boolean} True if the path is found.
 */
export function checkPathStatus(expectedPath: string): boolean {
    const pathEnv = process.env.PATH || "";
    const paths = pathEnv.split(delimiter);
    const normalizedExpected = resolve(expectedPath);

    return paths.some(p => {
        try {
            return resolve(p) === normalizedExpected;
        } catch {
            return false;
        }
    });
}
