
import { readdir, stat, rm, unlink } from 'fs/promises';
import { join } from 'path';
import { getSettings } from '../services/config/SettingsService.js';

const DEFAULT_CLEANUP_DAYS = 14; // Default from observations or safe default

/**
 * Calculates the date before which files should be deleted.
 */
export function getCleanupThresholdDate(cleanupPeriodDays?: number): Date {
    const days = cleanupPeriodDays ?? getSettings().cleanupPeriodDays ?? DEFAULT_CLEANUP_DAYS;
    return new Date(Date.now() - days * 24 * 60 * 60 * 1000);
}

/**
 * Removes subdirectories in the given path that are older than the threshold.
 */
export async function cleanupOldDirectories(targetDir: string, threshold: Date): Promise<{ deleted: number, errors: number }> {
    const result = { deleted: 0, errors: 0 };
    try {
        const entries = await readdir(targetDir);
        await Promise.all(entries.map(async (entry) => {
            const fullPath = join(targetDir, entry);
            try {
                const stats = await stat(fullPath);
                if (stats.isDirectory() && stats.mtime < threshold) {
                    await rm(fullPath, { recursive: true, force: true });
                    result.deleted++;
                }
            } catch {
                result.errors++;
            }
        }));
    } catch {
        // Directory usually doesn't exist, which is fine
    }
    return result;
}

/**
 * Removes files in the given path that are older than the threshold.
 */
export async function cleanupOldFiles(targetDir: string, threshold: Date, extension?: string): Promise<{ deleted: number, errors: number }> {
    const result = { deleted: 0, errors: 0 };
    try {
        const entries = await readdir(targetDir);
        await Promise.all(entries.map(async (entry) => {
            if (extension && !entry.endsWith(extension)) return;

            const fullPath = join(targetDir, entry);
            try {
                const stats = await stat(fullPath);
                if (stats.isFile() && stats.mtime < threshold) {
                    await unlink(fullPath);
                    result.deleted++;
                }
            } catch {
                result.errors++;
            }
        }));
    } catch {
        // Directory doesn't exist
    }
    return result;
}
