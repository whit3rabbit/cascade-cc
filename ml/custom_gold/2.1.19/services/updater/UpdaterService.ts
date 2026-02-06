/**
 * File: src/services/updater/UpdaterService.ts
 * Role: Check for updates from npm registry and handle atomic installs.
 */

import { request } from 'undici';
import semver from 'semver';
import { join, dirname } from 'path';
import { fileURLToPath } from 'url';
import { rename, chmod } from 'fs/promises';
import { readFileSync, existsSync } from 'fs';
import os from 'os';

const __dirname = dirname(fileURLToPath(import.meta.url));

export interface UpdateInfo {
    latestVersion: string;
    currentVersion: string;
    hasUpdate: boolean;
}

export interface UpdateChannelInfo {
    stableVersion?: string;
    latestVersion?: string;
}

import { EnvService } from '../config/EnvService.js';
import { getSettings } from '../config/SettingsService.js';
import { notificationQueue } from '../terminal/NotificationService.js';

export class UpdaterService {
    private static backgroundCheckInterval: NodeJS.Timeout | null = null;
    private static readonly PACKAGE_NAME = '@anthropic-ai/claude-code';
    private static readonly REGISTRY_URL = `https://registry.npmjs.org/${UpdaterService.PACKAGE_NAME}`;

    static async checkForUpdates(): Promise<UpdateInfo | null> {
        try {
            const currentVersion = this.getCurrentVersion();
            if (!currentVersion) return null;

            const { body } = await request(this.REGISTRY_URL, {
                headers: {
                    'Accept': 'application/vnd.npm.install-v1+json'
                }
            });

            const data = await body.json() as any;
            const settings = getSettings();
            const channel = settings.autoUpdatesChannel || 'latest';
            const latestVersion = data['dist-tags']?.[channel];

            if (!latestVersion) return null;

            const hasUpdate = semver.gt(latestVersion, currentVersion);

            return {
                latestVersion,
                currentVersion,
                hasUpdate
            };

        } catch (error) {
            if (EnvService.isTruthy("DEBUG_UPDATER")) {
                console.error('[Updater] Failed to check for updates:', error);
            }
            return null;
        }
    }

    static async getUpdateChannelInfo(): Promise<UpdateChannelInfo | null> {
        try {
            const { body } = await request(this.REGISTRY_URL, {
                headers: {
                    'Accept': 'application/vnd.npm.install-v1+json'
                }
            });

            const data = await body.json() as any;
            return {
                stableVersion: data['dist-tags']?.['stable'],
                latestVersion: data['dist-tags']?.['latest']
            };
        } catch (error) {
            if (EnvService.isTruthy("DEBUG_UPDATER")) {
                console.error('[Updater] Failed to fetch update channels:', error);
            }
            return null;
        }
    }

    static getCurrentVersion(): string | null {
        try {
            const packageJsonPath = join(__dirname, '../../../package.json');
            const content = readFileSync(packageJsonPath, 'utf-8');
            const pkg = JSON.parse(content);
            return pkg.version;
        } catch {
            return EnvService.get("npm_package_version") || null;
        }
    }

    /**
     * Verifies binary integrity.
     * Note: Current version (2.1.19) uses SHA256 checksums from the manifest.json
     * for verification during download, rather than GPG signatures.
     */
    static async verifyIntegrity(filePath: string, expectedChecksum: string): Promise<boolean> {
        try {
            const { readFile } = await import('fs/promises');
            const { createHash } = await import('crypto');
            const buffer = await readFile(filePath);
            const actualChecksum = createHash('sha256').update(buffer).digest('hex');
            return actualChecksum === expectedChecksum;
        } catch (error) {
            if (EnvService.isTruthy("DEBUG_UPDATER")) {
                console.error('[Updater] Integrity verification failed:', error);
            }
            return false;
        }
    }

    /**
     * Atomically installs a binary for the specified version.
     */
    static async atomicallyInstallBinary(version: string, newBinaryPath: string): Promise<boolean> {
        try {
            const executablePath = process.execPath;
            const isWindows = os.platform() === 'win32';

            if (isWindows) {
                const oldPath = `${executablePath}.old.${Date.now()}`;
                try {
                    // 1. Rename running executable to .old (allowed on Windows)
                    await rename(executablePath, oldPath);
                    // 2. Move new binary to executablePath
                    await rename(newBinaryPath, executablePath);
                } catch (e) {
                    // Try to restore if second rename fails
                    if (existsSync(oldPath)) {
                        await rename(oldPath, executablePath).catch(() => { });
                    }
                    throw e;
                }
            } else {
                // On Unix, a simple rename is atomic
                // Ensure executable permissions
                await chmod(newBinaryPath, 0o755);
                await rename(newBinaryPath, executablePath);
            }

            return true;
        } catch (error) {
            console.error('[Updater] Atomic installation failed:', error);
            return false;
        }
    }

    /**
     * Starts periodic background checks for updates.
     */
    static startBackgroundUpdateCheck(): void {
        if (this.backgroundCheckInterval) return;

        const check = async () => {
            const updateInfo = await this.checkForUpdates();
            if (updateInfo?.hasUpdate) {
                notificationQueue.add({
                    text: `A new version of Claude Code is available: ${updateInfo.latestVersion} (current: ${updateInfo.currentVersion}). Run 'claude update' to install.`,
                    type: 'info',
                    key: 'update-available'
                });
            }
        };

        // Initial check
        check();

        // Check every 6 hours
        this.backgroundCheckInterval = setInterval(check, 6 * 60 * 60 * 1000);
    }

    /**
     * Stops periodic background checks.
     */
    static stopBackgroundUpdateCheck(): void {
        if (this.backgroundCheckInterval) {
            clearInterval(this.backgroundCheckInterval);
            this.backgroundCheckInterval = null;
        }
    }

    // getCurrentVersion is intentionally public for diagnostics.
}
