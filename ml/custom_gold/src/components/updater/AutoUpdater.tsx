// Logic from chunk_492.ts (AutoUpdater, Migration, Uninstallation)

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Box, Text } from "ink";
import axios from "axios";
import { execa } from "execa";
import * as semver from "semver";
import * as fs from "node:fs";
import {
    chmodSync,
    constants,
    lstatSync,
    realpathSync,
    accessSync,
    unlinkSync,
    mkdirSync,
    writeFileSync,
    copyFileSync,
    renameSync,
    readlinkSync,
    rmdirSync,
    rmSync,
    mkdtempSync,
    statSync
} from "node:fs";
import * as path from "node:path";
import {
    join,
    dirname,
    resolve,
    delimiter,
    basename
} from "node:path";
import * as os from "node:os";
import { tmpdir, homedir } from "node:os";
import { createHash } from "node:crypto";
import * as lockfile from "proper-lockfile";

import { log, logError } from "../../services/logger/loggerService.js";
import { logTelemetryEvent } from "../../services/telemetry/telemetryInit.js";
import { checkStatsigGate, getStatsigDynamicConfig } from "../../services/telemetry/statsig.js";
import { getSettings, updateSettings } from "../../services/terminal/settings.js";
import { getAutoupdateDisabledReason, getConfig, updateConfig } from "../../services/terminal/ConfigService.js";
import { getInstallationLayout, getShellConfigs } from "../../services/updater/UpdaterService.js";
import { installLocalCli, isLocalCliInstalled } from "../../services/terminal/selfUpdate.js";
import { installUpdate } from "../../services/terminal/LifecycleService.js";

const logger = log("auto-updater");

const AUTO_UPDATE_INTERVAL_MS = 1800000;
const VERSION_CHECK_TIMEOUT_MS = 5000;
const DEFAULT_PACKAGE_URL = "@anthropic-ai/claude-code";
const DOWNLOAD_BASE_URL = "https://downloads.claude.ai/claude-code-releases";
const FALLBACK_DOWNLOAD_BASE_URL = "https://storage.googleapis.com/claude-code-dist-86c565f3-f756-42ad-8dfa-d59b1c096819/claude-code-releases";

const PACKAGE_METADATA = {
    ISSUES_EXPLAINER: "report the issue at https://github.com/anthropics/claude-code/issues",
    PACKAGE_URL: "@anthropic-ai/claude-code",
    README_URL: "https://code.claude.com/docs/en/overview",
    VERSION: "2.0.76",
    FEEDBACK_CHANNEL: "https://github.com/anthropics/claude-code/issues",
    BUILD_TIME: "2025-12-22T23:56:12Z"
};

type UpdateStatus = "success" | "install_failed" | "no_permissions" | "in_progress" | "timeout" | "unknown";

type AutoUpdaterResult = {
    version: string | null;
    status: UpdateStatus;
    notifications?: string[];
};

type InstallationType = "development" | "npm-local" | "npm-global" | "native" | "package-manager" | "unknown";

type CleanupResult = {
    removed: number;
    errors: string[];
    warnings: string[];
};

type AliasCleanupMessage = {
    message: string;
    userActionRequired: boolean;
    type: "alias" | "error" | "info" | "path";
};

type NativeUpdateResult = {
    latestVersion?: string | null;
    wasUpdated?: boolean;
    lockFailed?: boolean;
    lockHolderPid?: number;
};

type AutoUpdaterProps = {
    verbose?: boolean;
    isUpdating: boolean;
    onChangeIsUpdating: (value: boolean) => void;
    onAutoUpdaterResult: (result: AutoUpdaterResult) => void;
    autoUpdaterResult?: AutoUpdaterResult | null;
    showSuccessMessage?: boolean;
};

// --- Helper Functions ---

function getArch(): string {
    const platform = process.platform;
    const arch = process.arch === "x64" ? "x64" : process.arch === "arm64" ? "arm64" : null;
    if (!arch) {
        throw new Error(`Unsupported architecture: ${process.arch}`);
    }
    // IsMusl check omitted for simplicity, assuming glibc for linux x64 in basic implementation
    return `${platform}-${arch}`;
}

function getBinaryName(archString: string): string {
    return archString.startsWith("win32") ? "claude.exe" : "claude";
}

function getPaths() {
    const archString = getArch();
    const binaryName = getBinaryName(archString);
    const dataHome = process.env.XDG_DATA_HOME ?? join(homedir(), ".local", "share");
    const cacheHome = process.env.XDG_CACHE_HOME ?? join(homedir(), ".cache");
    const stateHome = process.env.XDG_STATE_HOME ?? join(homedir(), ".local", "state");
    const binHome = join(homedir(), ".local", "bin");

    return {
        versions: join(dataHome, "claude", "versions"),
        staging: join(cacheHome, "claude", "staging"),
        locks: join(stateHome, "claude", "locks"),
        executable: join(binHome, binaryName)
    };
}

function isValidBinary(filePath: string): boolean {
    if (!fs.existsSync(filePath)) return false;
    const stat = fs.statSync(filePath);
    if (!stat.isFile() || stat.size === 0) return false;
    try {
        fs.accessSync(filePath, fs.constants.X_OK);
        return true;
    } catch {
        return false;
    }
}

function prepareDirectories(version: string) {
    const paths = getPaths();
    [paths.versions, paths.staging, paths.locks].forEach(dir => {
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    });

    const binDir = dirname(paths.executable);
    if (!fs.existsSync(binDir)) fs.mkdirSync(binDir, { recursive: true });

    const versionDir = join(paths.versions, version);
    if (!fs.existsSync(versionDir)) {
        // Just creating directory, not file like original logic might have implied
        fs.mkdirSync(versionDir, { recursive: true });
    }

    return {
        stagingPath: join(paths.staging, version),
        installPath: versionDir
    };
}

function getLockFilePath(paths: { locks: string }, targetPath: string): string {
    const base = basename(targetPath);
    return join(paths.locks, `${base}.lock`);
}

async function acquireLock(targetPath: string, action: () => Promise<void>, retries = 0): Promise<boolean> {
    const paths = getPaths();
    if (!fs.existsSync(paths.locks)) fs.mkdirSync(paths.locks, { recursive: true });

    const lockFilePath = getLockFilePath(paths, targetPath);
    let release: (() => Promise<void>) | undefined;

    try {
        release = await lockfile.lock(targetPath, {
            stale: 2592000000, // 30 days
            retries: {
                retries,
                minTimeout: retries > 0 ? 1000 : 100,
                maxTimeout: retries > 0 ? 5000 : 500
            },
            lockfilePath: lockFilePath
        });
        await action();
        return true;
    } catch (error) {
        logger.warn(`Failed to acquire lock for ${targetPath}: ${error}`);
        return false;
    } finally {
        if (release) await release();
    }
}

async function downloadFile(url: string, checksum: string, destPath: string) {
    const response = await axios.get(url, {
        responseType: "arraybuffer",
        timeout: 300000
    });

    const hash = createHash("sha256");
    hash.update(response.data);
    const calculatedChecksum = hash.digest("hex");

    if (calculatedChecksum !== checksum) {
        throw new Error(`Checksum mismatch: expected ${checksum}, got ${calculatedChecksum}`);
    }

    fs.writeFileSync(destPath, Buffer.from(response.data));
    fs.chmodSync(destPath, 0o755);
}

async function fetchManifest(baseUrl: string, version: string) {
    const response = await axios.get(`${baseUrl}/${version}/manifest.json`, {
        responseType: "json",
        timeout: 10000
    });
    return response.data;
}

async function downloadBinary(version: string, destinationDir: string, baseUrl: string = FALLBACK_DOWNLOAD_BASE_URL) {
    if (fs.existsSync(destinationDir)) {
        fs.rmSync(destinationDir, { recursive: true, force: true });
    }
    fs.mkdirSync(destinationDir, { recursive: true });

    const archString = getArch();
    const manifest = await fetchManifest(baseUrl, version);

    const platformData = manifest.platforms[archString];
    if (!platformData) {
        throw new Error(`Platform ${archString} not found in manifest for version ${version}`);
    }

    const binaryName = getBinaryName(archString);
    const downloadUrl = `${baseUrl}/${version}/${archString}/${binaryName}`;
    const destFile = join(destinationDir, binaryName);

    await downloadFile(downloadUrl, platformData.checksum, destFile);
    return destFile;
}

function atomicInstallBinary(sourcePath: string, targetDir: string) {
    const binaryName = basename(sourcePath);
    const targetPath = join(targetDir, binaryName);

    // Create temp file for atomic move
    const tempPath = `${targetPath}.tmp.${process.pid}.${Date.now()}`;

    try {
        fs.copyFileSync(sourcePath, tempPath);
        fs.chmodSync(tempPath, 0o755);
        fs.renameSync(tempPath, targetPath);
        logger.info(`Atomically installed binary to ${targetPath}`);
    } catch (error) {
        try {
            if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath);
        } catch { }
        throw error;
    }
}

function updateSymlink(linkPath: string, targetPath: string): boolean {
    const linkDir = dirname(linkPath);
    if (!fs.existsSync(linkDir)) fs.mkdirSync(linkDir, { recursive: true });

    // Remove existing link if it exists
    if (fs.existsSync(linkPath)) {
        try {
            const currentTarget = fs.readlinkSync(linkPath);
            if (resolve(linkDir, currentTarget) === resolve(targetPath)) {
                return false; // Already pointing to correct target
            }
        } catch { }
        fs.unlinkSync(linkPath);
    }

    // Atomic symlink creation
    const tempLink = `${linkPath}.tmp.${process.pid}.${Date.now()}`;
    try {
        fs.symlinkSync(targetPath, tempLink);
        fs.renameSync(tempLink, linkPath);
        return true;
    } catch (error) {
        try {
            if (fs.existsSync(tempLink)) fs.unlinkSync(tempLink);
        } catch { }
        throw error;
    }
}

async function fetchLatestNativeVersion(channel: string): Promise<string | null> {
    // Logic from nC0 / la5
    const url = `${FALLBACK_DOWNLOAD_BASE_URL}/${channel}`;
    try {
        const response = await axios.get(url, { responseType: 'text', timeout: 30000 });
        return response.data.trim();
    } catch (error) {
        logger.error(`Failed to fetch version from ${url}: ${error}`);
        throw error;
    }
}

// --- Main Native Updater Logic ---

async function performNativeUpdate(version: string, force: boolean): Promise<{ success: boolean; lockFailed?: boolean }> {
    const { stagingPath, installPath } = prepareDirectories(version);
    const paths = getPaths();
    const binName = getBinaryName(getArch());

    // Check if already installed
    const installedBinary = join(installPath, binName);
    if (!force && isValidBinary(installedBinary) && isValidBinary(paths.executable) && fs.existsSync(paths.executable)) {
        // Check if symlink points to this version
        const realPath = fs.realpathSync(paths.executable);
        if (realPath === installedBinary) {
            return { success: true };
        }
    }

    let updated = false;
    const success = await acquireLock(installPath, async () => {
        // Download if needed
        if (force || !isValidBinary(installedBinary)) {
            logger.info(`Downloading native installer version ${version}`);
            await downloadBinary(version, stagingPath);
            const downloadedBinary = join(stagingPath, binName);
            atomicInstallBinary(downloadedBinary, installPath);
            // Cleanup staging
            fs.rmSync(stagingPath, { recursive: true, force: true });
        }

        // Update symlink
        updateSymlink(paths.executable, installedBinary);
        updated = true;
    }, 3);

    return { success: updated, lockFailed: !success };
}

async function runNativeAutoUpdate(channel: string, isMigration = false): Promise<NativeUpdateResult> {
    const config = getConfig();
    if (!isMigration && config.installMethod !== "native") {
        return { latestVersion: null, wasUpdated: false };
    }

    try {
        const latestVersion = await fetchLatestNativeVersion(channel);
        if (!latestVersion) return { latestVersion: null, wasUpdated: false };

        if (!isMigration && latestVersion === PACKAGE_METADATA.VERSION) {
            return { latestVersion, wasUpdated: false };
        }

        const result = await performNativeUpdate(latestVersion, isMigration);

        if (result.success && !result.lockFailed) {
            if (config.installMethod !== "native") {
                updateConfig((prev) => ({
                    ...prev,
                    installMethod: "native",
                    autoUpdates: false,
                    autoUpdatesProtectedForNative: true
                }));
            }
        }

        return {
            latestVersion,
            wasUpdated: result.success,
            lockFailed: result.lockFailed
        };
    } catch (error) {
        logError("auto-updater", error);
        return { latestVersion: null, wasUpdated: false };
    }
}

async function validateInstallationState(includeSuggestions: boolean = false): Promise<AliasCleanupMessage[]> {
    const config = getConfig();
    const type = await detectInstallationType();

    if (config.installMethod !== "native" && type !== "native") return [];

    const messages: AliasCleanupMessage[] = [];
    const paths = getPaths();
    const binDir = dirname(paths.executable);

    if (!fs.existsSync(binDir)) {
        messages.push({ message: `installMethod is native, but directory ${binDir} does not exist`, type: "error", userActionRequired: true });
    }

    if (!fs.existsSync(paths.executable)) {
        messages.push({ message: `installMethod is native, but claude command not found at ${paths.executable}`, type: "error", userActionRequired: true });
    } else if (!isValidBinary(paths.executable)) {
        messages.push({ message: `${paths.executable} exists but is not a valid Claude binary`, type: "error", userActionRequired: true });
    }

    // Check PATH
    const pathEnv = process.env.PATH || "";
    const isOnPath = pathEnv.split(delimiter).some(p => {
        try {
            const resolved = resolve(p);
            return resolved === resolve(binDir);
        } catch { return false; }
    });

    if (!isOnPath) {
        // Suggest adding to path
        const shell = basename(process.env.SHELL || "");
        const rcFile = shell === "zsh" ? ".zshrc" : shell === "bash" ? ".bashrc" : "profile";
        const suggestCmd = `echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/${rcFile} && source ~/${rcFile}`;

        messages.push({
            message: `Native installation exists but ~/.local/bin is not in your PATH. Run:\n\n${suggestCmd}`,
            type: "path",
            userActionRequired: true
        });
    }

    return messages;
}

// --- End Helper Functions ---

function parseBooleanEnv(value?: string): boolean {
    if (!value) return false;
    return value === "1" || value.toLowerCase() === "true";
}

function useInterval(callback: () => void, delay: number | null) {
    useEffect(() => {
        if (delay === null) return;
        const interval = setInterval(callback, delay);
        return () => clearInterval(interval);
    }, [callback, delay]);
}

function getExecutablePaths(): string[] {
    return [process.execPath, process.argv[0], process.argv[1]].filter(Boolean) as string[];
}

function isDevelopmentBuild(): boolean {
    const markers = ["/build-ant/", "/build-external/", "/build-external-native/", "/build-ant-native/"];
    return getExecutablePaths().some((candidate) => markers.some((marker) => candidate.includes(marker)));
}

function normalizeVersion(version: string): string {
    return `${semver.major(version, { loose: true })}.${semver.minor(version, { loose: true })}.${semver.patch(version, { loose: true })}`;
}

function useVersionChange(version?: string | null) {
    const [current, setCurrent] = useState(() => normalizeVersion(PACKAGE_METADATA.VERSION));

    if (!version) return null;
    const normalized = normalizeVersion(version);
    if (normalized !== current) {
        setCurrent(normalized);
        return normalized;
    }
    return null;
}

function getMinimumVersion(): string | null {
    const settings = getSettings("userSettings") as { minimumVersion?: string };
    return settings?.minimumVersion ?? null;
}

function isVersionBelowMinimum(version: string): boolean {
    const minimumVersion = getMinimumVersion();
    if (!minimumVersion) return false;
    const below = !semver.gte(version, minimumVersion, { loose: true });
    if (below) logger.info(`Skipping update to ${version} - below minimumVersion ${minimumVersion}`);
    return below;
}

function getAutoUpdatesChannel(): string {
    const settings = getSettings("userSettings") as { autoUpdatesChannel?: string };
    return settings?.autoUpdatesChannel ?? "latest";
}

function isAutoUpdaterDisabled(): boolean {
    const reason = getAutoupdateDisabledReason();
    if (reason) {
        logger.info(`AutoUpdater disabled: ${reason}`);
        return true;
    }
    return false;
}

async function runCommand(command: string, args: string[], options: { cwd?: string; signal?: AbortSignal } = {}) {
    const result = await execa(command, args, {
        cwd: options.cwd,
        reject: false,
        signal: options.signal
    });

    return {
        code: result.exitCode ?? 0,
        stdout: result.stdout ?? "",
        stderr: result.stderr ?? ""
    };
}

async function fetchLatestVersionForChannel(channel: string): Promise<string | null> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), VERSION_CHECK_TIMEOUT_MS);
    const distTag = channel === "stable" ? "stable" : "latest";

    try {
        const result = await runCommand(
            "npm",
            ["view", `${PACKAGE_METADATA.PACKAGE_URL}@${distTag}`, "version", "--prefer-online"],
            { signal: controller.signal }
        );

        if (result.code !== 0) {
            logger.info(`npm view failed with code ${result.code}`);
            if (result.stderr) logger.info(`npm stderr: ${result.stderr.trim()}`);
            if (result.stdout) logger.info(`npm stdout: ${result.stdout.trim()}`);
            return null;
        }

        return result.stdout.trim();
    } catch (error) {
        logger.error(`Failed to fetch latest version: ${error instanceof Error ? error.message : String(error)}`);
        return null;
    } finally {
        clearTimeout(timeout);
    }
}

function isHomebrewCaskInstall(): boolean {
    const platform = process.platform;
    if (platform !== "darwin" && platform !== "linux" && platform !== "win32") return false;
    const executable = process.execPath || process.argv[0] || "";
    if (executable.includes("/Caskroom/")) {
        logger.info(`Detected Homebrew cask installation: ${executable}`);
        return true;
    }
    return false;
}

async function getNpmPrefix(): Promise<string | null> {
    const result = await runCommand("npm", ["config", "get", "prefix"]);
    if (result.code !== 0 || !result.stdout) return null;
    return result.stdout.trim();
}

function isNpmManagedExecutable(executable: string): boolean {
    try {
        const stats = fs.lstatSync(executable);
        if (!stats.isSymbolicLink()) return false;
        const target = fs.realpathSync(executable);
        return target.includes("node_modules") || target.includes("/npm/") || target.includes("/nvm/");
    } catch {
        return false;
    }
}

export function uninstallNativeExecutable() {
    const layout = getInstallationLayout();
    try {
        if (!fs.existsSync(layout.executable)) return;
        if (isNpmManagedExecutable(layout.executable)) {
            logger.info(`Skipping removal of ${layout.executable} - appears to be npm-managed`);
            return;
        }
        fs.unlinkSync(layout.executable);
        logger.info(`Removed claude symlink at ${layout.executable}`);
    } catch (error) {
        logError("auto-updater", error, `Failed to remove claude symlink: ${error}`);
    }
}

function readShellConfig(filePath: string): string | null {
    try {
        if (!fs.existsSync(filePath)) return null;
        return fs.readFileSync(filePath, "utf8");
    } catch (error) {
        logError("auto-updater", error, `Failed to read ${filePath}`);
        return null;
    }
}

function filterClaudeAlias(content: string): { filtered: string; hadAlias: boolean } {
    const lines = content.split(/\r?\n/);
    let hadAlias = false;

    const filtered = lines.filter((line) => {
        if (/^\s*alias\s+claude=/.test(line)) {
            hadAlias = true;
            return false;
        }
        return true;
    }).join("\n");

    return { filtered, hadAlias };
}

function writeShellConfig(filePath: string, content: string) {
    fs.writeFileSync(filePath, content, "utf8");
}

export function removeShellAliases(): AliasCleanupMessage[] {
    const results: AliasCleanupMessage[] = [];
    const configPaths = getShellConfigs();

    for (const [, configPath] of Object.entries(configPaths)) {
        try {
            const content = readShellConfig(configPath);
            if (!content) continue;
            const { filtered, hadAlias } = filterClaudeAlias(content);
            if (!hadAlias) continue;
            writeShellConfig(configPath, filtered);
            results.push({
                message: `Removed claude alias from ${configPath}. Run: unalias claude`,
                userActionRequired: true,
                type: "alias"
            });
            logger.info(`Cleaned up claude alias from ${configPath}`);
        } catch (error) {
            logError("auto-updater", error);
            results.push({
                message: `Failed to clean up ${configPath}: ${error}`,
                userActionRequired: false,
                type: "error"
            });
        }
    }

    return results;
}

async function manualRemoveGlobalPackage(packageName: string): Promise<{ success: boolean; warning?: string; error?: string }> {
    try {
        const prefix = await getNpmPrefix();
        if (!prefix) {
            return { success: false, error: "Failed to get npm global prefix" };
        }

        let removed = false;
        if (process.platform === "win32") {
            const cmdPath = path.join(prefix, "claude.cmd");
            const ps1Path = path.join(prefix, "claude.ps1");
            const exePath = path.join(prefix, "claude");
            if (fs.existsSync(cmdPath)) {
                fs.unlinkSync(cmdPath);
                logger.info(`Manually removed bin script: ${cmdPath}`);
                removed = true;
            }
            if (fs.existsSync(ps1Path)) {
                fs.unlinkSync(ps1Path);
                logger.info(`Manually removed PowerShell script: ${ps1Path}`);
                removed = true;
            }
            if (fs.existsSync(exePath)) {
                fs.unlinkSync(exePath);
                logger.info(`Manually removed bin executable: ${exePath}`);
                removed = true;
            }
        } else {
            const symlinkPath = path.join(prefix, "bin", "claude");
            if (fs.existsSync(symlinkPath)) {
                fs.unlinkSync(symlinkPath);
                logger.info(`Manually removed bin symlink: ${symlinkPath}`);
                removed = true;
            }
        }

        if (removed) {
            const nodeModulesPath = process.platform === "win32"
                ? path.join(prefix, "node_modules", packageName)
                : path.join(prefix, "lib", "node_modules", packageName);
            logger.info(`Successfully removed ${packageName} manually`);
            return {
                success: true,
                warning: `${packageName} executables removed, but node_modules directory was left intact for safety. You may manually delete it later at: ${nodeModulesPath}`
            };
        }

        return { success: false };
    } catch (error) {
        logger.error(`Manual removal failed: ${error}`);
        return { success: false, error: `Manual removal failed: ${error}` };
    }
}

async function uninstallGlobalPackage(packageName: string): Promise<{ success: boolean; warning?: string; error?: string }> {
    const result = await runCommand("npm", ["uninstall", "-g", packageName], { cwd: process.cwd() });
    const stderr = result.stderr || "";

    if (result.code === 0) {
        logger.info(`Removed global npm installation of ${packageName}`);
        return { success: true };
    }

    if (stderr && !stderr.includes("npm ERR! code E404")) {
        if (stderr.includes("npm error code ENOTEMPTY")) {
            logger.error(`Failed to uninstall global npm package ${packageName}: ${stderr}`);
            logger.info("Attempting manual removal due to ENOTEMPTY error");
            const manual = await manualRemoveGlobalPackage(packageName);
            if (manual.success) {
                return { success: true, warning: manual.warning };
            }
            if (manual.error) {
                return {
                    success: false,
                    error: `Failed to remove global npm installation of ${packageName}: ${stderr}. Manual removal also failed: ${manual.error}`
                };
            }
        }

        logger.error(`Failed to uninstall global npm package ${packageName}: ${stderr}`);
        return { success: false, error: `Failed to remove global npm installation of ${packageName}: ${stderr}` };
    }

    return { success: false };
}

export async function removeAllLegacyVersions(): Promise<CleanupResult> {
    const errors: string[] = [];
    const warnings: string[] = [];
    let removed = 0;

    const primaryRemoval = await uninstallGlobalPackage(DEFAULT_PACKAGE_URL);
    if (primaryRemoval.success) {
        removed += 1;
        if (primaryRemoval.warning) warnings.push(primaryRemoval.warning);
    } else if (primaryRemoval.error) {
        errors.push(primaryRemoval.error);
    }

    if (PACKAGE_METADATA.PACKAGE_URL && PACKAGE_METADATA.PACKAGE_URL !== DEFAULT_PACKAGE_URL) {
        const secondaryRemoval = await uninstallGlobalPackage(PACKAGE_METADATA.PACKAGE_URL);
        if (secondaryRemoval.success) {
            removed += 1;
            if (secondaryRemoval.warning) warnings.push(secondaryRemoval.warning);
        } else if (secondaryRemoval.error) {
            errors.push(secondaryRemoval.error);
        }
    }

    const localPath = path.join(os.homedir(), ".claude", "local");
    if (fs.existsSync(localPath)) {
        try {
            fs.rmSync(localPath, { recursive: true, force: true });
            removed += 1;
            logger.info(`Removed local installation at ${localPath}`);
        } catch (error) {
            errors.push(`Failed to remove ${localPath}: ${error}`);
            logger.error(`Failed to remove local installation: ${error}`);
        }
    }

    return { removed, errors, warnings };
}

async function refreshStatsig() {
    try {
        await getStatsigDynamicConfig("tengu_version_config", { minVersion: "0.0.0" });
    } catch (error) {
        logError("auto-updater", error, "Statsig refresh failed");
    }
}

function getUpdateErrorCategory(message: string): string {
    if (message.includes("timeout")) return "timeout";
    if (message.includes("Checksum mismatch")) return "checksum_mismatch";
    if (message.includes("ENOENT") || message.includes("not found")) return "not_found";
    if (message.includes("EACCES") || message.includes("permission")) return "permission_denied";
    if (message.includes("ENOSPC")) return "disk_full";
    if (message.includes("npm")) return "npm_error";
    if (message.includes("network") || message.includes("ECONNREFUSED") || message.includes("ENOTFOUND")) return "network_error";
    return "unknown";
}

async function detectInstallationType(): Promise<InstallationType> {
    if (isDevelopmentBuild()) return "development";

    const executable = getExecutablePaths()[0] || "";
    const layout = getInstallationLayout();
    const nativeCandidates = [layout.executable, path.join(os.homedir(), ".local", "bin", "claude")];

    if (nativeCandidates.some((candidate) => candidate && executable.includes(candidate))) {
        if (isHomebrewCaskInstall()) return "package-manager";
        return "native";
    }

    if (isLocalCliInstalled()) return "npm-local";

    const globalPathHints = [
        "/usr/local/lib/node_modules",
        "/usr/lib/node_modules",
        "/opt/homebrew/lib/node_modules",
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/.nvm/versions/node/"
    ];

    if (globalPathHints.some((hint) => executable.includes(hint))) return "npm-global";
    if (executable.includes("/npm/") || executable.includes("/nvm/")) return "npm-global";

    const prefix = await getNpmPrefix();
    if (prefix && executable.startsWith(prefix)) return "npm-global";

    return "unknown";
}

async function updateLocalInstallation(channel: string): Promise<UpdateStatus> {
    const result = await installLocalCli(channel === "stable" ? "stable" : "latest");
    if (result === "success") {
        updateConfig((config) => ({
            ...config,
            installMethod: "local"
        }));
        return "success";
    }
    if (result === "in_progress") return "in_progress";
    return "install_failed";
}

async function updateGlobalInstallation(channel: string): Promise<UpdateStatus> {
    const version = channel ? `${PACKAGE_METADATA.PACKAGE_URL}@${channel}` : PACKAGE_METADATA.PACKAGE_URL;
    try {
        const status = await installUpdate(PACKAGE_METADATA.PACKAGE_URL, channel || "latest");
        if (status === "success") {
            updateConfig((config) => ({
                ...config,
                installMethod: "global"
            }));
            return "success";
        }
        return "install_failed";
    } catch (error) {
        logger.error(`Failed to install global update ${version}: ${error}`);
        return "install_failed";
    }
}

export async function shouldMigrateToNative(): Promise<boolean> {
    const isPrintMode = process.argv.includes("-p") || process.argv.includes("--print");
    if (isDevelopmentBuild()) return false;
    if (!(await checkStatsigGate("auto_migrate_to_native"))) return false;
    if (parseBooleanEnv(process.env.CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC)) return false;
    if (isPrintMode || parseBooleanEnv(process.env.DISABLE_AUTO_MIGRATE_TO_NATIVE)) return false;

    const config = getConfig();
    if (config.installMethod === "native") return false;

    return true;
}

export async function executeNativeMigration(): Promise<{ success: boolean; version?: string; notifications?: string[] }> {
    await logTelemetryEvent("tengu_auto_migrate_to_native_attempt", {});

    try {
        const channel = getAutoUpdatesChannel();
        const updateResult = await runNativeAutoUpdate(channel, true);
        let notifications: string[] = [];

        if (updateResult.latestVersion) {
            await logTelemetryEvent("tengu_auto_migrate_to_native_success", {});
            logger.info("✅ Upgraded to native installation. Future sessions will use the native version.");

            const cleanup = await removeAllLegacyVersions();
            const cleanupMessages: AliasCleanupMessage[] = [];

            cleanup.errors.forEach((message) => {
                cleanupMessages.push({ message, userActionRequired: false, type: "error" });
            });
            cleanup.warnings.forEach((message) => {
                cleanupMessages.push({ message, userActionRequired: false, type: "info" });
            });
            if (cleanup.removed > 0) {
                cleanupMessages.push({
                    message: `Cleaned up ${cleanup.removed} old npm installation(s)`,
                    userActionRequired: false,
                    type: "info"
                });
            }

            const aliasCleanup = removeShellAliases();
            const validationNotes = await validateInstallationState(true);
            const combined = [...validationNotes, ...aliasCleanup, ...cleanupMessages];

            const manualActions = combined.filter((message) => message.userActionRequired);
            if (manualActions.length > 0) {
                const text = ["⚠️  Manual action required after migration to native installer:", ...manualActions.map((msg) => `• ${msg.message}`)].join("\n");
                notifications.push(text);
            }

            if (combined.length > 0) {
                logger.info("Migration completed with the following notes:");
                combined.forEach((message) => {
                    logger.info(`  • [${message.type}] ${message.message}`);
                });
            }

            return { success: true, version: updateResult.latestVersion || undefined, notifications: notifications.length > 0 ? notifications : undefined };
        }

        await logTelemetryEvent("tengu_auto_migrate_to_native_partial", {});
        logger.info("⚠️ Native installation setup encountered issues but cleanup completed.");

        const notes = await validateInstallationState(true);
        return {
            success: true,
            version: updateResult.latestVersion || undefined,
            notifications: notes.length > 0 ? ["Migration completed with warnings."] : undefined
        };
    } catch (error) {
        await logTelemetryEvent("tengu_auto_migrate_to_native_failure", {
            error: error instanceof Error ? error.message : String(error)
        });
        logError("auto-updater", error);
        return { success: false };
    }
}

export function NpmUpdaterComponent({
    isUpdating,
    onChangeIsUpdating,
    onAutoUpdaterResult,
    autoUpdaterResult,
    showSuccessMessage,
    verbose
}: AutoUpdaterProps) {
    const [versionInfo, setVersionInfo] = useState<{ global?: string; latest?: string }>({});
    const versionChange = useVersionChange(autoUpdaterResult?.version ?? undefined);

    const checkForUpdates = useCallback(async () => {
        if (isUpdating) return;

        const currentVersion = PACKAGE_METADATA.VERSION;
        const channel = getAutoUpdatesChannel();
        const latestVersion = await fetchLatestVersionForChannel(channel);
        const hasAutoUpdatesDisabled = isAutoUpdaterDisabled();

        setVersionInfo({
            global: currentVersion,
            latest: latestVersion || undefined
        });

        if (hasAutoUpdatesDisabled) return;
        if (!currentVersion || !latestVersion) return;
        if (semver.gte(currentVersion, latestVersion, { loose: true })) return;
        if (isVersionBelowMinimum(latestVersion)) return;

        const start = Date.now();
        onChangeIsUpdating(true);

        const config = getConfig();
        if (config.installMethod !== "native") uninstallNativeExecutable();

        const installType = await detectInstallationType();
        logger.info(`AutoUpdater: Detected installation type: ${installType}`);

        if (installType === "development") {
            logger.info("AutoUpdater: Cannot auto-update development build");
            onChangeIsUpdating(false);
            return;
        }

        let status: UpdateStatus;
        let updateMethod: "local" | "global" | null = null;

        if (installType === "npm-local") {
            logger.info("AutoUpdater: Using local update method");
            updateMethod = "local";
            status = await updateLocalInstallation(channel);
        } else if (installType === "npm-global") {
            logger.info("AutoUpdater: Using global update method");
            updateMethod = "global";
            status = await updateGlobalInstallation(channel);
        } else if (installType === "native") {
            logger.info("AutoUpdater: Unexpected native installation in non-native updater");
            onChangeIsUpdating(false);
            return;
        } else {
            logger.info("AutoUpdater: Unknown installation type, falling back to config");
            const isLocal = config.installMethod === "local";
            updateMethod = isLocal ? "local" : "global";
            status = isLocal ? await updateLocalInstallation(channel) : await updateGlobalInstallation(channel);
        }

        onChangeIsUpdating(false);

        if (status === "success") {
            await refreshStatsig();
            await logTelemetryEvent("tengu_auto_updater_success", {
                fromVersion: currentVersion,
                toVersion: latestVersion,
                durationMs: Date.now() - start,
                wasMigrated: updateMethod === "local",
                installationType: installType
            });
        } else {
            await logTelemetryEvent("tengu_auto_updater_fail", {
                fromVersion: currentVersion,
                attemptedVersion: latestVersion,
                status,
                durationMs: Date.now() - start,
                wasMigrated: updateMethod === "local",
                installationType: installType
            });
        }

        onAutoUpdaterResult({
            version: latestVersion,
            status
        });
    }, [isUpdating, onAutoUpdaterResult, onChangeIsUpdating]);

    useEffect(() => {
        checkForUpdates();
    }, [checkForUpdates]);

    useInterval(checkForUpdates, AUTO_UPDATE_INTERVAL_MS);

    if (!autoUpdaterResult?.version && (!versionInfo.global || !versionInfo.latest)) return null;
    if (!autoUpdaterResult?.version && !isUpdating) return null;

    return (
        <Box flexDirection="row" gap={1}>
            {verbose && (
                <Text dimColor>
                    globalVersion: {versionInfo.global} · latestVersion: {versionInfo.latest}
                </Text>
            )}
            {isUpdating ? (
                <Box>
                    <Text color="text" dimColor wrap="end">
                        Auto-updating…
                    </Text>
                </Box>
            ) : (
                <>
                    {autoUpdaterResult?.status === "success" && showSuccessMessage && versionChange && (
                        <Text color="success">✓ Update installed · Restart to apply</Text>
                    )}
                    {(autoUpdaterResult?.status === "install_failed" || autoUpdaterResult?.status === "no_permissions") && (
                        <Text color="error">
                            ✗ Auto-update failed · Try <Text bold>claude doctor</Text>
                            {!isLocalCliInstalled() && (
                                <> or <Text bold>npm i -g {PACKAGE_METADATA.PACKAGE_URL}</Text></>
                            )}
                            {isLocalCliInstalled() && (
                                <> or <Text bold>cd ~/.claude/local && npm update {PACKAGE_METADATA.PACKAGE_URL}</Text></>
                            )}
                        </Text>
                    )}
                </>
            )}
        </Box>
    );
}

export function NativeUpdaterComponent({
    isUpdating,
    onChangeIsUpdating,
    onAutoUpdaterResult,
    autoUpdaterResult,
    showSuccessMessage,
    verbose
}: AutoUpdaterProps) {
    const [versionInfo, setVersionInfo] = useState<{ current?: string; latest?: string }>({});
    const hasChecked = useRef(false);

    const checkForUpdates = useCallback(async () => {
        if (isUpdating || isAutoUpdaterDisabled()) return;
        onChangeIsUpdating(true);

        const start = Date.now();
        await logTelemetryEvent("tengu_native_auto_updater_start", {});

        try {
            const channel = getAutoUpdatesChannel();
            const result = await runNativeAutoUpdate(channel);
            const currentVersion = PACKAGE_METADATA.VERSION;
            const elapsed = Date.now() - start;

            if (result.lockFailed) {
                await logTelemetryEvent("tengu_native_auto_updater_lock_contention", { latency_ms: elapsed });
                return;
            }

            setVersionInfo({ current: currentVersion, latest: result.latestVersion || undefined });

            if (result.wasUpdated) {
                await refreshStatsig();
                await logTelemetryEvent("tengu_native_auto_updater_success", { latency_ms: elapsed });
                onAutoUpdaterResult({ version: result.latestVersion ?? null, status: "success" });
            } else {
                await logTelemetryEvent("tengu_native_auto_updater_up_to_date", { latency_ms: elapsed });
            }
        } catch (error) {
            const elapsed = Date.now() - start;
            const message = error instanceof Error ? error.message : String(error);
            logError("auto-updater", error);
            const category = getUpdateErrorCategory(message);

            await logTelemetryEvent("tengu_native_auto_updater_fail", {
                latency_ms: elapsed,
                error_timeout: category === "timeout",
                error_checksum: category === "checksum_mismatch",
                error_not_found: category === "not_found",
                error_permission: category === "permission_denied",
                error_disk_full: category === "disk_full",
                error_npm: category === "npm_error",
                error_network: category === "network_error"
            });

            onAutoUpdaterResult({ version: null, status: "install_failed" });
        } finally {
            onChangeIsUpdating(false);
        }
    }, [isUpdating, onChangeIsUpdating, onAutoUpdaterResult]);

    useEffect(() => {
        if (!hasChecked.current) {
            hasChecked.current = true;
            checkForUpdates();
        }
    }, [checkForUpdates]);

    useInterval(checkForUpdates, AUTO_UPDATE_INTERVAL_MS);

    if (!autoUpdaterResult?.version && (!versionInfo.current || !versionInfo.latest)) return null;
    if (!autoUpdaterResult?.version && !isUpdating) return null;

    return (
        <Box flexDirection="row" gap={1}>
            {verbose && (
                <Text dimColor>
                    current: {versionInfo.current} · latest: {versionInfo.latest}
                </Text>
            )}
            {isUpdating ? (
                <Box>
                    <Text dimColor wrap="end">Checking for updates</Text>
                </Box>
            ) : (
                <>
                    {autoUpdaterResult?.status === "success" && showSuccessMessage && (
                        <Text color="success">✓ Update installed · Restart to update</Text>
                    )}
                    {autoUpdaterResult?.status === "install_failed" && (
                        <Text color="error">
                            ✗ Auto-update failed · Try <Text bold>/doctor</Text>
                        </Text>
                    )}
                </>
            )}
        </Box>
    );
}
