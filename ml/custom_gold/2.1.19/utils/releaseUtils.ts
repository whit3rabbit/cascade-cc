/**
 * File: src/utils/releaseUtils.ts
 * Role: Utilities for managing and downloading Claude Code releases.
 */

import axios from 'axios';
import { createHash } from 'node:crypto';
import { writeFileSync, chmodSync, rmSync, mkdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { track } from '../services/telemetry/Telemetry.js';
import { getCurrentHub } from '../services/telemetry/SentryHub.js';
import { getPlatArch } from './platform/detector.js';

export const RELEASES_URL = "https://storage.googleapis.com/claude-code-dist-86c565f3-f756-42ad-8dfa-d59b1c096819/claude-code-releases";
export const DOWNLOAD_TIMEOUT_MS = 60000;
export const MAX_RETRIES = 3;

export interface DownloadOptions {
    headers?: Record<string, string>;
    [key: string]: any;
}

export interface ManifestPlatform {
    checksum: string;
}

export interface ReleaseManifest {
    platforms: Record<string, ManifestPlatform>;
}

/**
 * Error thrown when a download stalls for too long.
 */
export class StallTimeoutError extends Error {
    constructor() {
        super(`Download stalled: no data received for ${DOWNLOAD_TIMEOUT_MS / 1000} seconds`);
        this.name = "StallTimeoutError";
    }
}

/**
 * Normalizes a version string.
 */
export function normalizeVersion(version: string): string {
    return version.replace(/^v/, "").trim();
}

/**
 * Fetches the version string for a given channel (e.g., 'latest', 'stable').
 */
export async function fetchVersionFromChannel(channel: string = "latest", baseUrl: string = RELEASES_URL): Promise<string> {
    const start = Date.now();
    try {
        const response = await axios.get(`${baseUrl}/${channel}`, {
            timeout: 30000,
            responseType: "text",
        });
        const latency = Date.now() - start;
        track("tengu_version_check_success", { latency_ms: latency });
        return response.data.trim();
    } catch (err) {
        const latency = Date.now() - start;
        const message = err instanceof Error ? err.message : String(err);
        let status: number | undefined;
        if (axios.isAxiosError(err) && err.response) {
            status = err.response.status;
        }
        track("tengu_version_check_failure", {
            latency_ms: latency,
            http_status: status,
            is_timeout: message.includes("timeout")
        });
        const error = new Error(`Failed to fetch version from ${baseUrl}/${channel}: ${message}`);
        getCurrentHub().captureException(error);
        throw error;
    }
}

/**
 * Resolves a version alias or normalizes a version string.
 */
export async function resolveVersion(versionOrChannel: string): Promise<string> {
    if (/^v?\d+\.\d+\.\d+(-\S+)?$/.test(versionOrChannel)) {
        return normalizeVersion(versionOrChannel);
    }
    if (versionOrChannel !== "stable" && versionOrChannel !== "latest") {
        throw new Error(`Invalid channel: ${versionOrChannel}. Use 'stable' or 'latest'`);
    }
    return fetchVersionFromChannel(versionOrChannel);
}

/**
 * Downloads a file and verifies its checksum.
 */
export async function downloadAndVerify(url: string, checksum: string, outputPath: string, options: DownloadOptions = {}): Promise<void> {
    let lastError: Error | undefined;
    for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
        const controller = new AbortController();
        let stallTimeout: NodeJS.Timeout | undefined;

        const clearStallTimeout = () => {
            if (stallTimeout) {
                clearTimeout(stallTimeout);
                stallTimeout = undefined;
            }
        };

        const resetStallTimeout = () => {
            clearStallTimeout();
            stallTimeout = setTimeout(() => {
                controller.abort();
            }, DOWNLOAD_TIMEOUT_MS);
        };

        try {
            resetStallTimeout();
            const response = await axios.get(url, {
                timeout: 300000,
                responseType: "arraybuffer",
                signal: controller.signal,
                onDownloadProgress: () => resetStallTimeout(),
                ...options
            });
            clearStallTimeout();

            const hash = createHash("sha256");
            hash.update(Buffer.from(response.data));
            const actualChecksum = hash.digest("hex");

            if (actualChecksum !== checksum) {
                throw new Error(`Checksum mismatch: expected ${checksum}, got ${actualChecksum}`);
            }

            writeFileSync(outputPath, Buffer.from(response.data));
            chmodSync(outputPath, 0o755); // 493 in octal is 755
            return;
        } catch (err) {
            clearStallTimeout();
            if (axios.isCancel(err)) {
                lastError = new StallTimeoutError();
            } else {
                lastError = err instanceof Error ? err : new Error(String(err));
            }

            if (axios.isCancel(err) && attempt < MAX_RETRIES) {
                console.log(`Download stalled on attempt ${attempt}/${MAX_RETRIES}, retrying...`);
                await new Promise(resolve => setTimeout(resolve, 1000));
                continue;
            }
            throw lastError;
        }
    }
    throw lastError ?? new Error("Download failed after all retries");
}

/**
 * Downloads a binary for the current platform.
 */
export async function downloadBinary(version: string, targetDir: string, baseUrl: string = RELEASES_URL, options: DownloadOptions = {}): Promise<void> {
    if (existsSync(targetDir)) {
        rmSync(targetDir, { recursive: !0, force: !0 });
    }

    const platform = getPlatArch();
    const start = Date.now();
    track("tengu_binary_download_attempt", {});

    let manifest: ReleaseManifest;
    try {
        const response = await axios.get(`${baseUrl}/${version}/manifest.json`, {
            timeout: 10000,
            responseType: "json",
            ...options
        });
        manifest = response.data;
    } catch (err) {
        const latency = Date.now() - start;
        const message = err instanceof Error ? err.message : String(err);
        let status: number | undefined;
        if (axios.isAxiosError(err) && err.response) {
            status = err.response.status;
        }
        track("tengu_binary_manifest_fetch_failure", {
            latency_ms: latency,
            http_status: status,
            is_timeout: message.includes("timeout")
        });
        throw new Error(`Failed to fetch manifest from ${baseUrl}/${version}/manifest.json: ${message}`);
    }

    const platformData = manifest.platforms[platform];
    if (!platformData) {
        track("tengu_binary_platform_not_found", {});
        throw new Error(`Platform ${platform} not found in manifest for version ${version}`);
    }

    const checksum = platformData.checksum;
    const executableName = platform.startsWith("windows") ? "claude.exe" : "claude";
    const downloadUrl = `${baseUrl}/${version}/${platform}/${executableName}`;

    mkdirSync(targetDir, { recursive: true });
    const targetPath = join(targetDir, executableName);

    try {
        await downloadAndVerify(downloadUrl, checksum, targetPath, options);
        const latency = Date.now() - start;
        track("tengu_binary_download_success", { latency_ms: latency });
    } catch (err) {
        const latency = Date.now() - start;
        const message = err instanceof Error ? err.message : String(err);
        let status: number | undefined;
        if (axios.isAxiosError(err) && err.response) {
            status = err.response.status;
        }
        track("tengu_binary_download_failure", {
            latency_ms: latency,
            http_status: status,
            is_timeout: message.includes("timeout"),
            is_checksum_mismatch: message.includes("Checksum mismatch")
        });
        throw err;
    }
}
