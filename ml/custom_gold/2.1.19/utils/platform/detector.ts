/**
 * File: src/utils/platform/detector.ts
 * Role: Platform detection utilities (OS, WSL, etc.)
 */

import { release } from "node:os";
import { readFileSync, existsSync } from "node:fs";

/**
 * Gets the current platform name.
 * 
 * @returns {string} Platform identifier (e.g., 'darwin', 'windows', 'linux').
 */
export function getPlatform(): string {
    if (process.platform === 'win32') {
        return 'windows';
    }
    return process.platform;
}

/**
 * Checks if the current environment is Windows Subsystem for Linux (WSL).
 * 
 * @returns {boolean} True if running under WSL.
 */
export function isWsl(): boolean {
    if (process.platform !== 'linux') {
        return false;
    }

    try {
        const rel = release().toLowerCase();
        if (rel.includes('microsoft')) {
            return true;
        }

        // Fallback: check /proc/version
        if (existsSync('/proc/version')) {
            const procVersion = readFileSync('/proc/version', 'utf8').toLowerCase();
            return procVersion.includes('microsoft') || procVersion.includes('wsl');
        }
    } catch {
        // Ignore errors in detection
    }

    return false;
}

/**
 * Checks if the current environment is using musl libc (common in Alpine Linux).
 * 
 * @returns {boolean} True if musl is detected.
 */
export function isMuslEnvironment(): boolean {
    if (process.platform !== 'linux') {
        return false;
    }

    try {
        // Check for common musl library paths
        if (existsSync('/lib/libc.musl-x86_64.so.1') || existsSync('/lib/libc.musl-aarch64.so.1')) {
            return true;
        }

        // Fallback: check ldd output
        const { execSync } = require('node:child_process');
        const output = execSync('ldd /bin/ls 2>/dev/null').toString();
        return output.includes('musl');
    } catch {
        return false;
    }
}

/**
 * Gets a hyphenated platform-architecture string (e.g., 'darwin-arm64').
 * 
 * @returns {string} The platform-architecture string.
 */
export function getPlatArch(): string {
    const platform = getPlatform();
    const arch = process.arch === 'x64' ? 'x64' : process.arch === 'arm64' ? 'arm64' : null;

    if (!arch) {
        throw new Error(`Unsupported architecture: ${process.arch}`);
    }

    if (platform === 'linux' && isMuslEnvironment()) {
        return `linux-${arch}-musl`;
    }

    return `${platform}-${arch}`;
}
