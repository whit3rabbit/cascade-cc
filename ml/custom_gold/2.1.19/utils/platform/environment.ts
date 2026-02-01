/**
 * File: src/utils/platform/environment.ts
 * Role: Environment Detection Utilities for containers, sandboxes, and terminal emulators.
 */

import * as fs from 'node:fs';
import { execSync } from 'node:child_process';
import { getPlatform } from './detector.js';

/**
 * Checks if the current environment is running inside a Docker container.
 */
export function isDocker(): boolean {
    try {
        if (fs.existsSync('/.dockerenv')) {
            return true;
        }
        // Check cgroup for docker indicators
        if (fs.existsSync('/proc/self/cgroup')) {
            const cgroup = fs.readFileSync('/proc/self/cgroup', 'utf8');
            return cgroup.includes('docker');
        }
        return false;
    } catch {
        return false;
    }
}

import { EnvService } from '../../services/config/EnvService.js';

/**
 * Checks if the environment is a Bubblewrap sandbox.
 */
export function isBubblewrapSandbox(): boolean {
    return process.platform === "linux" && EnvService.get("CLAUDE_CODE_BUBBLEWRAP") === "1";
}

/**
 * Checks if the system uses musl libc (common in Alpine Linux).
 */
export function isMusl(): boolean {
    if (process.platform !== "linux") {
        return false;
    }

    try {
        // Common musl library paths
        if (fs.existsSync("/lib/libc.musl-x86_64.so.1") || fs.existsSync("/lib/libc.musl-aarch64.so.1")) {
            return true;
        }

        // Check ldd output
        const lddOutput = execSync("ldd --version 2>/dev/null", { encoding: 'utf8' });
        return lddOutput.toLowerCase().includes("musl");
    } catch {
        return false;
    }
}

/**
 * Attempts to detect the current terminal emulator.
 */
export function getTerminalEmulator(): string | undefined {
    if (EnvService.get("TERMINAL_EMULATOR") === "JetBrains-JediTerm") {
        if (getPlatform() !== "darwin") {
            return detectJetBrainsProduct() || "jetbrains";
        }
    }
    return EnvService.get("TERM_PROGRAM") || EnvService.get("TERM");
}

/**
 * Internal helper to detect JetBrains product from environment or process.
 */
function detectJetBrainsProduct(): string | null {
    // Simplified detection; could be expanded to check process tree or specific JB env vars.
    if (EnvService.get("PYCHARM_VM_OPTIONS")) return "pycharm";
    if (EnvService.get("WEBIDE_VM_OPTIONS")) return "webstorm";
    if (EnvService.get("IDEA_VM_OPTIONS")) return "intellij";
    return null;
}
