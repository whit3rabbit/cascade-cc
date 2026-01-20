import {
    initializeSandbox,
    wrapCommand,
    cleanupSandbox,
    updateSandboxConfig,
    violationManager,
    SandboxConfig
} from "./sandboxManager.js";
import { isLinuxSandboxAvailable } from "./linuxSandbox.js";
import { isRipgrepAvailable } from "../../utils/shared/ripgrep.js";

/**
 * Public Sandbox API.
 * Deobfuscated from pJ in chunk_222.ts.
 */
export const sandbox = {
    /**
     * Initializes the sandbox infrastructure (proxies, bridges).
     */
    initialize: initializeSandbox,

    /**
     * Wraps a shell command with platform-specific sandboxing (bwrap or sandbox-exec).
     */
    wrapWithSandbox: wrapCommand,

    /**
     * Cleans up all sandbox-related processes and sockets.
     */
    reset: cleanupSandbox,

    /**
     * Checks if the current platform supports sandboxing.
     */
    isSupportedPlatform: (): boolean => ["darwin", "linux"].includes(process.platform),

    /**
     * Checks if mandatory dependencies are installed.
     */
    checkDependencies: (): boolean => {
        const platform = process.platform;
        if (platform === "linux") return isLinuxSandboxAvailable();
        if (platform === "darwin") return isRipgrepAvailable();
        return false;
    },

    /**
     * Access to the sandbox violation store.
     */
    getSandboxViolationStore: () => violationManager,

    /**
     * Updates the sandbox configuration.
     */
    updateConfig: (config: SandboxConfig) => updateSandboxConfig(config),

    /**
     * Appends sandbox violations to a string (e.g., stderr).
     * Deobfuscated from k93 in chunk_222.ts.
     */
    annotateStderrWithSandboxFailures: (command: string, stderr: string): string => {
        const violations = violationManager.getViolationsForCommand(command);
        if (violations.length === 0) return stderr;

        let result = stderr;
        result += "\n<sandbox_violations>\n";
        for (const v of violations) {
            result += v.line + "\n";
        }
        result += "</sandbox_violations>";
        return result;
    }
};

export type { SandboxConfig };
