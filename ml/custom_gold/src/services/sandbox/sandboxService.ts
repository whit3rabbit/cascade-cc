import {
    sandbox,
    SandboxConfig
} from "./sandboxApi.js";
import { getSettings, mergeSettings, subscribeToSettings } from "../terminal/settings.js";
import { getSandboxConfigFromSettings } from "./sandboxConfigGenerator.js";
import { getProjectRoot } from "../../utils/shared/pathUtils.js";

/**
 * High-level sandbox service.
 * Deobfuscated from _B in chunk_224.ts.
 */

let initializationPromise: Promise<void> | undefined;
let settingsSubscription: (() => void) | undefined;

/**
 * Checks if sandboxing is currently enabled in settings and supported by the platform.
 * Deobfuscated from iA1 in chunk_224.ts.
 */
export function isSandboxingEnabled(): boolean {
    if (!sandbox.isSupportedPlatform()) return false;
    if (!sandbox.checkDependencies()) return false;

    const settings = mergeSettings();
    return settings?.sandbox?.enabled ?? false;
}

/**
 * Initializes the sandbox service with current settings.
 * Deobfuscated from Q63 in chunk_224.ts.
 */
export async function initSandboxService(
    onPermissionAsk?: (req: { host: string, port: number }) => Promise<boolean>
): Promise<void> {
    if (initializationPromise) return initializationPromise;
    if (!isSandboxingEnabled()) return;

    const refreshConfig = () => {
        const settings = mergeSettings();
        const config = getSandboxConfigFromSettings(
            settings,
            getProjectRoot()
        );
        sandbox.updateConfig(config);
    };

    initializationPromise = (async () => {
        try {
            const settings = mergeSettings();
            const config = getSandboxConfigFromSettings(
                settings,
                getProjectRoot()
            );

            await sandbox.initialize(config, onPermissionAsk);

            // Watch for settings changes to update config dynamically
            settingsSubscription = subscribeToSettings(() => {
                refreshConfig();
            });
        } catch (err) {
            initializationPromise = undefined;
            throw err;
        }
    })();

    return initializationPromise;
}

/**
 * Convenience wrapper for executing a command in the sandbox.
 * Deobfuscated from A63 in chunk_224.ts.
 */
export async function executeInSandbox(
    command: string,
    options: { binShell?: string, abortSignal?: AbortSignal } = {}
): Promise<string> {
    await initSandboxService();
    return sandbox.wrapWithSandbox(command, options);
}

/**
 * Resets the sandbox service and cleans up infrastructure.
 * Deobfuscated from G63 in chunk_224.ts.
 */
export async function resetSandboxService() {
    if (settingsSubscription) {
        settingsSubscription();
        settingsSubscription = undefined;
    }
    initializationPromise = undefined;
    await sandbox.reset();
}

/**
 * Exposes the sandbox configuration and status utilities.
 */
export const sandboxService = {
    initialize: initSandboxService,
    isEnabled: isSandboxingEnabled,
    execute: executeInSandbox,
    reset: resetSandboxService,

    // Re-exports from primitive sandbox API
    getViolationStore: sandbox.getSandboxViolationStore,
    annotateStderr: sandbox.annotateStderrWithSandboxFailures,

    // Settings-specific helpers
    isAutoAllowBashEnabled: () => mergeSettings()?.sandbox?.autoAllowBashIfSandboxed ?? true,
    isUnsandboxedAllowed: () => mergeSettings()?.sandbox?.allowUnsandboxedCommands ?? true,
    getExcludedCommands: () => mergeSettings()?.sandbox?.excludedCommands ?? []
};
