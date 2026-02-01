/**
 * File: src/utils/browser.ts
 * Role: Aggregated utilities for environment, IDE detection, screenshots, and plugin management.
 */

export * from "./shared/screenshotUtils.js";
export * from "./fs/pathSanitizer.js";
export * from "../services/ide/ideDetection.js";
export * from "./shared/fileTypeUtils.js";
export * from "./fs/binaryUtils.js";

import { track } from "../services/telemetry/Telemetry.js";
import { installPlugin, uninstallPlugin, enablePlugin, disablePlugin, updatePlugin } from "../services/mcp/PluginManager.js";

type PluginActionResult = {
    success: boolean;
    message: string;
    pluginId?: string;
    alreadyUpToDate?: boolean;
    oldVersion?: string;
    newVersion?: string;
};

/**
 * Logs an error and exits the process.
 */
export function handleCliErrorAndExit(error: any, action: string): never {
    console.error(`✖ Failed to ${action}: ${error instanceof Error ? error.message : String(error)}`);
    process.exit(1);
}

/**
 * Installs a plugin via CLI with logging.
 */
export async function installPluginCli(pluginId: string, scope: string = "user") {
    try {
        console.log(`Installing plugin "${pluginId}"...`);
        const result = await installPlugin(pluginId, scope as any) as PluginActionResult;
        if (!result.success) {
            throw new Error(result.message);
        }
        console.log(`✔ ${result.message}`);
        track("tengu_plugin_installed_cli", {
            plugin_id: result.pluginId || pluginId,
            marketplace_name: (result.pluginId || pluginId)?.split("@")[1] || "unknown",
            scope: scope,
        });
        process.exit(0);
    } catch (error) {
        handleCliErrorAndExit(error, `install plugin "${pluginId}"`);
    }
}

/**
 * Uninstalls a plugin via CLI.
 */
export async function uninstallPluginCli(pluginId: string, scope: string = "user") {
    try {
        const result = await uninstallPlugin(pluginId, scope) as PluginActionResult;
        if (!result.success) {
            throw new Error(result.message);
        }
        console.log(`✔ ${result.message}`);
        track("tengu_plugin_uninstalled_cli", {
            plugin_id: result.pluginId || pluginId,
            scope: scope,
        });
        process.exit(0);
    } catch (error) {
        handleCliErrorAndExit(error, `uninstall plugin "${pluginId}"`);
    }
}

/**
 * Enables a plugin via CLI.
 */
export async function enablePluginCli(pluginId: string, scope: string) {
    try {
        const result = await enablePlugin(pluginId, scope) as PluginActionResult;
        if (!result.success) {
            throw new Error(result.message);
        }
        console.log(`✔ ${result.message}`);
        track("tengu_plugin_enabled_cli", {
            plugin_id: result.pluginId || pluginId,
            scope: scope,
        });
        process.exit(0);
    } catch (error) {
        handleCliErrorAndExit(error, `enable plugin "${pluginId}"`);
    }
}

/**
 * Disables a plugin via CLI.
 */
export async function disablePluginCli(pluginId: string, scope: string) {
    try {
        const result = await disablePlugin(pluginId, scope) as PluginActionResult;
        if (!result.success) {
            throw new Error(result.message);
        }
        console.log(`✔ ${result.message}`);
        track("tengu_plugin_disabled_cli", {
            plugin_id: result.pluginId || pluginId,
            scope: scope,
        });
        process.exit(0);
    } catch (error) {
        handleCliErrorAndExit(error, `disable plugin "${pluginId}"`);
    }
}

/**
 * Updates a plugin via CLI.
 */
export async function updatePluginCli(pluginId: string, scope: string) {
    try {
        console.log(`Checking for updates for plugin "${pluginId}" at ${scope} scope...`);
        const result = await updatePlugin(pluginId, scope) as PluginActionResult;
        if (!result.success) {
            throw new Error(result.message);
        }
        console.log(`✔ ${result.message}\n`);
        if (!result.alreadyUpToDate) {
            track("tengu_plugin_updated_cli", {
                plugin_id: pluginId,
                old_version: result.oldVersion || "unknown",
                new_version: result.newVersion || "unknown",
            });
        }
        process.exit(0);
    } catch (error) {
        handleCliErrorAndExit(error, `update plugin "${pluginId}"`);
    }
}
