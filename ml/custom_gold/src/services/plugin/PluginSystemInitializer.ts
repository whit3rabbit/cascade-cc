import * as fs from 'node:fs';
import * as path from 'node:path';
import * as os from 'node:os';
import { getSettings, updateSettings } from '../terminal/settings.js';
import { log } from '../logger/loggerService.js';
import {
    getPluginStore,
    recordInstallation,
    getPluginInstallPath,
} from '../mcp/McpClientManager.js';
import {
    getPluginFromMarketplace,
} from '../mcp/PluginManager.js';
import { getSettingsScope, getProjectPathForScope } from '../mcp/PluginUtils.js';
import { getVersion } from '../../utils/version.js';

const logger = log('PluginSystemInitializer');

/**
 * Syncs the installed_plugins.json with enabled plugins in settings.
 * Based on Z70 in chunk_342.ts.
 */
export async function initializePluginSystem() {
    logger.info("Initializing plugin system...");

    const store = getPluginStore();
    const settingsScopes = ['userSettings', 'projectSettings', 'localSettings'] as const;

    // 1. Gather all enabled plugins from all settings files
    const enabledPluginsMap = new Map<string, { scope: string; projectPath?: string }>();

    for (const settingsScope of settingsScopes) {
        const settings = getSettings(settingsScope);
        if (settings?.enabledPlugins) {
            for (const [pluginId, isEnabled] of Object.entries(settings.enabledPlugins)) {
                if (isEnabled && pluginId.includes('@')) {
                    // Map settings scope back to plugin scope
                    let scope = 'user';
                    if (settingsScope === 'projectSettings') scope = 'project';
                    else if (settingsScope === 'localSettings') scope = 'local';

                    enabledPluginsMap.set(pluginId, {
                        scope,
                        projectPath: getProjectPathForScope(scope)
                    });
                }
            }
        }
    }

    if (enabledPluginsMap.size === 0) {
        logger.info("No enabled plugins found, skipping sync.");
        return;
    }

    // 2. Check if we need to migrate/add any missing plugins
    let needsUpdate = false;
    const now = new Date().toISOString();

    for (const [pluginId, ctx] of enabledPluginsMap) {
        const installations = store.plugins[pluginId];
        const existing = installations?.find((i: any) => i.scope === ctx.scope && i.projectPath === ctx.projectPath);

        if (!existing) {
            logger.info(`Missing metadata for enabled plugin ${pluginId}, attempting to restore...`);
            try {
                const marketplaceEntry = await getPluginFromMarketplace(pluginId);
                if (!marketplaceEntry) {
                    logger.warn(`Plugin ${pluginId} not found in any marketplace, skipping migration.`);
                    continue;
                }

                const { entry } = marketplaceEntry;
                const version = entry.version || "1.0.0";
                const installPath = getPluginInstallPath(pluginId, version);

                // Check if it's already on disk (maybe from a previous installation type)
                if (fs.existsSync(installPath)) {
                    recordInstallation(pluginId, ctx.scope, ctx.projectPath || "", installPath, version);
                    needsUpdate = true;
                    logger.info(`Restored metadata for ${pluginId} at ${installPath}`);
                } else {
                    logger.warn(`Plugin ${pluginId} is enabled but its install path ${installPath} does not exist. It may need re-installation.`);
                }
            } catch (err) {
                logger.error(`Failed to migrate plugin ${pluginId}`, err);
            }
        }
    }

    logger.info("Plugin system initialization complete.");
}
