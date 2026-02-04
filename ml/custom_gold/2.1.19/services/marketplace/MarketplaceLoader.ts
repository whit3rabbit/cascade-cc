
import * as fs from 'node:fs';
import * as path from 'node:path';
import { PluginManager } from './PluginManager.js';

// Official marketplace constants
const OFFICIAL_MARKETPLACE_REPO = 'anthropics/claude-plugins-official';
const OFFICIAL_MARKETPLACE_ID = 'plugin:claude-plugins-official';

/**
 * Service to load marketplace metadata.
 */
export class MarketplaceLoader {
    /**
     * Fetches the list of available plugins from the marketplace.
     */
    static async fetchPluginDirectory(): Promise<any[]> {
        console.log("[MarketplaceLoader] Fetching plugin directory...");

        try {
            // 1. Check if official marketplace is installed
            let plugins = await PluginManager.getInstalledPlugins();
            let marketplacePlugin = plugins.find(p => p.id === OFFICIAL_MARKETPLACE_ID);

            // 2. If not installed, auto-install it
            if (!marketplacePlugin) {
                console.log("[MarketplaceLoader] Official marketplace not found, auto-installing...");
                const result = await PluginManager.installPlugin({
                    source: 'github',
                    repository: OFFICIAL_MARKETPLACE_REPO,
                    name: 'claude-plugins-official',
                    version: '1.0.0', // Should come from repo ideally
                    mcp: { type: 'noop' } // It's just a data repo
                }, 'user');

                if (!result.success) {
                    console.error("[MarketplaceLoader] Failed to auto-install official marketplace:", result.message);
                    return [];
                }

                // Refresh list to get the install path
                plugins = await PluginManager.getInstalledPlugins();
                marketplacePlugin = plugins.find(p => p.id === OFFICIAL_MARKETPLACE_ID);
            }

            if (!marketplacePlugin || !marketplacePlugin.installPath) {
                console.error("[MarketplaceLoader] Failed to locate installed marketplace plugin.");
                return [];
            }

            // 3. Read marketplace.json or plugins.json
            const possiblePaths = [
                path.join(marketplacePlugin.installPath, 'marketplace.json'),
                path.join(marketplacePlugin.installPath, 'plugins.json'),
                path.join(marketplacePlugin.installPath, '.claude/marketplace.json')
            ];

            let marketplaceJsonPath: string | null = null;
            for (const p of possiblePaths) {
                if (fs.existsSync(p)) {
                    marketplaceJsonPath = p;
                    break;
                }
            }

            if (marketplaceJsonPath) {
                try {
                    const content = fs.readFileSync(marketplaceJsonPath, 'utf-8');
                    const data = JSON.parse(content);
                    const list = Array.isArray(data) ? data : (data.plugins || []);
                    return list;
                } catch (e) {
                    console.error("[MarketplaceLoader] Failed to parse marketplace data:", e);
                    return [];
                }
            } else {
                console.warn(`[MarketplaceLoader] No marketplace data found in ${marketplacePlugin.installPath}`);
                return [];
            }

        } catch (e) {
            console.error("[MarketplaceLoader] Error in fetchPluginDirectory:", e);
            return [];
        }
    }
}
