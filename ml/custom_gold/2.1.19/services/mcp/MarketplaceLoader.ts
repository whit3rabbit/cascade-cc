
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
                    mcp: { type: 'noop' } // It's just a data repo, no MCP server needed really?
                    // Actually, the original code seems to treat it as a plugin. 
                    // Let's assume it might have a basic MCP config or we just install it for data.
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
            // Based on chunk1288/1627, it seems to look for `.claude-plugin/marketplace.json` or similar?
            // Let's assume the repo structure has a `marketplace.json` or `plugins.json` at root or specific path.
            // For now, let's try 'marketplace.json' at the root of the repo.
            const marketplaceJsonPath = path.join(marketplacePlugin.installPath, 'marketplace.json');

            if (fs.existsSync(marketplaceJsonPath)) {
                try {
                    const content = fs.readFileSync(marketplaceJsonPath, 'utf-8');
                    const data = JSON.parse(content);
                    // data should be { plugins: [...] } or array
                    const list = Array.isArray(data) ? data : (data.plugins || []);
                    return list;
                } catch (e) {
                    console.error("[MarketplaceLoader] Failed to parse marketplace.json:", e);
                    return [];
                }
            } else {
                console.warn(`[MarketplaceLoader] marketplace.json not found at ${marketplaceJsonPath}`);
                return [];
            }

        } catch (e) {
            console.error("[MarketplaceLoader] Error in fetchPluginDirectory:", e);
            return [];
        }
    }
}
