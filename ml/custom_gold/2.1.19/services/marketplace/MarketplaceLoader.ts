import * as fs from 'node:fs';
import * as path from 'node:path';
import { PluginManager, PluginInstallation } from '../mcp/PluginManager.js';

// Official marketplace constants
const OFFICIAL_MARKETPLACE_REPO = 'anthropics/claude-plugins-official';
const OFFICIAL_MARKETPLACE_ID = 'plugin:claude-plugins-official';

export interface LoadedMarketplace {
    name: string;
    config: any;
    status: 'loaded' | 'failed';
}

export interface MarketplaceFailure {
    name: string;
    error: string;
}

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
            let marketplacePlugin = plugins.find((p: PluginInstallation) => p.id === OFFICIAL_MARKETPLACE_ID);

            // 2. If not installed, auto-install it. If installed, update it occasionally.
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
                marketplacePlugin = plugins.find((p: PluginInstallation) => p.id === OFFICIAL_MARKETPLACE_ID);
            } else {
                // Occasional update check (e.g. once per session or every 24h)
                // For now, let's just trigger a background update
                if (process.env.CLAUDE_MARKETPLACE_AUTO_UPDATE !== 'false') {
                    PluginManager.updatePlugin(OFFICIAL_MARKETPLACE_ID, 'user').catch(err => {
                        console.warn("[MarketplaceLoader] Background update of official marketplace failed:", err.message);
                    });
                }
            }

            if (!marketplacePlugin || !marketplacePlugin.installPath) {
                console.error("[MarketplaceLoader] Failed to locate installed marketplace plugin.");
                return [];
            }

            // 3. Read marketplace.json or plugins.json
            const searchPaths = [
                marketplacePlugin.installPath,
                path.join(marketplacePlugin.installPath, '.claude'),
                path.join(marketplacePlugin.installPath, 'dist')
            ];

            const fileNames = ['marketplace.json', 'plugins.json'];
            let marketplaceJsonPath: string | null = null;

            for (const dir of searchPaths) {
                if (!fs.existsSync(dir)) continue;
                for (const name of fileNames) {
                    const p = path.join(dir, name);
                    if (fs.existsSync(p)) {
                        marketplaceJsonPath = p;
                        break;
                    }
                }
                if (marketplaceJsonPath) break;
            }

            // Still not found? Try recursive search (limited depth)
            if (!marketplaceJsonPath) {
                const findManifest = (dir: string, depth = 0): string | null => {
                    if (depth > 2) return null;
                    const files = fs.readdirSync(dir);
                    for (const file of files) {
                        const fullPath = path.join(dir, file);
                        const stat = fs.statSync(fullPath);
                        if (stat.isDirectory()) {
                            const found = findManifest(fullPath, depth + 1);
                            if (found) return found;
                        } else if (fileNames.includes(file)) {
                            return fullPath;
                        }
                    }
                    return null;
                };
                try {
                    marketplaceJsonPath = findManifest(marketplacePlugin.installPath);
                } catch { }
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
