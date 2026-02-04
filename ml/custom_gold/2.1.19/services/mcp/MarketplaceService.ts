
import { getSettings } from '../config/SettingsService.js';
import { PluginManager } from './PluginManager.js';

export class MarketplaceService {
    static async checkAndInstallOfficialPlugins(): Promise<void> {
        const settings = getSettings();
        // Check if explicitly disabled
        if (settings.officialMarketplaceAutoInstall === false) {
            console.log("[Marketplace] Official marketplace auto-install disabled.");
            return;
        }

        const officialRepo = {
            source: "github",
            repository: "anthropics/claude-plugins-official",
            name: "claude-plugins-official",
            description: "Official Claude Code plugins"
        };

        // We use PluginManager to handle installation.
        // It will clone if missing, or update if present (although current updatePlugin logic is separate)
        // installPlugin in our implementation handles "already installed" by returning success false but that's fine.
        // But better: checks installed plugins first.

        try {
            const installed = await PluginManager.getInstalledPlugins();
            const officialPlugin = installed.find(p =>
                p.id === 'plugin:claude-plugins-official' ||
                (p.name === 'claude-plugins-official' && p.scope === 'user')
                // Or check repository if we stored it? We stored repo in McpServerManager registry but maybe not in simple list?
                // Our PluginManager stores it in config. 
            );

            if (officialPlugin) {
                // Already installed. Maybe update?
                // chunk1627 implies it runs on startup, so maybe it updates?
                // "Attempting to auto-install official marketplace"
                // If it's just install, PluginManager.installPlugin handles existence check.
                // But we might want to ensure it's updated.
                // For now, let's just try to install.
                await PluginManager.installPlugin(officialRepo, 'user');
            } else {
                console.log("[Marketplace] Installing official plugins...");
                await PluginManager.installPlugin(officialRepo, 'user');
            }
        } catch (e) {
            console.warn("[Marketplace] Failed to auto-install official plugins:", e);
        }
    }
}
