
import { getSettings, updateSettings } from '../config/SettingsService.js';
import { PluginManager } from './PluginManager.js';

export class MarketplaceService {
    static async checkAndInstallOfficialPlugins(): Promise<void> {
        const settings = getSettings();

        // Skip if explicitly disabled
        if (settings.officialMarketplaceAutoInstall === false) {
            console.log("[Marketplace] Official marketplace auto-install disabled.");
            return;
        }

        // Skip if already attempted and successful, or if we should skip due to a previous failure
        if (settings.officialMarketplaceAutoInstallAttempted && settings.officialMarketplaceAutoInstalled) {
            return;
        }

        // Researching 2.1.19 reveals they track fail reasons like "git_unavailable"
        // and have a retry policy. For now, let's implement basic status tracking.
        if (settings.officialMarketplaceAutoInstallAttempted && settings.officialMarketplaceAutoInstallFailReason === "git_unavailable") {
            // Check if we should retry (e.g., if more than 24h passed)
            const lastAttempt = settings.officialMarketplaceAutoInstallLastAttemptTime || 0;
            const now = Date.now();
            if (now - lastAttempt < 24 * 60 * 60 * 1000) {
                console.log("[Marketplace] Skipping official marketplace auto-install: Git was previously unavailable.");
                return;
            }
        }

        const officialRepo = {
            source: "github",
            repository: "anthropics/claude-plugins-official",
            name: "claude-plugins-official",
            description: "Official Claude Code plugins"
        };

        try {
            console.log("[Marketplace] Attempting to auto-install official marketplace");

            // Check for git availability if needed by PluginManager/git.ts
            // PluginManager.installPlugin will throw if git is missing.

            const result = await PluginManager.installPlugin(officialRepo, 'user');

            updateSettings((current) => ({
                ...current,
                officialMarketplaceAutoInstallAttempted: true,
                officialMarketplaceAutoInstalled: result.success,
                officialMarketplaceAutoInstallFailReason: result.success ? undefined : "unknown",
                officialMarketplaceAutoInstallLastAttemptTime: Date.now()
            }));

            if (result.success) {
                console.log("[Marketplace] Successfully auto-installed official marketplace");
            } else {
                console.warn(`[Marketplace] Official marketplace auto-install skipped: ${result.message}`);
            }
        } catch (e: any) {
            const reason = e.message?.includes('git') ? "git_unavailable" : "unknown";
            console.error(`[Marketplace] Failed to auto-install official marketplace: ${e.message}`);

            updateSettings((current) => ({
                ...current,
                officialMarketplaceAutoInstallAttempted: true,
                officialMarketplaceAutoInstalled: false,
                officialMarketplaceAutoInstallFailReason: reason,
                officialMarketplaceAutoInstallLastAttemptTime: Date.now()
            }));
        }
    }
}
