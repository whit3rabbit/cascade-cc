import { LoadedMarketplace, MarketplaceFailure } from './MarketplaceLoader.js';
import { getSettings, updateSettings } from '../config/SettingsService.js';

export interface MarketplaceServiceType {
    addMarketplace: (name: string, url: string) => Promise<void>;
    removeMarketplace: (name: string) => Promise<void>;
    refreshMarketplace: (name: string, onProgress?: (msg: string) => void) => Promise<void>;
    refreshAllMarketplaces: (onProgress?: (msg: string) => void) => Promise<void>;
    listMarketplaces: () => Promise<LoadedMarketplace[]>;
    autoInstallOfficialMarketplace: () => Promise<void>;
}

import axios from 'axios';

/**
 * Service for managing external plugin marketplaces.
 */
export const MarketplaceService: MarketplaceServiceType = {
    /**
     * Adds a new marketplace source.
     */
    async addMarketplace(name: string, url: string): Promise<void> {
        updateSettings((current) => {
            const marketplaces = current.marketplaces || {};
            return {
                ...current,
                marketplaces: {
                    ...marketplaces,
                    [name]: { url }
                }
            };
        });
    },

    /**
     * Removes a marketplace source.
     */
    async removeMarketplace(name: string): Promise<void> {
        updateSettings((current) => {
            const marketplaces = current.marketplaces || {};
            const { [name]: removed, ...rest } = marketplaces;
            return {
                ...current,
                marketplaces: rest
            };
        });
    },

    /**
     * Refreshes a specific marketplace by fetching its marketplace.json.
     */
    async refreshMarketplace(name: string, onProgress?: (msg: string) => void): Promise<void> {
        if (onProgress) onProgress(`Refreshing ${name}...`);

        const settings = getSettings();
        const marketplaceConfig = settings.marketplaces?.[name];

        if (!marketplaceConfig || !marketplaceConfig.url) {
            throw new Error(`Marketplace '${name}' configuration missing or invalid URL`);
        }

        try {
            const response = await axios.get(marketplaceConfig.url, { timeout: 10000 });
            const data = response.data;

            // Simple validation
            if (!data.plugins || !Array.isArray(data.plugins)) {
                throw new Error("Invalid marketplace format: expected 'plugins' array");
            }

            // Cache the results in local settings or a dedicated cache file
            updateSettings((current) => {
                const marketplaces = current.marketplaces || {};
                return {
                    ...current,
                    marketplaces: {
                        ...marketplaces,
                        [name]: {
                            ...marketplaceConfig,
                            lastUpdated: new Date().toISOString(),
                            plugins: data.plugins
                        }
                    }
                };
            });

            if (onProgress) onProgress(`Marketplace ${name} refreshed with ${data.plugins.length} plugins.`);
        } catch (error: any) {
            const msg = error.response ? `HTTP ${error.response.status}` : error.message;
            throw new Error(`Failed to refresh ${name}: ${msg}`);
        }
    },

    /**
     * Refreshes all configured marketplaces.
     */
    async refreshAllMarketplaces(onProgress?: (msg: string) => void): Promise<void> {
        const settings = getSettings();
        const marketplaces = settings.marketplaces || {};
        const names = Object.keys(marketplaces);

        if (names.length === 0) {
            // If no marketplaces, maybe add a default one?
            // For now just warn
            if (onProgress) onProgress("No marketplaces configured.");
            return;
        }

        for (const name of names) {
            try {
                await this.refreshMarketplace(name, onProgress);
            } catch (err: any) {
                if (onProgress) onProgress(`Error refreshing ${name}: ${err.message}`);
            }
        }
    },

    /**
     * Lists all loaded marketplaces and their plugins.
     */
    async listMarketplaces(): Promise<LoadedMarketplace[]> {
        const settings = getSettings();
        const marketplaces = settings.marketplaces || {};

        return Object.entries(marketplaces).map(([name, config]: [string, any]) => ({
            name,
            config: config,
            status: config.plugins ? 'loaded' : 'failed'
        }));
    },

    /**
     * Attempts to auto-install the official marketplace.
     * Matches logic from chunk1627 (tengu_official_marketplace_auto_install).
     */
    async autoInstallOfficialMarketplace(): Promise<void> {
        const OFFICIAL_MARKETPLACE_NAME = "claude-plugins-official";
        const OFFICIAL_MARKETPLACE_REPO = "anthropics/claude-plugins-official";

        // 1. Check if disabled via env var
        if (process.env.CLAUDE_CODE_DISABLE_OFFICIAL_MARKETPLACE_AUTOINSTALL) {
            updateSettings(current => ({
                ...current,
                officialMarketplaceAutoInstallAttempted: true,
                officialMarketplaceAutoInstalled: false,
                officialMarketplaceAutoInstallFailReason: "policy_blocked"
            }));
            return;
        }

        const settings = getSettings();

        // 2. Check if already installed
        if (settings.marketplaces?.[OFFICIAL_MARKETPLACE_NAME]) {
            updateSettings(current => ({
                ...current,
                officialMarketplaceAutoInstallAttempted: true,
                officialMarketplaceAutoInstalled: true
            }));
            return;
        }

        // 3. Retry Logic & State Check
        if (settings.officialMarketplaceAutoInstallAttempted) {
            if (settings.officialMarketplaceAutoInstalled) return;

            const failReason = settings.officialMarketplaceAutoInstallFailReason;
            const retryCount = settings.officialMarketplaceAutoInstallRetryCount || 0;
            const nextRetry = settings.officialMarketplaceAutoInstallNextRetryTime;
            const now = Date.now();
            const MAX_ATTEMPTS = 5; // Hv1.MAX_ATTEMPTS from reference

            if (retryCount >= MAX_ATTEMPTS) return;
            if (failReason === 'policy_blocked') return;
            if (nextRetry && now < nextRetry) return; // Not time yet
            // If unknown or git_unavailable or undefined, we might retry
        }

        // 4. Check for Git availability (simplified check)
        // In the reference, it calls U31() -> likely checking if git is in PATH
        // We'll try to run `git --version`
        const { exec } = await import('child_process');
        const isGitAvailable = await new Promise<boolean>(resolve => {
            exec('git --version', (err) => resolve(!err));
        });

        if (!isGitAvailable) {
            const retryCount = (settings.officialMarketplaceAutoInstallRetryCount || 0) + 1;
            // Backoff: 60s * 2^retryCount, capped at 1 hour? Reference uses Hv1 constants.
            // Let's assume: INITIAL_DELAY 1min, MAX 1hr.
            const delay = Math.min(60000 * Math.pow(2, retryCount), 3600000);

            updateSettings(current => ({
                ...current,
                officialMarketplaceAutoInstallAttempted: true,
                officialMarketplaceAutoInstalled: false,
                officialMarketplaceAutoInstallFailReason: "git_unavailable",
                officialMarketplaceAutoInstallRetryCount: retryCount,
                officialMarketplaceAutoInstallLastAttemptTime: Date.now(),
                officialMarketplaceAutoInstallNextRetryTime: Date.now() + delay
            }));
            return;
        }

        // 5. Attempt Installation
        try {
            // We use addMarketplace which expects a URL.
            // The official marketplace in the reference (chunk1364) seems to point to a github repo.
            // "anthropics/claude-plugins-official"
            // Our addMarketplace implementation currently expects a URL (http/https). 
            // If we support github shortnames, we should convert it or handle it.
            // For now, let's assume raw github user content URL or similar if we strictly follow `addMarketplace(url)`.
            // HOWEVER, looking at chunk1364: `repository: "anthropics/claude-plugins-official"`.
            // And chunk1627 calls `My(dB6)` where dB6 is the config object.
            // If our `addMarketplace` only takes a URL, we might need to adjust or pass the raw JSON URL.
            // Let's use the raw githubusercontent URL for main branch as a safe bet for now.
            const rawUrl = `https://raw.githubusercontent.com/${OFFICIAL_MARKETPLACE_REPO}/main/marketplace.json`;

            await MarketplaceService.addMarketplace(OFFICIAL_MARKETPLACE_NAME, rawUrl);
            // Verify it was added and refresh
            await MarketplaceService.refreshMarketplace(OFFICIAL_MARKETPLACE_NAME);

            updateSettings(current => ({
                ...current,
                officialMarketplaceAutoInstallAttempted: true,
                officialMarketplaceAutoInstalled: true,
                officialMarketplaceAutoInstallFailReason: undefined,
                officialMarketplaceAutoInstallRetryCount: undefined,
                officialMarketplaceAutoInstallLastAttemptTime: undefined,
                officialMarketplaceAutoInstallNextRetryTime: undefined
            }));

        } catch (error: any) {
            const retryCount = (settings.officialMarketplaceAutoInstallRetryCount || 0) + 1;
            const delay = Math.min(60000 * Math.pow(2, retryCount), 3600000);

            updateSettings(current => ({
                ...current,
                officialMarketplaceAutoInstallAttempted: true,
                officialMarketplaceAutoInstalled: false,
                officialMarketplaceAutoInstallFailReason: "unknown", // or error message if short
                officialMarketplaceAutoInstallRetryCount: retryCount,
                officialMarketplaceAutoInstallLastAttemptTime: Date.now(),
                officialMarketplaceAutoInstallNextRetryTime: Date.now() + delay
            }));
            console.error("Failed to auto-install official marketplace:", error);
        }
    }
};
