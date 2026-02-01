import { LoadedMarketplace, MarketplaceFailure } from './MarketplaceLoader.js';
import { getSettings, updateSettings } from '../config/SettingsService.js';

export interface MarketplaceServiceType {
    addMarketplace: (name: string, url: string) => Promise<void>;
    removeMarketplace: (name: string) => Promise<void>;
    refreshMarketplace: (name: string, onProgress?: (msg: string) => void) => Promise<void>;
    refreshAllMarketplaces: (onProgress?: (msg: string) => void) => Promise<void>;
    listMarketplaces: () => Promise<LoadedMarketplace[]>;
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
    }
};
