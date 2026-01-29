/**
 * File: src/services/marketplace/MarketplaceLoader.ts
 * Role: Logic for discovering and loading external marketplaces for plugins.
 */

export interface MarketplaceConfig {
    [key: string]: any;
}

export interface LoadedMarketplace {
    name: string;
    config: any;
    status: 'loaded' | 'failed';
}

export interface MarketplaceFailure {
    name: string;
    error: string;
}

export interface MarketplaceLoadResult {
    loaded: LoadedMarketplace[];
    failures: MarketplaceFailure[];
}

/**
 * Orchestrates marketplace loading from a config map.
 * 
 * @param {Record<string, any>} configs - Map of marketplace names to configurations.
 * @returns {Promise<MarketplaceLoadResult>}
 */
export async function loadMarketplaces(configs: Record<string, any>): Promise<MarketplaceLoadResult> {
    const loaded: LoadedMarketplace[] = [];
    const failures: MarketplaceFailure[] = [];

    for (const [name, config] of Object.entries(configs)) {
        try {
            // Logic to fetch marketplace metadata would go here
            loaded.push({ name, config, status: 'loaded' });
        } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            failures.push({ name, error: message });
        }
    }

    return { loaded, failures };
}

/**
 * Formats marketplace loading errors for display.
 * 
 * @param {MarketplaceFailure[]} failures - List of failures.
 * @returns {string | null}
 */
export function formatMarketplaceFailures(failures: MarketplaceFailure[]): string | null {
    if (failures.length === 0) return null;
    if (failures.length === 1) return `Failed to load marketplace '${failures[0].name}': ${failures[0].error}`;

    const names = failures.map(f => f.name).join(', ');
    return `Failed to load ${failures.length} marketplaces: ${names}`;
}
