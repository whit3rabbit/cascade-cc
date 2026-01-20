import * as fs from "fs";
import * as path from "path";
import { readMarketplaceConfigFile, writeMarketplaceConfigFile, getMarketplacesCacheDir } from "./MarketplaceConfig.js";
import { downloadMarketplace, loadMarketplace } from "./MarketplaceLoader.js";
import { validateMarketplaceName, isReservedName } from "../../utils/validation/MarketplaceValidators.js";
import { MarketplaceSource } from "./MarketplaceSchemas.js";
import { parseMarketplaceSource, MarketplaceSourceResult } from "./MarketplaceSourceParser.js";
import { log } from "../logger/loggerService.js";
import { getSettings, updateSettings, SettingsSource } from "../terminal/settings.js";
import { getSettingsScope } from "../mcp/PluginUtils.js";

const logger = log("MarketplaceService");

export class MarketplaceService {
    /**
     * Adds a marketplace from a source string or structured source.
     * Based on FP in chunk_566.ts / Marketplace add command logic in chunk_848.ts.
     */
    static async addMarketplace(sourceInput: string | MarketplaceSource, progressCallback?: (msg: string) => void) {
        let source: MarketplaceSource;
        if (typeof sourceInput === 'string') {
            const parsed = parseMarketplaceSource(sourceInput);
            if (!parsed) throw new Error("Invalid marketplace source format");
            if ('error' in parsed) throw new Error(parsed.error);
            source = parsed;
        } else {
            source = sourceInput;
        }

        logger.info(`Adding marketplace source: ${JSON.stringify(source)}`);

        // 1. Download and load manifest to get metadata
        const { marketplace, cachePath } = await downloadMarketplace(source, progressCallback || ((msg) => logger.info(msg)));

        // 2. Validate name
        const name = marketplace.name;
        const nameValidationError = validateMarketplaceName(name, source);
        if (nameValidationError && source.source !== "file" && source.source !== "directory") {
            throw new Error(nameValidationError);
        }

        // 3. Update config
        const config = await readMarketplaceConfigFile();
        if (config[name]) {
            throw new Error(`Marketplace '${name}' is already installed. Please remove it first using 'claude plugin marketplace remove ${name}' if you want to re-install it.`);
        }

        config[name] = {
            source,
            installLocation: cachePath,
            lastUpdated: new Date().toISOString()
        };

        await writeMarketplaceConfigFile(config);
        logger.info(`Added marketplace source: ${name}`);

        return { name };
    }

    /**
     * Removes a marketplace.
     * Based on M31 in chunk_848.ts.
     */
    static async removeMarketplace(name: string) {
        const config = await readMarketplaceConfigFile();
        if (!config[name]) throw new Error(`Marketplace '${name}' not found`);

        delete config[name];
        await writeMarketplaceConfigFile(config);

        // Cleanup cache logic
        const cacheRootDir = getMarketplacesCacheDir();
        const cacheDir = path.join(cacheRootDir, name);

        if (fs.existsSync(cacheDir)) {
            try {
                fs.rmSync(cacheDir, { recursive: true, force: true });
            } catch (e) {
                logger.warn(`Failed to remove marketplace cache directory ${cacheDir}: ${e}`);
            }
        }

        const jsonCache = path.join(cacheRootDir, `${name}.json`);
        if (fs.existsSync(jsonCache)) {
            try {
                fs.rmSync(jsonCache, { force: true });
            } catch (e) { }
        }

        // Remove from settings
        const scopes: SettingsSource[] = ["userSettings", "projectSettings", "localSettings"];

        for (const scope of scopes) {
            try {
                const settings = getSettings(scope);
                let changed = false;
                const update: any = {};

                if (settings.extraKnownMarketplaces?.[name]) {
                    const extra = { ...settings.extraKnownMarketplaces };
                    delete extra[name];
                    update.extraKnownMarketplaces = extra;
                    changed = true;
                }

                if (settings.enabledPlugins) {
                    const enabled = { ...settings.enabledPlugins };
                    let pluginsChanged = false;
                    const suffix = `@${name}`;
                    for (const key of Object.keys(enabled)) {
                        if (key.endsWith(suffix)) {
                            delete enabled[key];
                            pluginsChanged = true;
                        }
                    }
                    if (pluginsChanged) {
                        update.enabledPlugins = enabled;
                        changed = true;
                    }
                }

                if (changed) {
                    updateSettings(scope, update);
                }
            } catch (err) {
                logger.warn(`Error cleaning up settings for ${scope}: ${err}`);
            }
        }

        logger.info(`Removed marketplace source: ${name}`);
        loadMarketplace.cache?.clear?.();
    }

    /**
     * Refreshes a marketplace.
     * Based on ga in chunk_848.ts.
     */
    static async refreshMarketplace(name: string, progressCallback?: (msg: string) => void) {
        const config = await readMarketplaceConfigFile();
        const entry = config[name];
        if (!entry) throw new Error(`Marketplace '${name}' not found.`);

        loadMarketplace.cache?.delete?.(name);

        try {
            await downloadMarketplace(entry.source, progressCallback);
            entry.lastUpdated = new Date().toISOString();
            await writeMarketplaceConfigFile(config);
            logger.info(`Successfully refreshed marketplace: ${name}`);
        } catch (error) {
            const msg = error instanceof Error ? error.message : String(error);
            logger.error(`Failed to refresh marketplace ${name}: ${msg}`);
            throw new Error(`Failed to refresh marketplace '${name}': ${msg}`);
        }
    }

    /**
     * Refreshes all marketplaces.
     */
    static async refreshAllMarketplaces(progressCallback?: (msg: string) => void) {
        const config = await readMarketplaceConfigFile();
        for (const name of Object.keys(config)) {
            try {
                await this.refreshMarketplace(name, progressCallback);
            } catch (error) {
                logger.error(`Failed to refresh ${name}: ${error}`);
            }
        }
    }
}
