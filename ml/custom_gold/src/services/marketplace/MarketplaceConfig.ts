import * as fs from "fs";
import * as path from "path";
import { z } from "zod";
import { getConfigDir } from "../../utils/settings/runtimeSettingsAndAuth.js";
import { MarketplaceSourceSchema } from "./MarketplaceSchemas.js";
import { log } from "../logger/loggerService.js";

const logger = log("MarketplaceConfig");
const MARKETPLACE_CONFIG_FILE = "known_marketplaces.json";

export function getGlobalPluginsDir(): string {
    return path.join(getConfigDir(), "plugins");
}

export function getMarketplaceConfigPath(): string {
    return path.join(getGlobalPluginsDir(), MARKETPLACE_CONFIG_FILE);
}

export function getMarketplacesCacheDir(): string {
    return path.join(getGlobalPluginsDir(), "marketplaces");
}

export const MarketplaceConfigEntrySchema = z.object({
    source: MarketplaceSourceSchema,
    installLocation: z.string(),
    lastUpdated: z.string(),
    autoUpdate: z.boolean().optional()
});

export const MarketplaceConfigSchema = z.record(z.string(), MarketplaceConfigEntrySchema);

export type MarketplaceConfig = z.infer<typeof MarketplaceConfigSchema>;

export async function getMarketplaceConfig(): Promise<MarketplaceConfig> {
    const configPath = getMarketplaceConfigPath();
    if (!fs.existsSync(configPath)) return {};

    try {
        const content = fs.readFileSync(configPath, "utf-8");
        const json = JSON.parse(content);
        const parsed = MarketplaceConfigSchema.safeParse(json);

        if (!parsed.success) {
            const errorMsg = `Marketplace configuration file is corrupted: ${(parsed.error as any).errors.map((e: any) => `${e.path.join('.')}: ${e.message}`).join(', ')}`;
            logger.error(errorMsg);
            throw new Error(errorMsg);
        }
        return parsed.data;
    } catch (error) {
        // If it's the specific error we threw, rethrow it
        if (error instanceof Error && error.message.startsWith("Marketplace configuration file is corrupted")) {
            throw error;
        }
        const msg = `Failed to load marketplace configuration: ${error instanceof Error ? error.message : String(error)}`;
        logger.error(msg);
        throw new Error(msg);
    }
}

// Alias for readMarketplaceConfigFile
export const readMarketplaceConfigFile = getMarketplaceConfig;

export async function saveMarketplaceConfig(config: MarketplaceConfig): Promise<void> {
    const parsed = MarketplaceConfigSchema.safeParse(config);
    if (!parsed.success) throw new Error(`Invalid marketplace config: ${parsed.error.message}`);

    const configPath = getMarketplaceConfigPath();
    const dir = path.dirname(configPath);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });

    fs.writeFileSync(configPath, JSON.stringify(parsed.data, null, 2), { encoding: 'utf-8', flush: true } as any);
}

// Alias for writeMarketplaceConfigFile
export const writeMarketplaceConfigFile = saveMarketplaceConfig;

