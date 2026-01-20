import * as fs from "fs";
import * as path from "path";
import axios from "axios";
import { z } from "zod";
import { memoize } from "../../utils/shared/lodashLikeRuntimeAndEnv.js";
import { MarketplaceConfig, readMarketplaceConfigFile, getMarketplacesCacheDir } from "./MarketplaceConfig.js";
import { MarketplaceManifestSchema, MarketplaceSource } from "./MarketplaceSchemas.js";
import { updateGitRepository, runGitCommand, checkGitInstalled } from "../../utils/git/GitUtils.js";
import { log } from "../logger/loggerService.js";

const logger = log("MarketplaceLoader");

// JQ2
async function downloadUrlMarketplace(url: string, destPath: string, headers: Record<string, string> = {}): Promise<void> {
    logger.info(`Downloading marketplace from URL: ${url}`);

    // Mask headers for logging
    const maskedHeaders = Object.fromEntries(Object.entries(headers).map(([k, v]) => [k, "***REDACTED***"]));
    if (Object.keys(headers).length > 0) logger.debug(`Using custom headers: ${JSON.stringify(maskedHeaders)}`);

    const requestHeaders = {
        ...headers,
        "User-Agent": "Claude-Code-Plugin-Manager"
    };

    try {
        const response = await axios.get(url, {
            timeout: 10000,
            headers: requestHeaders,
            responseType: 'json'
        });

        logger.debug("Validating marketplace data");
        const parsed = MarketplaceManifestSchema.safeParse(response.data);

        if (!parsed.success) {
            const issues = (parsed.error as any).errors.map((e: any) => `${e.path.join('.')}: ${e.message}`).join(', ');
            throw new Error(`Invalid marketplace schema from URL: ${issues}`);
        }

        const parentDir = path.dirname(destPath);
        if (!fs.existsSync(parentDir)) fs.mkdirSync(parentDir, { recursive: true });

        fs.writeFileSync(destPath, JSON.stringify(parsed.data, null, 2), { encoding: "utf-8", flush: true } as any);

    } catch (error: any) {
        if (axios.isAxiosError(error)) {
            if (error.code === "ECONNREFUSED" || error.code === "ENOTFOUND") {
                throw new Error(`Could not connect to ${url}. Please check your internet connection and verify the URL is correct.\n\nTechnical details: ${error.message}`);
            }
            if (error.code === "ETIMEDOUT") {
                throw new Error(`Request timed out while downloading marketplace from ${url}. The server may be slow or unreachable.\n\nTechnical details: ${error.message}`);
            }
            if (error.response) {
                throw new Error(`HTTP ${error.response.status} error while downloading marketplace from ${url}.\n\nTechnical details: ${error.message}`);
            }
        }
        throw new Error(`Failed to download marketplace from ${url}: ${error instanceof Error ? error.message : String(error)}`);
    }
}

// _WA
async function cloneGitMarketplace(source: string, destPath: string, ref?: string, progressCallback?: (msg: string) => void) {
    if (fs.existsSync(destPath)) {
        if (!fs.existsSync(path.join(destPath, ".git"))) {
            throw new Error(`Cache directory exists at ${destPath} but is not a git repository. Please remove it manually and try again.`);
        }

        progressCallback?.("Updating existing marketplace cache...");
        const updateResult = await updateGitRepository(destPath, ref);

        if (updateResult.code !== 0) {
            logger.error(`Failed to update marketplace cache: ${updateResult.stderr}`);
            progressCallback?.("Update failed, cleaning up and re-cloning...");
            try {
                fs.rmSync(destPath, { recursive: true, force: true });
            } catch (err) {
                throw new Error(`Failed to clean up existing marketplace directory. Please manually delete the directory at ${destPath} and try again.\n\nTechnical details: ${err}`);
            }
        } else {
            return;
        }
    }

    const refMsg = ref ? ` (ref: ${ref})` : "";
    progressCallback?.(`Cloning repository: ${source}${refMsg}`);

    // jA5 equivalent
    const args = ["clone", "--depth", "1"];
    if (ref) args.push("--branch", ref);
    args.push(source, destPath);

    // Using runGitCommand for cloning explicitly if needed, but updateGitRepository generally handles updates.
    // For clone, we use runGitCommand

    // jA5 has logic to inject SSH command but assuming GitUtils/env handles basic auth or keys are set up.
    // Chunk 341 adds -c credential.helper= etc.
    const cloneResult = await runGitCommand(args, process.cwd()); // run from cwd, git clone creates dir

    if (cloneResult.code !== 0) {
        throw new Error(`Failed to clone marketplace repository: ${cloneResult.stderr}`);
    }
    progressCallback?.("Clone complete, validating marketplace...");
}


function getCacheNameForSource(source: MarketplaceSource): string {
    if (source.source === "github") {
        return source.repo.replace("/", "-");
    }
    if (source.source === "npm") {
        return source.package.replace("@", "").replace("/", "-");
    }
    if (source.source === "file") {
        return path.basename(source.path).replace(".json", "");
    }
    if (source.source === "directory") {
        return path.basename(source.path);
    }
    return "temp_" + Date.now();
}

function readMarketplaceFile(filePath: string) {
    const content = fs.readFileSync(filePath, "utf-8");
    const json = JSON.parse(content);
    const result = MarketplaceManifestSchema.safeParse(json);
    if (!result.success) {
        const issues = (result.error as any).errors.map((e: any) => `${e.path.join('.')}: ${e.message}`).join(', ');
        throw new Error(`Invalid schema: ${issues}`);
    }
    return result.data;
}

// n50
export async function downloadMarketplace(source: MarketplaceSource, progressCallback?: (msg: string) => void): Promise<{ marketplace: z.infer<typeof MarketplaceManifestSchema>, cachePath: string }> {
    const cacheRootDir = getMarketplacesCacheDir();
    if (!fs.existsSync(cacheRootDir)) fs.mkdirSync(cacheRootDir, { recursive: true });

    const tempDir = path.join(cacheRootDir, "temp_" + Date.now()); // Using unique temp per op? Original uses name based cache always?
    // Original n50 creates YQ2() dir (marketplaces).
    // Calculates X (cache name) from source (PA5).
    // Determines target paths Z, Y.

    const cacheName = getCacheNameForSource(source);

    // Ideally we download to a temp location first then rename/swap if successful to be atomic-ish
    // But original code seems to operate on cached location directly for git?

    let cachePath = path.join(cacheRootDir, cacheName);
    let marketplaceFile = cachePath;
    let isTemp = false;

    // Logic based on source type
    if (source.source === "url") {
        // use temp dir logic from original n50?
        // original switch: case "url": Z = K$(G, `${X}.json`), J=!0...
        cachePath = path.join(cacheRootDir, `${cacheName}.json`);
        await downloadUrlMarketplace(source.url, cachePath, source.headers as Record<string, string>);
        marketplaceFile = cachePath;
    } else if (source.source === "github" || source.source === "git") {
        // handle git
        let gitUrl = source.source === "github" ? `https://github.com/${source.repo}.git` : source.url;
        // SSH fallback logic omitted for brevity, assuming HTTPS or proper SSH setup
        await cloneGitMarketplace(gitUrl, cachePath, source.ref, progressCallback);
        marketplaceFile = path.join(cachePath, source.path || ".claude-plugin/marketplace.json");
    } else if (source.source === "file" || source.source === "directory") {
        cachePath = source.path;
        marketplaceFile = source.source === "directory" ? path.join(source.path, ".claude-plugin/marketplace.json") : source.path;
    } else {
        throw new Error(`Unsupported marketplace source type: ${source.source}`);
    }

    if (!fs.existsSync(marketplaceFile)) {
        throw new Error(`Marketplace file not found at ${marketplaceFile}`);
    }

    const marketplace = readMarketplaceFile(marketplaceFile);

    // Finalize: if we downloaded to a generic name but the marketplace has a specific name 'marketplace.name',
    // we should rename the cache folder to match the marketplace name if possible?
    // Original code logic: "if (Z !== W && !K) ... renameSync(Z, W)"
    // It renames the cache directory to match the marketplace NAME found in the manifest.

    if (source.source !== "file" && source.source !== "directory") {
        const finalCachePath = path.join(cacheRootDir, marketplace.name);
        // Only rename if different
        if (path.resolve(cachePath) !== path.resolve(finalCachePath)) {
            if (fs.existsSync(finalCachePath)) {
                progressCallback?.("Cleaning up old marketplace cache...");
                fs.rmSync(finalCachePath, { recursive: true, force: true });
            }
            if (fs.existsSync(cachePath)) {
                fs.renameSync(cachePath, finalCachePath);
            }
            cachePath = finalCachePath;
        }
    }

    return { marketplace, cachePath };
}

// W$ (memoized loader)
export const loadMarketplace = memoize(async (marketplaceName: string) => {
    const config = await readMarketplaceConfigFile();
    const entry = config[marketplaceName];

    if (!entry) {
        // Fallback or error?
        throw new Error(`Marketplace '${marketplaceName}' not found in configuration.`);
    }

    const { installLocation } = entry;

    // Verify it exists, else re-fetch?
    // Original W$: try to load from installLocation (a50). If fail, re-fetch (n50).
    try {
        let manifestPath = installLocation;
        if (fs.statSync(installLocation).isDirectory()) {
            manifestPath = path.join(installLocation, ".claude-plugin/marketplace.json");
        }
        return readMarketplaceFile(manifestPath);
    } catch (err) {
        logger.warn(`Cache corrupted or missing for marketplace ${marketplaceName}, re-fetching from source: ${err}`);
        // Re-fetch
        const { marketplace } = await downloadMarketplace(entry.source);
        // Update timestamp? original does Q[A].lastUpdated = new Date()...
        return marketplace;
    }
});

// oBA
export async function loadConfiguredMarketplaces(config: Record<string, any>) {
    const marketplaces: any[] = [];
    const failures: any[] = [];

    for (const [name, entry] of Object.entries(config)) {
        try {
            const data = await loadMarketplace(name);
            marketplaces.push({ name, config: entry, data });
        } catch (error) {
            failures.push({ name, error: error instanceof Error ? error.message : String(error) });
        }
    }
    return { marketplaces, failures };
}

