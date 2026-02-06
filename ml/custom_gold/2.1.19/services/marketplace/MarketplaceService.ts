import * as fs from 'node:fs';
import * as path from 'node:path';
import { execSync } from 'node:child_process';
import axios from 'axios';
import { LoadedMarketplace } from './MarketplaceLoader.js';
import { getSettings, updateSettings } from '../config/SettingsService.js';
import { MarketplaceSource, parseMarketplaceSource } from '../../utils/marketplace/SourceParser.js';
import { sanitizeFilePath } from '../../utils/fs/pathSanitizer.js';
import { getBaseConfigDir } from '../../utils/shared/runtimeAndEnv.js';
import { gitClone, gitPull } from '../../utils/shared/git.js';

export interface MarketplaceServiceType {
    addMarketplace: (name: string, source: string | MarketplaceSource) => Promise<void>;
    removeMarketplace: (name: string) => Promise<void>;
    refreshMarketplace: (name: string, onProgress?: (msg: string) => void) => Promise<void>;
    refreshAllMarketplaces: (onProgress?: (msg: string) => void) => Promise<void>;
    listMarketplaces: () => Promise<LoadedMarketplace[]>;
    autoInstallOfficialMarketplace: () => Promise<void>;
}

const DEFAULT_MARKETPLACE_MANIFEST_PATH = path.join('.claude-plugin', 'marketplace.json');
type ValidMarketplaceSource = Exclude<MarketplaceSource, { error: string }>;

function getMarketplaceCacheDir(): string {
    const cacheDir = path.join(getBaseConfigDir(), 'marketplaces');
    if (!fs.existsSync(cacheDir)) {
        fs.mkdirSync(cacheDir, { recursive: true });
    }
    return cacheDir;
}

function sanitizeMarketplaceName(name: string): string {
    return name.replace(/[^a-zA-Z0-9._-]/g, '-');
}

function normalizeMarketplaceSource(input: string | MarketplaceSource): ValidMarketplaceSource {
    if (typeof input === 'string') {
        const parsed = parseMarketplaceSource(input);
        if (!parsed) {
            throw new Error(`Invalid marketplace source: ${input}`);
        }
        if ('error' in parsed) {
            throw new Error(parsed.error);
        }
        return parsed;
    }
    if ('error' in input) {
        throw new Error(input.error);
    }
    return input;
}

async function ensureGitRepo(url: string, targetPath: string, ref?: string, onProgress?: (msg: string) => void) {
    if (fs.existsSync(targetPath)) {
        if (onProgress) onProgress(`Updating marketplace at ${targetPath}...`);
        await gitPull(targetPath);
    } else {
        if (onProgress) onProgress(`Cloning marketplace from ${url}...`);
        await gitClone(url, targetPath);
    }
    if (ref) {
        execSync(`git checkout ${ref}`, { cwd: targetPath, stdio: 'ignore' });
    }
}

function getSanitizedPath(inputPath: string): string {
    const sanitized = sanitizeFilePath(inputPath);
    if (!sanitized) {
        throw new Error(`Invalid path: ${inputPath}`);
    }
    return sanitized;
}

function getRelativeManifestPath(inputPath?: string): string {
    const raw = inputPath || DEFAULT_MARKETPLACE_MANIFEST_PATH;
    const sanitized = sanitizeFilePath(raw);
    if (!sanitized) {
        throw new Error(`Invalid marketplace manifest path: ${raw}`);
    }
    return sanitized;
}

async function loadMarketplaceData(
    source: ValidMarketplaceSource,
    name: string,
    onProgress?: (msg: string) => void
): Promise<{ data: any; installLocation?: string }> {
    const src = source;
    switch (src.source) {
        case 'url': {
            const response = await axios.get(src.url, { timeout: 10000 });
            return { data: response.data, installLocation: src.url };
        }
        case 'github': {
            const cacheDir = getMarketplaceCacheDir();
            const cachePath = path.join(cacheDir, sanitizeMarketplaceName(name));
            const repoUrl = `https://github.com/${src.repository}.git`;
            await ensureGitRepo(repoUrl, cachePath, src.ref, onProgress);
            const manifestPath = path.join(cachePath, getRelativeManifestPath((src as any).path));
            if (!fs.existsSync(manifestPath)) {
                throw new Error(`Marketplace file not found at ${manifestPath}`);
            }
            const content = fs.readFileSync(manifestPath, 'utf8');
            return { data: JSON.parse(content), installLocation: cachePath };
        }
        case 'git': {
            const cacheDir = getMarketplaceCacheDir();
            const cachePath = path.join(cacheDir, sanitizeMarketplaceName(name));
            await ensureGitRepo(src.url, cachePath, src.ref, onProgress);
            const manifestPath = path.join(cachePath, getRelativeManifestPath((src as any).path));
            if (!fs.existsSync(manifestPath)) {
                throw new Error(`Marketplace file not found at ${manifestPath}`);
            }
            const content = fs.readFileSync(manifestPath, 'utf8');
            return { data: JSON.parse(content), installLocation: cachePath };
        }
        case 'file': {
            const filePath = getSanitizedPath(src.path);
            if (!fs.existsSync(filePath)) {
                throw new Error(`Marketplace file not found at ${filePath}`);
            }
            const content = fs.readFileSync(filePath, 'utf8');
            return { data: JSON.parse(content), installLocation: filePath };
        }
        case 'directory': {
            const dirPath = getSanitizedPath(src.path);
            const manifestPath = path.join(dirPath, DEFAULT_MARKETPLACE_MANIFEST_PATH);
            if (!fs.existsSync(manifestPath)) {
                throw new Error(`Marketplace file not found at ${manifestPath}`);
            }
            const content = fs.readFileSync(manifestPath, 'utf8');
            return { data: JSON.parse(content), installLocation: dirPath };
        }
        default:
            throw new Error(`Unsupported marketplace source type: ${(src as any).source}`);
    }
}

/**
 * Service for managing external plugin marketplaces.
 */
export const MarketplaceService: MarketplaceServiceType = {
    /**
     * Adds a new marketplace source.
     */
    async addMarketplace(name: string, sourceInput: string | MarketplaceSource): Promise<void> {
        const source = normalizeMarketplaceSource(sourceInput);
        updateSettings((current) => {
            const marketplaces = current.marketplaces || {};
            return {
                ...current,
                marketplaces: {
                    ...marketplaces,
                    [name]: {
                        source,
                        ...((source as any).source === 'url' ? { url: (source as any).url } : {})
                    }
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
            const { [name]: _removed, ...rest } = marketplaces;
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

        if (!marketplaceConfig) {
            throw new Error(`Marketplace '${name}' not found`);
        }

        try {
            const source: ValidMarketplaceSource | null = marketplaceConfig.source
                ? normalizeMarketplaceSource(marketplaceConfig.source)
                : (marketplaceConfig.url ? { source: 'url', url: marketplaceConfig.url } : null);

            if (!source) {
                throw new Error(`Marketplace '${name}' configuration missing source`);
            }

            const { data, installLocation } = await loadMarketplaceData(source, name, onProgress);

            const plugins = Array.isArray(data) ? data : data.plugins;
            if (!plugins || !Array.isArray(plugins)) {
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
                            source,
                            installLocation,
                            lastUpdated: new Date().toISOString(),
                            plugins
                        }
                    }
                };
            });

            if (onProgress) onProgress(`Marketplace ${name} refreshed with ${plugins.length} plugins.`);
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
        const OFFICIAL_MARKETPLACE_SOURCE: MarketplaceSource = {
            source: "github",
            repository: OFFICIAL_MARKETPLACE_REPO
        };

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
            await MarketplaceService.addMarketplace(OFFICIAL_MARKETPLACE_NAME, OFFICIAL_MARKETPLACE_SOURCE);
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
