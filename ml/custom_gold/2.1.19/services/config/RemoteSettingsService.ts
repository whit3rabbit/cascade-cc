/**
 * File: src/services/config/RemoteSettingsService.ts
 * Role: Handles fetching and caching of remote terminal settings from the Anthropic API.
 */

import { join } from 'node:path';
import { existsSync, readFileSync, writeFileSync, unlinkSync } from 'node:fs';
import axios from 'axios';
import { getBaseConfigDir } from '../../utils/shared/runtimeAndEnv.js';
import { getAuthHeaders } from '../auth/AuthService.js';

const REMOTE_SETTINGS_FILE = 'remote-settings.json';

export interface RemoteSettings {
    [key: string]: any;
}

/**
 * Fetches remote settings from the Anthropic API and caches them locally.
 * 
 * @returns {Promise<RemoteSettings>} The latest remote settings.
 */
export async function fetchRemoteSettings(): Promise<RemoteSettings> {
    try {
        const headers = await getAuthHeaders();
        // If no auth, we can't fetch remote settings
        if (Object.keys(headers).length === 0) return {};

        // Hardcoded endpoint for now (would usually come from config)
        const url = 'https://api.anthropic.com/api/claude_code/settings';

        const response = await axios.get(url, {
            headers,
            timeout: 5000
        });

        if (response.data && response.data.settings) {
            cacheRemoteSettings(response.data.settings);
            return response.data.settings;
        }

        return {};
    } catch (error) {
        // console.error("Failed to fetch remote settings, using cache if available.");
        return getCachedRemoteSettings();
    }
}

/**
 * Persists settings to a local cache file.
 */
function cacheRemoteSettings(settings: RemoteSettings): void {
    const cachePath = join(getBaseConfigDir(), REMOTE_SETTINGS_FILE);
    try {
        writeFileSync(cachePath, JSON.stringify(settings, null, 2), 'utf8');
    } catch (err) {
        // Ignore cache write errors
    }
}

/**
 * Retrieves settings from the local cache.
 */
export function getCachedRemoteSettings(): RemoteSettings {
    const cachePath = join(getBaseConfigDir(), REMOTE_SETTINGS_FILE);
    if (!existsSync(cachePath)) return {};

    try {
        const data = readFileSync(cachePath, 'utf8');
        return JSON.parse(data);
    } catch (err) {
        return {};
    }
}

/**
 * Clears the local remote settings cache.
 */
export function clearRemoteSettingsCache(): void {
    const cachePath = join(getBaseConfigDir(), REMOTE_SETTINGS_FILE);
    try {
        if (existsSync(cachePath)) {
            unlinkSync(cachePath);
        }
    } catch (err) {
        // Ignore cache delete errors
    }
}
