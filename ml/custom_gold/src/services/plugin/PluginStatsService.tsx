
// Logic from chunk_568.ts (Plugin Stats & UI Footer)
// Merged contents of src/services/plugins/PluginStatsService.ts and src/services/plugins/PluginStatsService.tsx

import React from "react";
import { Box, Text } from "ink";
import path from "node:path";
import fs from "node:fs";
import { randomBytes } from "crypto";
import { getConfigDir } from "../../utils/settings/runtimeSettingsAndAuth.js";

// --- Plugin List Footer (wX9) ---
export function PluginListFooter({ hasSelection }: { hasSelection: boolean }) {
    return (
        <Box marginLeft={3}>
            <Text italic>
                {hasSelection && (
                    <Text bold color="cyan">Press i to install · </Text>
                )}
                <Text dimColor>Space: (de)select · Enter: details · Esc: back</Text>
            </Text>
        </Box>
    );
}

const CACHE_VERSION = 1;
const CACHE_FILENAME = "install-counts-cache.json";
const STATS_URL = "https://raw.githubusercontent.com/anthropics/claude-plugins-official/refs/heads/stats/stats/plugin-installs.json";
const CACHE_TTL_MS = 24 * 60 * 60 * 1000; // 24 hours

interface PluginInstallCount {
    plugin: string;
    unique_installs: number;
}

interface CacheData {
    version: number;
    fetchedAt: string;
    counts: PluginInstallCount[];
}

function getCacheDir() {
    return path.join(getConfigDir(), "plugins");
}

function getCachePath() {
    return path.join(getCacheDir(), CACHE_FILENAME);
}

// --- Formatting Utils (oH1) ---
export function formatInstallCount(count: number): string {
    if (count < 1000) return String(count);
    if (count < 1000000) {
        const k = (count / 1000).toFixed(1);
        return k.endsWith(".0") ? `${k.slice(0, -2)}K` : `${k}K`;
    }
    const m = (count / 1000000).toFixed(1);
    return m.endsWith(".0") ? `${m.slice(0, -2)}M` : `${m}M`;
}

function loadCache(): CacheData | null {
    const cachePath = getCachePath();
    try {
        if (!fs.existsSync(cachePath)) return null;
        const content = fs.readFileSync(cachePath, "utf-8");
        const data = JSON.parse(content);

        if (data.version !== CACHE_VERSION) return null;
        if (Date.now() - new Date(data.fetchedAt).getTime() > CACHE_TTL_MS) return null;

        return data as CacheData;
    } catch (e) {
        return null;
    }
}

function saveCache(counts: PluginInstallCount[]) {
    const cacheDir = getCacheDir();
    const cachePath = getCachePath();
    const tempPath = `${cachePath}.${randomBytes(8).toString("hex")}.tmp`;

    try {
        if (!fs.existsSync(cacheDir)) fs.mkdirSync(cacheDir, { recursive: true });

        const data: CacheData = {
            version: CACHE_VERSION,
            fetchedAt: new Date().toISOString(),
            counts
        };

        fs.writeFileSync(tempPath, JSON.stringify(data, null, 2), { encoding: 'utf-8', mode: 0o600 });
        fs.renameSync(tempPath, cachePath);
    } catch (e) {
        console.error("Failed to save plugin stats cache:", e);
        try { if (fs.existsSync(tempPath)) fs.unlinkSync(tempPath); } catch { }
    }
}

async function fetchStats(): Promise<PluginInstallCount[]> {
    try {
        const response = await fetch(STATS_URL);
        if (!response.ok) throw new Error(`Failed to fetch stats: ${response.statusText}`);
        const data: any = await response.json();

        // Handle format: { plugins: [...] } or just array if source changes
        let counts: PluginInstallCount[] = [];
        if (data.plugins && Array.isArray(data.plugins)) {
            counts = data.plugins;
        } else if (Array.isArray(data)) {
            counts = data;
        } else {
            // throw new Error("Invalid stats format");
            return [];
        }

        return counts;
    } catch (e) {
        // console.warn("Failed to fetch plugin stats:", e);
        return [];
    }
}

export async function getPluginInstallCounts(): Promise<Map<string, number>> {
    const cached = loadCache();
    if (cached) {
        const map = new Map<string, number>();
        cached.counts.forEach(c => map.set(c.plugin, c.unique_installs));
        return map;
    }

    try {
        const counts = await fetchStats();
        if (counts.length > 0) {
            saveCache(counts);
        }
        const map = new Map<string, number>();
        counts.forEach(c => map.set(c.plugin, c.unique_installs));
        return map;
    } catch (e) {
        return new Map();
    }
}

// Fallback for direct cache read compatibility if needed by other components
export function loadInstallCountsCache(storageDir: string): any {
    // We ignore storageDir as we use standard path
    const cached = loadCache();
    return cached ? cached.counts : null;
}

export function saveInstallCountsCache(storageDir: string, counts: any[]) {
    saveCache(counts);
}
