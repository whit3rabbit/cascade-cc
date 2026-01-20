import { sep, dirname } from 'node:path';
import chokidar, { FSWatcher } from 'chokidar';
import { SETTING_SOURCES, getSettingsPath, SettingsSource } from './settings.js';
import { log } from '../logger/loggerService.js';

const logger = log('settings');

const STABILITY_THRESHOLD = 1000;
const POLL_INTERVAL = 500;
const INTERNAL_WRITE_COOLDOWN = 5000;

interface SettingsWatcherOptions {
    stabilityThreshold?: number;
    pollInterval?: number;
}

let watcher: FSWatcher | null = null;
let isInitialized = false;
let isDisposed = false;
let internalWriteTimes = new Map<string, number>();
let subscribers = new Set<(source: SettingsSource) => void>();
let currentOptions: SettingsWatcherOptions | null = null;

/**
 * Gets the source for a given file path.
 */
function getSourceByPath(filePath: string): SettingsSource | undefined {
    return SETTING_SOURCES.find(source => getSettingsPath(source) === filePath);
}

/**
 * Handles file change events from chokidar.
 */
function handleSettingsChange(filePath: string) {
    const source = getSourceByPath(filePath);
    if (!source) return;

    const internalWriteTime = internalWriteTimes.get(filePath);
    if (internalWriteTime && Date.now() - internalWriteTime < INTERNAL_WRITE_COOLDOWN) {
        internalWriteTimes.delete(filePath);
        return;
    }

    logger.debug(`Detected change to ${filePath}`);
    subscribers.forEach(callback => callback(source));
}

/**
 * Handles file deletion events from chokidar.
 */
function handleSettingsUnlink(filePath: string) {
    const source = getSourceByPath(filePath);
    if (!source) return;

    logger.debug(`Detected deletion of ${filePath}`);
    subscribers.forEach(callback => callback(source));
}

/**
 * Gets the base directories for setting files to watch.
 */
function getSettingsDirectoriesToWatch(): string[] {
    return SETTING_SOURCES.map(source => {
        const filePath = getSettingsPath(source);
        if (!filePath) return undefined;
        // In original code: checks if it's a file
        return dirname(filePath);
    }).filter((dir): dir is string => dir !== undefined);
}

/**
 * Starts watching setting files for changes.
 */
export function initializeSettingsWatcher() {
    if (isInitialized || isDisposed) return;
    isInitialized = true;

    const dirs = getSettingsDirectoriesToWatch();
    if (dirs.length === 0) return;

    logger.info(`Watching for changes in setting directories: ${dirs.join(", ")}...`);

    watcher = chokidar.watch(dirs, {
        persistent: true,
        ignoreInitial: true,
        depth: 0,
        awaitWriteFinish: {
            stabilityThreshold: currentOptions?.stabilityThreshold ?? STABILITY_THRESHOLD,
            pollInterval: currentOptions?.pollInterval ?? POLL_INTERVAL
        },
        ignored: (path: string) => path.split(sep).some(part => part === ".git"),
        ignorePermissionErrors: true,
        usePolling: false,
        atomic: true
    });

    watcher.on("change", handleSettingsChange);
    watcher.on("unlink", handleSettingsUnlink);
}

/**
 * Stops watching setting files.
 */
export function disposeSettingsWatcher() {
    isDisposed = true;
    if (watcher) {
        watcher.close();
        watcher = null;
    }
    internalWriteTimes.clear();
    subscribers.clear();
}

/**
 * Registers a callback for settings changes.
 */
export function subscribeToSettings(callback: (source: SettingsSource) => void): () => void {
    subscribers.add(callback);
    return () => {
        subscribers.delete(callback);
    };
}

/**
 * Marks a file write as internal to prevent a loop when the watcher sees the change.
 */
export function markInternalSettingsWrite(source: SettingsSource) {
    const filePath = getSettingsPath(source);
    if (filePath) {
        internalWriteTimes.set(filePath, Date.now());
    }
}

/**
 * Programmatically triggers a change notification.
 */
export function notifySettingsChange(source: SettingsSource) {
    logger.debug(`Programmatic settings change notification for ${source}`);
    subscribers.forEach(callback => callback(source));
}

/**
 * Resets internal state for testing.
 */
export function resetSettingsWatcherForTesting(options?: SettingsWatcherOptions) {
    isInitialized = false;
    isDisposed = false;
    currentOptions = options ?? null;
}

export const settingsWatcher = {
    initialize: initializeSettingsWatcher,
    dispose: disposeSettingsWatcher,
    subscribe: subscribeToSettings,
    markInternalWrite: markInternalSettingsWrite,
    notifyChange: notifySettingsChange,
    resetForTesting: resetSettingsWatcherForTesting
};
