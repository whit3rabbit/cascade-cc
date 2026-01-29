/**
 * File: src/utils/fs/Watcher.ts
 * Role: Watches for changes in configuration files and notifies subscribers.
 */

import * as path from "node:path";
import { watch, FSWatcher } from 'chokidar';
import { getBaseConfigDir } from '../shared/runtimeAndEnv.js';

let fileWatcher: FSWatcher | null = null;
let isWatcherInitialized = false;
let isWatchDisposed = false;
let watcherOptions: any = null;

const DEFAULT_STABILITY_THRESHOLD = 1000;
const DEFAULT_POLL_INTERVAL = 500;
const CHANGE_DEBOUNCE = 5000;

const subscribers = new Set<(filePath: string) => void>();
const writeTimes = new Map<string, number>();

/**
 * Resolves a setting name to its corresponding file path.
 */
function resolveSettingsPath(settingName: string): string {
    // Currently, we mostly watch settings.json
    if (settingName === 'settings' || settingName === 'flagSettings') {
        return path.join(getBaseConfigDir(), 'settings.json');
    }
    return '';
}

/**
 * Handles a file change event.
 */
function handleFileChange(filePath: string) {
    const lastWriteTime = writeTimes.get(filePath);
    if (lastWriteTime && Date.now() - lastWriteTime < CHANGE_DEBOUNCE) {
        writeTimes.delete(filePath);
        return;
    }

    console.log(`[Watcher] Detected change to ${filePath}`);
    subscribers.forEach(callback => callback(filePath));
}

/**
 * Handles a file deletion event.
 */
function handleFileUnlink(filePath: string) {
    console.log(`[Watcher] Detected deletion of ${filePath}`);
    subscribers.forEach(callback => callback(filePath));
}

/**
 * Returns the directories/files to watch.
 */
function getWatchTargets(): string[] {
    const settingsPath = resolveSettingsPath('settings');
    if (settingsPath) {
        return [path.dirname(settingsPath)];
    }
    return [];
}

/**
 * Initializes the file watcher.
 */
export function initializeFileWatcher() {
    if (isWatcherInitialized || isWatchDisposed) {
        return;
    }
    isWatcherInitialized = true;

    const targets = getWatchTargets();
    if (targets.length === 0) {
        return;
    }

    console.log(`[Watcher] Watching for changes in: ${targets.join(", ")}...`);

    fileWatcher = watch(targets, {
        persistent: true,
        ignoreInitial: true,
        depth: 0,
        awaitWriteFinish: {
            stabilityThreshold: watcherOptions?.stabilityThreshold ?? DEFAULT_STABILITY_THRESHOLD,
            pollInterval: watcherOptions?.pollInterval ?? DEFAULT_POLL_INTERVAL,
        },
        ignored: (filePath: string) => {
            return filePath.split(path.sep).some(part => part === ".git");
        },
        ignorePermissionErrors: true,
        usePolling: false,
        atomic: true,
    });

    fileWatcher.on("change", handleFileChange);
    fileWatcher.on("unlink", handleFileUnlink);

    process.on('exit', () => disposeFileWatcher());
}

/**
 * Disposes of the file watcher and cleans up resources.
 */
export function disposeFileWatcher() {
    isWatchDisposed = true;
    if (fileWatcher) {
        fileWatcher.close();
        fileWatcher = null;
    }
    writeTimes.clear();
    subscribers.clear();
}

/**
 * Subscribes to file change notifications.
 */
export function subscribeToFileChange(callback: (filePath: string) => void) {
    subscribers.add(callback);
    return () => {
        subscribers.delete(callback);
    };
}

/**
 * Marks a write as internal to avoid debouncing unnecessary notifications.
 */
export function markInternalWrite(settingName: string) {
    const settingPath = resolveSettingsPath(settingName);
    if (settingPath) {
        writeTimes.set(settingPath, Date.now());
    }
}

/**
 * Resets the watcher state for testing purposes.
 */
export function resetForTesting(options?: any) {
    isWatcherInitialized = false;
    isWatchDisposed = false;
    watcherOptions = options ?? null;
}
