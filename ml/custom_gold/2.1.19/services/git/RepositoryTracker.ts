/**
 * File: src/services/git/RepositoryTracker.ts
 * Role: Tracks local filesystem paths corresponding to Git repositories.
 */

import { realpathSync, existsSync } from "node:fs";
import { getSettings, updateSettings } from "../config/SettingsService.js";

export interface RepositoryTrackerService {
    addPath(repoUrl: string, localPath: string): void;
    getExistingPaths(repoUrl: string): string[];
}

/**
 * Service for tracking and retrieving local paths for Git repositories.
 */
export const RepositoryTracker: RepositoryTrackerService = {
    /**
     * Adds a local path to the tracking list for a given repository.
     */
    addPath(repoUrl: string, localPath: string): void {
        let realPath: string;
        try {
            realPath = realpathSync(localPath);
        } catch {
            realPath = localPath;
        }

        const normalizedRepo = repoUrl.toLowerCase();
        const settings = getSettings();
        const repoPaths = settings.repoPaths || {};

        const existingPaths = repoPaths[normalizedRepo] || [];
        if (!existingPaths.includes(realPath)) {
            existingPaths.push(realPath);
            repoPaths[normalizedRepo] = existingPaths;
            updateSettings({ repoPaths });
            console.log(`[RepoTracker] Tracking ${realPath} for ${normalizedRepo}`);
        }
    },

    /**
     * Returns all tracked local paths for a repository that still exist.
     */
    getExistingPaths(repoUrl: string): string[] {
        const normalizedRepo = repoUrl.toLowerCase();
        const settings = getSettings();
        const repoPaths = settings.repoPaths || {};

        const paths = repoPaths[normalizedRepo] || [];
        return paths.filter(p => existsSync(p));
    }
};
