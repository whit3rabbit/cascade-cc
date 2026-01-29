/**
 * File: src/services/git/RepositoryTracker.ts
 * Role: Tracks local filesystem paths corresponding to Git repositories.
 */

import { realpathSync, existsSync } from "node:fs";

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
        // Here we would interact with a persistent state store (e.g. settings)
        // For deobfuscation, we're stubbing the intent.
        console.log(`[RepoTracker] Tracking ${realPath} for ${normalizedRepo}`);
    },

    /**
     * Returns all tracked local paths for a repository that still exist.
     */
    getExistingPaths(repoUrl: string): string[] {
        const normalizedRepo = repoUrl.toLowerCase();
        // Mocked retrieval from state
        // In a real implementation, we would filter stored paths by normalizedRepo
        const paths: string[] = [];
        return paths.filter(p => existsSync(p));
    }
};
