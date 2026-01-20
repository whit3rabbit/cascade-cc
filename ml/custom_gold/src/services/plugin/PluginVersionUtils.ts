import { spawn } from 'node:child_process';
import { log } from '../logger/loggerService.js';

const logger = log('PluginVersionUtils');

/**
 * Gets the current git commit SHA for a directory.
 */
export async function getGitSha(dir: string): Promise<string | undefined> {
    return new Promise((resolve) => {
        const git = spawn('git', ['-j', 'rev-parse', 'HEAD'], {
            cwd: dir,
            shell: true
        });

        let stdout = '';
        git.stdout.on('data', (data) => {
            stdout += data.toString();
        });

        git.on('close', (code) => {
            if (code === 0 && stdout.trim()) {
                resolve(stdout.trim());
            } else {
                resolve(undefined);
            }
        });

        git.on('error', (err) => {
            logger.warn(`Failed to get git SHA for ${dir}: ${err.message}`);
            resolve(undefined);
        });
    });
}

/**
 * Gets the version of a plugin from its manifest or directory state.
 */
export function getPluginVersion(manifest: any, dir: string): string {
    if (manifest?.version) return manifest.version;
    // Fallback logic if needed, maybe using mtime or similar
    return 'unknown';
}
