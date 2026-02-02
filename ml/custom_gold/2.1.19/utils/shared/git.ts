import { execSync } from 'node:child_process';

export async function isGitRepo(): Promise<boolean> {
    try {
        execSync('git rev-parse --is-inside-work-tree', { stdio: 'ignore' });
        return true;
    } catch {
        return false;
    }
}

export async function getGitState(): Promise<any> {
    try {
        const commitHash = execSync('git rev-parse HEAD', { encoding: 'utf8' }).trim();
        const branchName = execSync('git rev-parse --abbrev-ref HEAD', { encoding: 'utf8' }).trim();
        let remoteUrl = null;
        try {
            remoteUrl = execSync('git remote get-url origin', { encoding: 'utf8' }).trim();
        } catch { }

        const isClean = execSync('git status --porcelain', { encoding: 'utf8' }).trim().length === 0;

        return {
            commitHash,
            branchName,
            remoteUrl,
            isClean,
            isHeadOnRemote: true // Simplified
        };
    } catch (e) {
        return null;
    }
}
