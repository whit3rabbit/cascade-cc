
import { homedir } from 'os';
import { join } from 'path';

export function getStateHome(): string {
    return process.env.XDG_STATE_HOME ?? join(homedir(), '.local', 'state');
}

export function getCacheHome(): string {
    return process.env.XDG_CACHE_HOME ?? join(homedir(), '.cache');
}

export function getDataHome(): string {
    return process.env.XDG_DATA_HOME ?? join(homedir(), '.local', 'share');
}

export function getLocalBin(): string {
    return join(homedir(), '.local', 'bin');
}

import { spawn } from 'node:child_process';

export function openUrl(url: string): void {
    const command = process.platform === 'darwin' ? 'open' : process.platform === 'win32' ? 'start' : 'xdg-open';
    const args = process.platform === 'win32' ? [url] : [url]; // 'start' might need more args on windows but this is a stub
    spawn(command, args, { stdio: 'ignore', detached: true }).unref();
}
