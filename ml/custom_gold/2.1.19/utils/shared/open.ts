import { exec } from 'node:child_process';

export function openUrl(url: string): void {
    const start = process.platform === 'darwin' ? 'open' : process.platform === 'win32' ? 'start' : 'xdg-open';
    exec(`${start} ${url}`);
}
