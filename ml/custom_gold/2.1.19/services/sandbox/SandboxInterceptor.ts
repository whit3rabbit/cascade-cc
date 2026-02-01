/**
 * File: src/services/sandbox/SandboxInterceptor.ts
 * Role: OS-level sandbox wrapper for shell commands.
 */

import os from 'node:os';
import { isSandboxEnabled, getSandboxConfig } from './SandboxSettings.js';
import { quote } from 'shell-quote';
import fs from 'node:fs';
import path from 'node:path';

export function interceptCommand(command: string): string {
    if (!isSandboxEnabled()) return command;

    const platform = os.platform();
    const config = getSandboxConfig();

    if (platform === 'darwin') {
        return wrapMacOS(command, config);
    } else if (platform === 'linux') {
        return wrapLinux(command, config);
    }

    return command;
}

function wrapMacOS(command: string, config: any): string {
    const allowWrite = config.filesystem.allowWrite || [];
    const denyRead = config.filesystem.denyRead || [];
    const denyWrite = config.filesystem.denyWrite || [];

    const profile = `
(version 1)
(deny default)
(allow process-exec)
(allow process-fork)
(allow process-info* (target same-sandbox))
(allow signal (target same-sandbox))
(allow file-read*
    (literal "/dev/null")
    (literal "/dev/zero")
    (literal "/dev/random")
    (literal "/dev/urandom")
    (subpath "/usr/lib")
    (subpath "/usr/share")
    (subpath "/usr/bin")
    (subpath "/bin")
    (subpath "/sbin")
    (subpath "/private/var/db/icu")
    (subpath "/usr/local/bin")
)
${allowWrite.map((p: string) => `(allow file-read* file-write* (subpath "${p}"))`).join('\n')}
${denyRead.map((p: string) => `(deny file-read* (subpath "${p}"))`).join('\n')}
${denyWrite.map((p: string) => `(deny file-write* (subpath "${p}"))`).join('\n')}

; Allow reading standard locations if not explicitly denied
(allow file-read*
    (subpath "/Library/Fonts")
    (subpath "/System/Library/Fonts")
    (subpath "/System/Library/Frameworks")
    (subpath "/System/Library/PrivateFrameworks")
)

${(config.network.allowedDomains && config.network.allowedDomains.length > 0) || !config.network.deniedDomains || config.network.deniedDomains.length === 0 ? "(allow network-outbound)" : "(deny network-outbound)"}
${config.network.allowUnixSockets ? "(allow network* (remote unix-socket))" : ""}
`;
    // Write profile to temporary file
    const profilePath = path.join(os.tmpdir(), `claude-sandbox-${process.pid}.sb`);
    try {
        fs.writeFileSync(profilePath, profile);
        // We don't delete it immediately as it's needed for the execution.
        // Usually, it's better to use a temporary file that gets cleaned up on process exit.
        return `sandbox-exec -f ${quote([profilePath])} ${command}`;
    } catch (e) {
        console.error("Failed to write sandbox profile:", e);
        return command;
    }
}

function wrapLinux(command: string, config: any): string {
    const args = [
        '--new-session',
        '--die-with-parent',
        '--unshare-pid',
        '--dev', '/dev',
        '--proc', '/proc'
    ];

    if (config.network.allowedDomains.length > 0 || config.network.deniedDomains.length > 0) {
        // Simple binary network isolation on Linux for now
        args.push('--unshare-net');
    }

    const allowWrite = config.filesystem.allowWrite || [];
    const denyRead = config.filesystem.denyRead || [];

    if (allowWrite.length > 0) {
        args.push('--ro-bind', '/', '/');
        for (const p of allowWrite) {
            try {
                if (fs.existsSync(p)) {
                    args.push('--bind', p, p);
                }
            } catch { }
        }
    } else {
        args.push('--bind', '/', '/');
    }

    for (const p of denyRead) {
        try {
            if (fs.existsSync(p)) {
                if (fs.statSync(p).isDirectory()) {
                    args.push('--tmpfs', p);
                } else {
                    args.push('--ro-bind', '/dev/null', p);
                }
            }
        } catch { }
    }

    const shell = process.env.SHELL || '/bin/bash';
    return quote(['bwrap', ...args, '--', shell, '-c', command]);
}
