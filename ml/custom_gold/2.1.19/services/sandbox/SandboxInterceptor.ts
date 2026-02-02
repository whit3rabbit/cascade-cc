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
    const allowUnixSockets = config.network.allowUnixSockets || false;

    // Build subpaths for allow/deny
    const formatSubpath = (p: string) => `(subpath "${path.resolve(p)}")`;

    const profile = `
(version 1)
(deny default)

; Process permissions
(allow process-exec)
(allow process-fork)
(allow process-info* (target same-sandbox))
(allow signal (target same-sandbox))
(allow mach-priv-task-port (target same-sandbox))

; User preferences
(allow user-preference-read)

; Mach IPC
(allow mach-lookup
    (global-name "com.apple.audio.systemsoundserver")
    (global-name "com.apple.distributed_notifications@Uv3")
    (global-name "com.apple.FontObjectsServer")
    (global-name "com.apple.fonts")
    (global-name "com.apple.logd")
    (global-name "com.apple.lsd.mapdb")
    (global-name "com.apple.PowerManagement.control")
    (global-name "com.apple.system.logger")
    (global-name "com.apple.system.notification_center")
    (global-name "com.apple.trustd.agent")
    (global-name "com.apple.system.opendirectoryd.libinfo")
    (global-name "com.apple.system.opendirectoryd.membership")
    (global-name "com.apple.bsd.dirhelper")
    (global-name "com.apple.securityd.xpc")
    (global-name "com.apple.coreservices.launchservicesd")
    (global-name "com.apple.SecurityServer")
)

; POSIX IPC
(allow ipc-posix-shm)
(allow ipc-posix-sem)

; IOKit
(allow iokit-open
    (iokit-registry-entry-class "IOSurfaceRootUserClient")
    (iokit-registry-entry-class "RootDomainUserClient")
    (iokit-user-client-class "IOSurfaceSendRight")
)
(allow iokit-get-properties)

; System Sockets (no general network)
(allow system-socket (require-all (socket-domain AF_SYSTEM) (socket-protocol 2)))

; sysctl
(allow sysctl-read
    (sysctl-name "hw.activecpu")
    (sysctl-name "hw.busfrequency_compat")
    (sysctl-name "hw.byteorder")
    (sysctl-name "hw.cacheconfig")
    (sysctl-name "hw.cachelinesize_compat")
    (sysctl-name "hw.cpufamily")
    (sysctl-name "hw.cpufrequency")
    (sysctl-name "hw.cpufrequency_compat")
    (sysctl-name "hw.cputype")
    (sysctl-name "hw.l1dcachesize_compat")
    (sysctl-name "hw.l1icachesize_compat")
    (sysctl-name "hw.l2cachesize_compat")
    (sysctl-name "hw.l3cachesize_compat")
    (sysctl-name "hw.logicalcpu")
    (sysctl-name "hw.logicalcpu_max")
    (sysctl-name "hw.machine")
    (sysctl-name "hw.memsize")
    (sysctl-name "hw.ncpu")
    (sysctl-name "hw.nperflevels")
    (sysctl-name "hw.packages")
    (sysctl-name "hw.pagesize_compat")
    (sysctl-name "hw.pagesize")
    (sysctl-name "hw.physicalcpu")
    (sysctl-name "hw.physicalcpu_max")
    (sysctl-name "hw.tbfrequency_compat")
    (sysctl-name "hw.vectorunit")
    (sysctl-name "kern.argmax")
    (sysctl-name "kern.bootargs")
    (sysctl-name "kern.hostname")
    (sysctl-name "kern.maxfiles")
    (sysctl-name "kern.maxfilesperproc")
    (sysctl-name "kern.maxproc")
    (sysctl-name "kern.ngroups")
    (sysctl-name "kern.osproductversion")
    (sysctl-name "kern.osrelease")
    (sysctl-name "kern.ostype")
    (sysctl-name "kern.osvariant_status")
    (sysctl-name "kern.osversion")
    (sysctl-name "kern.secure_kernel")
    (sysctl-name "kern.tcsm_available")
    (sysctl-name "kern.tcsm_enable")
    (sysctl-name "kern.usrstack64")
    (sysctl-name "kern.version")
    (sysctl-name "kern.willshutdown")
    (sysctl-name "machdep.cpu.brand_string")
    (sysctl-name "machdep.ptrauth_enabled")
    (sysctl-name "security.mac.lockdown_mode_state")
    (sysctl-name "sysctl.proc_cputype")
    (sysctl-name "vm.loadavg")
    (sysctl-name-prefix "hw.optional.arm")
    (sysctl-name-prefix "hw.optional.arm.")
    (sysctl-name-prefix "hw.optional.armv8_")
    (sysctl-name-prefix "hw.perflevel")
    (sysctl-name-prefix "kern.proc.all")
    (sysctl-name-prefix "kern.proc.pgrp.")
    (sysctl-name-prefix "kern.proc.pid.")
    (sysctl-name-prefix "machdep.cpu.")
    (sysctl-name-prefix "net.routetable.")
)
(allow sysctl-write (sysctl-name "kern.tcsm_enable"))

; Distributed notifications
(allow distributed-notification-post)

; Device files
(allow file-ioctl (literal "/dev/null") (literal "/dev/zero") (literal "/dev/random") (literal "/dev/urandom") (literal "/dev/dtracehelper") (literal "/dev/tty"))
(allow file-read* file-write* (require-all (literal "/dev/null") (vnode-type CHARACTER-DEVICE)))

; Filesystem
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
    (subpath "/Library/Fonts")
    (subpath "/System/Library/Fonts")
    (subpath "/System/Library/Frameworks")
    (subpath "/System/Library/PrivateFrameworks")
)

${allowWrite.map((p: string) => `(allow file-read* file-write* ${formatSubpath(p)})`).join('\n')}
${denyRead.map((p: string) => `(deny file-read* ${formatSubpath(p)})`).join('\n')}
${denyWrite.map((p: string) => `(deny file-write* ${formatSubpath(p)})`).join('\n')}

; Pseudo-terminal
(allow pseudo-tty)
(allow file-ioctl (literal "/dev/ptmx") (regex #"^/dev/ttys"))
(allow file-read* file-write* (literal "/dev/ptmx") (regex #"^/dev/ttys"))

; Network
${(config.network.allowedDomains && config.network.allowedDomains.length > 0) || !config.network.deniedDomains || config.network.deniedDomains.length === 0 ? "(allow network-outbound)" : "(deny network-outbound)"}
${allowUnixSockets ? "(allow network* (remote unix-socket))" : ""}
`;
    // Write profile to temporary file
    const profilePath = path.join(os.tmpdir(), `claude-sandbox-${process.pid}.sb`);
    try {
        fs.writeFileSync(profilePath, profile);
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
        '--proc', '/proc',
        '--unshare-user'
    ];

    const needsNetworkRestriction = (config.network.allowedDomains?.length > 0 || config.network.deniedDomains?.length > 0);
    if (needsNetworkRestriction) {
        args.push('--unshare-net');
        // If we had the bridge logic, we would bind the sockets here.
        // For now, we unshare and block network completely unless a bridge is provided.
    }

    const allowWrite = config.filesystem.allowWrite || [];
    const denyRead = config.filesystem.denyRead || [];

    // Basic system directories (read-only)
    const commonReadOnly = ['/usr', '/bin', '/sbin', '/lib', '/lib64', '/etc/alternatives', '/etc/ssl/certs'];
    for (const p of commonReadOnly) {
        if (fs.existsSync(p)) {
            args.push('--ro-bind-try', p, p);
        }
    }

    // Bind current working directory as project root
    const cwd = process.cwd();
    args.push('--bind', cwd, cwd);

    if (allowWrite.length > 0) {
        for (const p of allowWrite) {
            const resolved = path.resolve(p);
            if (fs.existsSync(resolved)) {
                args.push('--bind-try', resolved, resolved);
            }
        }
    }

    for (const p of denyRead) {
        const resolved = path.resolve(p);
        if (fs.existsSync(resolved)) {
            if (fs.statSync(resolved).isDirectory()) {
                args.push('--tmpfs', resolved);
            } else {
                args.push('--ro-bind', '/dev/null', resolved);
            }
        }
    }

    // Minimal home
    args.push('--tmpfs', os.homedir());

    const shell = process.env.SHELL || '/bin/bash';
    return quote(['bwrap', ...args, '--', shell, '-c', command]);
}
