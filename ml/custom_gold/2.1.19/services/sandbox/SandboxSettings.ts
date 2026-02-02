/**
 * File: src/services/sandbox/SandboxSettings.ts
 * Role: Configuration for sandbox profiles and security policies.
 * Derived from chunk233 and chunk231.
 */

export const MACOS_SBPL_PROFILE = `
(version 1)
(deny default)

; Essential permissions
(allow process-exec)
(allow process-fork)
(allow process-info* (target same-sandbox))
(allow signal (target same-sandbox))
(allow mach-priv-task-port (target same-sandbox))

; User preferences
(allow user-preference-read)

; Mach IPC - specific services only
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

; Safe system sockets (no network)
(allow system-socket (require-all (socket-domain AF_SYSTEM) (socket-protocol 2)))

; sysctl - restricted set
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

(allow sysctl-write
  (sysctl-name "kern.tcsm_enable")
)

; Generic file IO for safe devices
(allow file-ioctl (literal "/dev/null"))
(allow file-ioctl (literal "/dev/zero"))
(allow file-ioctl (literal "/dev/random"))
(allow file-ioctl (literal "/dev/urandom"))
(allow file-ioctl (literal "/dev/dtracehelper"))
(allow file-ioctl (literal "/dev/tty"))

(allow file-read-data file-write-data
  (require-all (literal "/dev/null") (vnode-type CHARACTER-DEVICE))
)
`;

export const LINUX_BWRAP_BASE_ARGS = [
    "--new-session",
    "--die-with-parent",
    "--dev", "/dev",
    "--unshare-pid",
    "--proc", "/proc"
];

export interface SandboxOptions {
    needsNetworkRestriction?: boolean;
    readAllowPaths?: string[];
    readDenyPaths?: string[];
    writeAllowPaths?: string[];
    writeDenyPaths?: string[];
    allowUnixSockets?: boolean;
}

/**
 * Checks if the sandbox is enabled via environment variables.
 */
export function isSandboxEnabled(): boolean {
    return process.env.CLAUDE_CODE_DISABLE_SANDBOX !== 'true';
}

/**
 * Checks if unsandboxed (dangerous) commands are allowed.
 */
export function areUnsandboxedCommandsAllowed(): boolean {
    return process.env.CLAUDE_CODE_ALLOW_DANGEROUS_COMMANDS === 'true';
}

/**
 * Checks if a URL is allowed to be fetched based on the sandbox policy.
 */
export function isUrlAllowed(url: string): boolean {
    // Basic policy: allow most public URLs but block local/private network by default
    // In a real implementation, this would be more sophisticated.
    try {
        const parsed = new URL(url);
        const host = parsed.hostname.toLowerCase();

        // Block localhost and private IPs
        if (host === 'localhost' || host === '127.0.0.1' || host === '::1') {
            return false;
        }

        // Example: block internal domains
        if (host.endsWith('.internal')) {
            return false;
        }

        return true;
    } catch {
        return false;
    }
}
