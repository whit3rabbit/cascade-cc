/**
 * File: src/utils/shared/sandbox.ts
 * Role: Generates sandbox policies for secure execution of commands on macOS (sandbox-exec).
 */

export interface SandboxOptions {
    readConfig: boolean | string[];
    writeConfig: boolean | string[];
    httpProxyPort?: number;
    socksProxyPort?: number;
    needsNetworkRestriction?: boolean;
    allowUnixSockets?: string[];
    allowAllUnixSockets?: boolean;
    allowLocalBinding?: boolean;
    allowPty?: boolean;
    allowGitConfig?: boolean;
    logTag: string;
}

/**
 * Escapes a path for use in the sandbox policy string.
 */
function escapePath(path: string): string {
    return JSON.stringify(path);
}

/**
 * Generates file read permissions for the policy.
 */
function generateReadPermissions(config: boolean | string[]): string[] {
    if (!config) return [];
    if (Array.isArray(config)) {
        return config.map(p => `(allow file-read* (literal ${escapePath(p)}))`);
    }
    return ["(allow file-read*)"];
}

/**
 * Generates file write permissions for the policy.
 */
function generateWritePermissions(config: boolean | string[], allowGitConfig?: boolean): string[] {
    const lines: string[] = [];
    if (config) {
        if (Array.isArray(config)) {
            config.forEach(p => lines.push(`(allow file-write* (literal ${escapePath(p)}))`));
        } else {
            lines.push("(allow file-write*)");
        }
    }
    if (allowGitConfig) {
        lines.push('(allow file-write* (regex #"\\.git/config$"))');
    }
    return lines;
}

/**
 * Generates a macOS sandbox policy string.
 */
export function generateSandboxPolicy(options: SandboxOptions): string {
    const {
        readConfig,
        writeConfig,
        httpProxyPort,
        socksProxyPort,
        needsNetworkRestriction,
        allowUnixSockets,
        allowAllUnixSockets,
        allowLocalBinding,
        allowPty,
        allowGitConfig = false,
        logTag
    } = options;

    const lines = [
        "(version 1)",
        `(deny default (with message "${logTag}"))`,
        "",
        `; LogTag: ${logTag}`,
        "",
        "; Essential process and system permissions",
        "(allow process-exec)",
        "(allow process-fork)",
        "(allow process-info* (target same-sandbox))",
        "(allow signal (target same-sandbox))",
        "(allow mach-priv-task-port (target same-sandbox))",
        "(allow user-preference-read)",
        "",
        "; Mach IPC - Common services",
        "(allow mach-lookup",
        '  (global-name "com.apple.audio.systemsoundserver")',
        '  (global-name "com.apple.distributed_notifications@Uv3")',
        '  (global-name "com.apple.FontObjectsServer")',
        '  (global-name "com.apple.fonts")',
        '  (global-name "com.apple.logd")',
        '  (global-name "com.apple.lsd.mapdb")',
        '  (global-name "com.apple.PowerManagement.control")',
        '  (global-name "com.apple.system.logger")',
        '  (global-name "com.apple.system.notification_center")',
        '  (global-name "com.apple.trustd.agent")',
        '  (global-name "com.apple.system.opendirectoryd.libinfo")',
        '  (global-name "com.apple.system.opendirectoryd.membership")',
        '  (global-name "com.apple.bsd.dirhelper")',
        '  (global-name "com.apple.securityd.xpc")',
        '  (global-name "com.apple.coreservices.launchservicesd")',
        '  (global-name "com.apple.SecurityServer")',
        ")",
        "",
        "; Shared memory and semaphores",
        "(allow ipc-posix-shm)",
        "(allow ipc-posix-sem)",
        "",
        "; IOKit",
        "(allow iokit-open",
        '  (iokit-registry-entry-class "IOSurfaceRootUserClient")',
        '  (iokit-registry-entry-class "RootDomainUserClient")',
        '  (iokit-user-client-class "IOSurfaceSendRight")',
        ")",
        "(allow iokit-get-properties)",
        "",
        "; Safe system sockets",
        "(allow system-socket (require-all (socket-domain AF_SYSTEM) (socket-protocol 2)))",
        "",
        "; Common sysctls",
        "(allow sysctl-read)", // Simplified for brevity in this conversion
        "",
        "; Basic device IO",
        '(allow file-ioctl (literal "/dev/null"))',
        '(allow file-ioctl (literal "/dev/zero"))',
        '(allow file-ioctl (literal "/dev/random"))',
        '(allow file-ioctl (literal "/dev/urandom"))',
        '(allow file-ioctl (literal "/dev/tty"))',
        "",
        "; Network logic"
    ];

    if (!needsNetworkRestriction) {
        lines.push("(allow network*)");
    } else {
        if (allowLocalBinding) {
            lines.push('(allow network-bind (local ip "localhost:*"))');
            lines.push('(allow network-inbound (local ip "localhost:*"))');
            lines.push('(allow network-outbound (local ip "localhost:*"))');
        }
        if (allowAllUnixSockets) {
            lines.push('(allow network* (subpath "/"))');
        } else if (allowUnixSockets && allowUnixSockets.length > 0) {
            allowUnixSockets.forEach(s => lines.push(`(allow network* (subpath ${escapePath(s)}))`));
        }

        if (httpProxyPort !== undefined) {
            lines.push(`(allow network* (remote ip "localhost:${httpProxyPort}"))`);
        }
        if (socksProxyPort !== undefined) {
            lines.push(`(allow network* (remote ip "localhost:${socksProxyPort}"))`);
        }
    }

    lines.push("", "; File permissions");
    lines.push(...generateReadPermissions(readConfig));
    lines.push(...generateWritePermissions(writeConfig, allowGitConfig));

    if (allowPty) {
        lines.push("", "; PTY support");
        lines.push("(allow pseudo-tty)");
        lines.push('(allow file-read* file-write* (literal "/dev/ptmx") (regex #"^/dev/ttys"))');
    }

    return lines.join("\n");
}
