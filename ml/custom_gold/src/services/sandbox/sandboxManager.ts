import * as fs from "node:fs";
import * as os from "node:os";
import { wrapLinuxCommand, startLinuxBridges, LinuxBridge } from "./linuxSandbox.js";
import { getMacReadProfile, getMacWriteProfile } from "./macosSandbox.js";
import { generateSandboxEnv } from "./envGenerator.js";
import { createHttpProxy, createSocksProxy } from "./sandboxProxy.js";
import { resolvePath } from "./pathResolver.js";
import { monitorMacosViolations, ViolationManager } from "./violationMonitor.js";

/**
 * Main sandbox orchestration.
 * Deobfuscated from chunk_221.ts.
 */

export interface SandboxConfig {
    network: {
        allowedDomains: string[];
        deniedDomains: string[];
        httpProxyPort?: number;
        socksProxyPort?: number;
        allowUnixSockets?: string[];
        allowAllUnixSockets?: boolean;
        allowLocalBinding?: boolean;
    };
    filesystem: {
        allowRead: string[];
        denyRead: string[];
        allowWrite: string[];
        denyWrite: string[];
        allowGitConfig?: boolean;
    };
    ignoreViolations?: Record<string, string[]>;
    enableWeakerNestedSandbox?: boolean;
    allowPty?: boolean;
    mandatoryDenySearchDepth?: number;
    ripgrep?: {
        command: string;
        args?: string[];
    };
}

let activeConfig: SandboxConfig | undefined;
let infrastructurePromise: Promise<Infrastructure> | undefined;
let activeInfrastructure: Infrastructure | undefined;
let violationMonitorCleanup: (() => void) | undefined;
export const violationManager = new ViolationManager();

interface Infrastructure {
    httpProxyPort: number;
    socksProxyPort: number;
    linuxBridge?: LinuxBridge;
}

/**
 * Generates a full macOS sandbox-exec profile.
 * Deobfuscated from U93 in chunk_221.ts.
 */
async function generateMacosProfile(config: SandboxConfig, logTag: string): Promise<string> {
    const { network, filesystem, allowPty } = config;

    const profile = [
        "(version 1)",
        `(deny default (with message "${logTag}"))`,
        "",
        ";; Essential permissions",
        "(allow process-exec)",
        "(allow process-fork)",
        "(allow process-info* (target same-sandbox))",
        "(allow signal (target same-sandbox))",
        "(allow mach-priv-task-port (target same-sandbox))",
        "(allow user-preference-read)",
        "",
        ";; Mach IPC",
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
        ";; POSIX IPC",
        "(allow ipc-posix-shm)",
        "(allow ipc-posix-sem)",
        "",
        ";; IOKit",
        "(allow iokit-open",
        '  (iokit-registry-entry-class "IOSurfaceRootUserClient")',
        '  (iokit-registry-entry-class "RootDomainUserClient")',
        '  (iokit-user-client-class "IOSurfaceSendRight")',
        ")",
        "(allow iokit-get-properties)",
        "",
        ";; System Sockets and sysctl",
        "(allow system-socket (require-all (socket-domain AF_SYSTEM) (socket-protocol 2)))",
        "(allow sysctl-read",
        '  (sysctl-name "hw.activecpu")',
        '  (sysctl-name "hw.busfrequency_compat")',
        '  (sysctl-name "hw.byteorder")',
        '  (sysctl-name "hw.cacheconfig")',
        '  (sysctl-name "hw.cachelinesize_compat")',
        '  (sysctl-name "hw.cpufamily")',
        '  (sysctl-name "hw.cpufrequency")',
        '  (sysctl-name "hw.cpufrequency_compat")',
        '  (sysctl-name "hw.cputype")',
        '  (sysctl-name "hw.l1dcachesize_compat")',
        '  (sysctl-name "hw.l1icachesize_compat")',
        '  (sysctl-name "hw.l2cachesize_compat")',
        '  (sysctl-name "hw.l3cachesize_compat")',
        '  (sysctl-name "hw.logicalcpu")',
        '  (sysctl-name "hw.logicalcpu_max")',
        '  (sysctl-name "hw.machine")',
        '  (sysctl-name "hw.memsize")',
        '  (sysctl-name "hw.ncpu")',
        '  (sysctl-name "hw.nperflevels")',
        '  (sysctl-name "hw.packages")',
        '  (sysctl-name "hw.pagesize_compat")',
        '  (sysctl-name "hw.pagesize")',
        '  (sysctl-name "hw.physicalcpu")',
        '  (sysctl-name "hw.physicalcpu_max")',
        '  (sysctl-name "hw.tbfrequency_compat")',
        '  (sysctl-name "hw.vectorunit")',
        '  (sysctl-name "kern.argmax")',
        '  (sysctl-name "kern.bootargs")',
        '  (sysctl-name "kern.hostname")',
        '  (sysctl-name "kern.maxfiles")',
        '  (sysctl-name "kern.maxfilesperproc")',
        '  (sysctl-name "kern.maxproc")',
        '  (sysctl-name "kern.ngroups")',
        '  (sysctl-name "kern.osproductversion")',
        '  (sysctl-name "kern.osrelease")',
        '  (sysctl-name "kern.ostype")',
        '  (sysctl-name "kern.osvariant_status")',
        '  (sysctl-name "kern.osversion")',
        '  (sysctl-name "kern.secure_kernel")',
        '  (sysctl-name "kern.tcsm_available")',
        '  (sysctl-name "kern.tcsm_enable")',
        '  (sysctl-name "kern.usrstack64")',
        '  (sysctl-name "kern.version")',
        '  (sysctl-name "kern.willshutdown")',
        '  (sysctl-name "machdep.cpu.brand_string")',
        '  (sysctl-name "machdep.ptrauth_enabled")',
        '  (sysctl-name "security.mac.lockdown_mode_state")',
        '  (sysctl-name "sysctl.proc_cputype")',
        '  (sysctl-name "vm.loadavg")',
        '  (sysctl-name-prefix "hw.optional.arm")',
        '  (sysctl-name-prefix "hw.optional.arm.")',
        '  (sysctl-name-prefix "hw.optional.armv8_")',
        '  (sysctl-name-prefix "hw.perflevel")',
        '  (sysctl-name-prefix "kern.proc.pgrp.")',
        '  (sysctl-name-prefix "kern.proc.pid.")',
        '  (sysctl-name-prefix "machdep.cpu.")',
        '  (sysctl-name-prefix "net.routetable.")',
        ")",
        "(allow sysctl-write (sysctl-name \"kern.tcsm_enable\"))",
        "(allow distributed-notification-post)",
        "",
        ";; Fixed devices",
        '(allow file-read* file-write* (literal "/dev/null") (literal "/dev/zero") (literal "/dev/random") (literal "/dev/urandom") (literal "/dev/dtracehelper") (literal "/dev/tty"))',
        '(allow file-ioctl (literal "/dev/null") (literal "/dev/zero") (literal "/dev/random") (literal "/dev/urandom") (literal "/dev/dtracehelper") (literal "/dev/tty"))',
        '(allow file-ioctl file-read-data file-write-data (require-all (literal "/dev/null") (vnode-type CHARACTER-DEVICE)))'
    ];

    // Network
    if (!network.allowedDomains.length && !network.deniedDomains.length && !network.httpProxyPort && !network.socksProxyPort) {
        profile.push("(allow network*)");
    } else {
        if (network.allowLocalBinding) {
            profile.push('(allow network-bind (local ip "localhost:*"))');
            profile.push('(allow network-inbound (local ip "localhost:*"))');
            profile.push('(allow network-outbound (local ip "localhost:*"))');
        }
        if (network.allowAllUnixSockets) {
            profile.push('(allow network* (subpath "/"))');
        }
        if (network.httpProxyPort) {
            profile.push(`(allow network-bind (local ip "localhost:${network.httpProxyPort}"))`);
            profile.push(`(allow network-inbound (local ip "localhost:${network.httpProxyPort}"))`);
            profile.push(`(allow network-outbound (remote ip "localhost:${network.httpProxyPort}"))`);
        }
        if (network.socksProxyPort) {
            profile.push(`(allow network-bind (local ip "localhost:${network.socksProxyPort}"))`);
            profile.push(`(allow network-inbound (local ip "localhost:${network.socksProxyPort}"))`);
            profile.push(`(allow network-outbound (remote ip "localhost:${network.socksProxyPort}"))`);
        }
    }

    // Filesystem
    profile.push("; Read access");
    profile.push(getMacReadProfile(filesystem.denyRead, logTag));

    profile.push("; Write access");
    const writeProfile = await getMacWriteProfile(
        filesystem.allowWrite,
        filesystem.denyWrite,
        logTag
    );
    profile.push(writeProfile);

    // PTY
    if (allowPty) {
        profile.push("(allow pseudo-tty)");
        profile.push('(allow file-ioctl (literal "/dev/ptmx") (regex #"^/dev/ttys"))');
    }

    return profile.join("\n");
}

/**
 * Initializes the proxy infrastructure.
 */
export async function initializeSandbox(
    config: SandboxConfig,
    onPermissionAsk?: (req: { host: string, port: number }) => Promise<boolean>
): Promise<Infrastructure> {
    if (infrastructurePromise) return infrastructurePromise;

    activeConfig = config;

    infrastructurePromise = (async () => {
        const filter = {
            filter: async (port: number, host: string) => {
                // Simple domain matching logic from MEB
                const isMatch = (domain: string, pattern: string) => {
                    if (pattern.startsWith("*.")) return domain.endsWith(pattern.slice(1));
                    return domain === pattern;
                };

                if (config.network.deniedDomains.some(d => isMatch(host, d))) return false;
                if (config.network.allowedDomains.some(d => isMatch(host, d))) return true;

                if (onPermissionAsk) {
                    return await onPermissionAsk({ host, port });
                }
                return false;
            }
        };

        const httpProxy = createHttpProxy(filter);
        const socksProxy = createSocksProxy(filter);

        const httpPort = await new Promise<number>((resolve) => {
            httpProxy.listen(0, "127.0.0.1", () => resolve((httpProxy.address() as any).port));
        });

        const socksPort = await socksProxy.listen(0, "127.0.0.1");

        let bridge: LinuxBridge | undefined;
        if (os.platform() === "linux") {
            bridge = await startLinuxBridges(httpPort, socksPort);
        }

        if (os.platform() === "darwin" && config.ignoreViolations) {
            violationMonitorCleanup = monitorMacosViolations(
                v => violationManager.addViolation(v),
                { ignoreViolations: config.ignoreViolations }
            );
        }

        const infra = {
            httpProxyPort: httpPort,
            socksProxyPort: socksPort,
            linuxBridge: bridge
        };
        activeInfrastructure = infra;
        return infra;
    })();

    return infrastructurePromise;
}

/**
 * Main command wrapper.
 */
export async function wrapCommand(
    command: string,
    options: { binShell?: string, abortSignal?: AbortSignal } = {}
): Promise<string> {
    if (!activeConfig || !activeInfrastructure) {
        throw new Error("Sandbox not initialized");
    }

    const platform = os.platform();
    const logTag = `SBX_${Math.random().toString(36).slice(2, 8).toUpperCase()}`;

    if (platform === "darwin") {
        const profile = await generateMacosProfile(activeConfig, logTag);
        const env = generateSandboxEnv(activeInfrastructure.httpProxyPort, activeInfrastructure.socksProxyPort);

        // Command encoding for violation matching (CMD64_..._END)
        const encoded = Buffer.from(command.slice(0, 100)).toString("base64");
        const wrapped = `echo CMD64_${encoded}_END_${logTag} && export ${env.join(" ")} && ${command}`;

        return `sandbox-exec -p ${JSON.stringify(profile)} ${options.binShell || "bash"} -c ${JSON.stringify(wrapped)}`;
    }

    if (platform === "linux") {
        return wrapLinuxCommand(command, {
            restrictNetwork: activeConfig.network.allowedDomains.length > 0,
            bridge: activeInfrastructure.linuxBridge,
            readDeny: activeConfig.filesystem.denyRead,
            writeAllow: activeConfig.filesystem.allowWrite,
            writeDeny: activeConfig.filesystem.denyWrite,
            binShell: options.binShell,
            abortSignal: options.abortSignal
        } as any);
    }

    throw new Error(`Platform ${platform} not supported for sandboxing`);
}

/**
 * Updates the sandbox configuration.
 */
export function updateSandboxConfig(config: SandboxConfig) {
    activeConfig = config;
}

/**
 * Cleans up sandbox infrastructure.
 * Deobfuscated from Wn1 in chunk_222.ts.
 */
export async function cleanupSandbox() {
    if (violationMonitorCleanup) {
        violationMonitorCleanup();
        violationMonitorCleanup = undefined;
    }

    const cleanupTasks: Promise<void>[] = [];

    if (activeInfrastructure?.linuxBridge) {
        const {
            httpSocketPath,
            socksSocketPath,
            httpBridgeProcess,
            socksBridgeProcess
        } = activeInfrastructure.linuxBridge;

        const killProcess = (p: any) => {
            if (p.pid && !p.killed) {
                try {
                    process.kill(p.pid, "SIGTERM");
                    return new Promise<void>((resolve) => {
                        p.once("exit", resolve);
                        setTimeout(() => {
                            if (!p.killed && p.pid) {
                                try { process.kill(p.pid, "SIGKILL"); } catch { }
                            }
                            resolve();
                        }, 5000);
                    });
                } catch { }
            }
            return Promise.resolve();
        };

        cleanupTasks.push(killProcess(httpBridgeProcess));
        cleanupTasks.push(killProcess(socksBridgeProcess));

        await Promise.all(cleanupTasks);

        // Clean up Unix sockets
        if (httpSocketPath && fs.existsSync(httpSocketPath)) fs.rmSync(httpSocketPath, { force: true });
        if (socksSocketPath && fs.existsSync(socksSocketPath)) fs.rmSync(socksSocketPath, { force: true });
    }

    activeInfrastructure = undefined;
    infrastructurePromise = undefined;
}
