import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";
import { spawn, spawnSync, ChildProcess } from "node:child_process";
import { randomBytes } from "node:crypto";
import { generateSandboxEnv } from "./envGenerator.js";
import { resolvePath } from "./pathResolver.js";
import { scanSensitiveFiles } from "./sensitiveFiles.js";
import { getSeccompFilterPath, getApplySeccompBinaryPath } from "./seccompFilter.js";

/**
 * Linux sandbox implementation using Bubblewrap (bwrap).
 * Deobfuscated from HEB, VEB, V93, H93 in chunk_220.ts.
 */

export interface LinuxBridge {
    httpSocketPath: string;
    socksSocketPath: string;
    httpBridgeProcess: ChildProcess;
    socksBridgeProcess: ChildProcess;
    httpProxyPort: number;
    socksProxyPort: number;
}

/**
 * Checks if mandatory tools for Linux sandbox are installed.
 */
export function isLinuxSandboxAvailable(): boolean {
    try {
        const bwrap = spawnSync("which", ["bwrap"], { stdio: "ignore" });
        const socat = spawnSync("which", ["socat"], { stdio: "ignore" });
        return bwrap.status === 0 && socat.status === 0;
    } catch {
        return false;
    }
}

/**
 * Starts socat bridges to map Unix sockets to the internal proxy ports.
 */
export async function startLinuxBridges(httpPort: number, socksPort: number): Promise<LinuxBridge> {
    const id = randomBytes(8).toString("hex");
    const httpSock = path.join(os.tmpdir(), `claude-http-${id}.sock`);
    const socksSock = path.join(os.tmpdir(), `claude-socks-${id}.sock`);

    const startBridge = (sock: string, port: number) => {
        return spawn("socat", [
            `UNIX-LISTEN:${sock},fork,reuseaddr`,
            `TCP:localhost:${port},keepalive,keepidle=10,keepintvl=5,keepcnt=3`
        ], { stdio: "ignore" });
    };

    const httpBridge = startBridge(httpSock, httpPort);
    const socksBridge = startBridge(socksSock, socksPort);

    // Wait for sockets to be created
    for (let i = 0; i < 10; i++) {
        if (fs.existsSync(httpSock) && fs.existsSync(socksSock)) break;
        await new Promise(r => setTimeout(r, 100));
    }

    return {
        httpSocketPath: httpSock,
        socksSocketPath: socksSock,
        httpBridgeProcess: httpBridge,
        socksBridgeProcess: socksBridge,
        httpProxyPort: httpPort,
        socksProxyPort: socksPort
    };
}

/**
 * Wraps a command with Bubblewrap and Seccomp.
 */
export async function wrapLinuxCommand(
    command: string,
    options: {
        restrictNetwork?: boolean;
        bridge?: LinuxBridge;
        readDeny?: string[];
        writeAllow?: string[];
        writeDeny?: string[];
        binShell?: string;
        abortSignal?: AbortSignal;
    } = {}
): Promise<string> {
    const {
        restrictNetwork = false,
        bridge,
        readDeny = [],
        writeAllow,
        writeDeny = [],
        binShell = "bash"
    } = options;

    const args: string[] = ["--new-session", "--die-with-parent"];

    // Basic filesystem setup
    if (!writeAllow) {
        args.push("--bind", "/", "/");
    } else {
        args.push("--ro-bind", "/", "/");
        for (const p of writeAllow) {
            const resolved = resolvePath(p);
            if (fs.existsSync(resolved)) args.push("--bind", resolved, resolved);
        }

        // Deny sensitive files even if in allowed write paths
        const sensitive = await scanSensitiveFiles();
        for (const p of [...writeDeny, ...sensitive]) {
            const resolved = resolvePath(p);
            if (fs.existsSync(resolved)) args.push("--ro-bind", resolved, resolved);
        }
    }

    // Deny read access
    for (const p of readDeny) {
        const resolved = resolvePath(p);
        if (fs.existsSync(resolved)) {
            if (fs.statSync(resolved).isDirectory()) args.push("--tmpfs", resolved);
            else args.push("--ro-bind", "/dev/null", resolved);
        }
    }

    args.push("--dev", "/dev");
    args.push("--unshare-pid");
    args.push("--proc", "/proc");

    // Network restriction
    const seccomp = getSeccompFilterPath();
    const applySeccomp = getApplySeccompBinaryPath();

    if (restrictNetwork) {
        args.push("--unshare-net");
        if (bridge) {
            args.push("--bind", bridge.httpSocketPath, bridge.httpSocketPath);
            args.push("--bind", bridge.socksSocketPath, bridge.socksSocketPath);

            const env = generateSandboxEnv(3128, 1080); // Fixed ports inside sandbox
            for (const e of env) {
                const [k, v] = e.split("=");
                args.push("--setenv", k, v);
            }
        }
    }

    // Execution
    const shell = spawnSync("which", [binShell]).stdout?.toString().trim() || "/bin/bash";
    args.push("--", shell, "-c");

    let finalCommand = command;

    // Apply Seccomp to block AF_UNIX if possible
    if (seccomp && applySeccomp) {
        // This is a simplified version of the V93/HEB wrapper
        // The actual implementation uses apply-seccomp to wrap the shell
        finalCommand = `${applySeccomp} ${seccomp} ${shell} -c ${JSON.stringify(command)}`;
    }

    if (restrictNetwork && bridge) {
        // Map internal ports back to Unix sockets using socat inside the sandbox
        const setupBridges = [
            `socat TCP-LISTEN:3128,fork,reuseaddr UNIX-CONNECT:${bridge.httpSocketPath} >/dev/null 2>&1 &`,
            `socat TCP-LISTEN:1080,fork,reuseaddr UNIX-CONNECT:${bridge.socksSocketPath} >/dev/null 2>&1 &`,
            'trap "kill %1 %2 2>/dev/null; exit" EXIT',
            finalCommand
        ].join("\n");
        return `bwrap ${args.join(" ")} ${JSON.stringify(setupBridges)}`;
    }

    return `bwrap ${args.join(" ")} ${JSON.stringify(finalCommand)}`;
}
