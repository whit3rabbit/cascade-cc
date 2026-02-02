/**
 * File: src/services/sandbox/SandboxInterceptor.ts
 * Role: Intercepts and wraps shell commands in a sandbox.
 * Derived from chunk234 and chunk231.
 */

import { MACOS_SBPL_PROFILE, LINUX_BWRAP_BASE_ARGS, SandboxOptions } from "./SandboxSettings.js";

/**
 * Intercepts and wraps a shell command in a sandbox based on the current OS.
 */
export function interceptCommand(command: string, options: SandboxOptions = {}): string {
    const platform = process.platform;

    if (platform === "darwin") {
        return wrapMacOS(command, options);
    } else if (platform === "linux") {
        return wrapLinux(command, options);
    }

    // Fallback for unsupported platforms or if no restrictions are needed
    return command;
}

function wrapMacOS(command: string, options: SandboxOptions): string {
    let profile = MACOS_SBPL_PROFILE;

    // Add dynamic rules to the profile based on options
    if (options.readAllowPaths) {
        options.readAllowPaths.forEach(path => {
            profile += `\n(allow file-read* (subpath "${path}"))`;
        });
    }
    if (options.writeAllowPaths) {
        options.writeAllowPaths.forEach(path => {
            profile += `\n(allow file-write* (subpath "${path}"))`;
        });
    }

    if (!options.needsNetworkRestriction) {
        profile += "\n(allow network*)";
    }

    // Use a temporary file for the profile or pass it via stdin if possible.
    // However, the original code seems to pass it as a string or a generated file path.
    // For simplicity in this implementation, we assume we can pass a profile string.
    // Note: sandbox-exec -p takes the profile content directly.
    const shell = process.env.SHELL || "bash";

    // Escaping the profile and command for the final shell command
    const escapedProfile = profile.replace(/"/g, '\\"').replace(/\n/g, ' ');
    return `sandbox-exec -p "${escapedProfile}" ${shell} -c "${command.replace(/"/g, '\\"')}"`;
}

function wrapLinux(command: string, options: SandboxOptions): string {
    const args = [...LINUX_BWRAP_BASE_ARGS];

    if (options.needsNetworkRestriction) {
        args.push("--unshare-net");
    }

    if (options.readAllowPaths) {
        options.readAllowPaths.forEach(path => {
            args.push("--bind", path, path);
        });
    }

    if (options.writeAllowPaths) {
        options.writeAllowPaths.forEach(path => {
            args.push("--bind", path, path);
        });
    }

    const shell = process.env.SHELL || "bash";
    return `bwrap ${args.join(" ")} -- ${shell} -c "${command.replace(/"/g, '\\"')}"`;
}
