import os from "os";

export function getUserAgent(): string {
    const version = process.env.npm_package_version || "unknown";
    return `ClaudeCode/${version} (${os.platform()}; ${os.arch()})`;
}
