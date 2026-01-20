import { platform, release, homedir } from "node:os";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import * as fs from "node:fs";
import * as path from "node:path";
import { getSettings } from "./settings.js";
// import { getVersion } from "../../utils/version.js";
const getVersion = () => "2.0.76";

const execFileAsync = promisify(execFile);

export interface DiagnosticsInfo {
    version: string;
    installationType: string;
    installationPath: string;
    invokedBinary: string;
    packageManager: string | null;
    configInstallMethod: string;
    autoUpdates: string;
    hasUpdatePermissions: boolean | null;
    ripgrepStatus: {
        working: boolean;
        mode: "vendor" | "system" | "builtin";
        systemPath?: string;
    };
    recommendation?: string;
    multipleInstallations: Array<{ type: string; path: string }>;
    warnings: Array<{ issue: string; fix: string }>;
}

/**
 * Gets the installation type based on path and environment. (Bs in chunk_588)
 */
function getInstallationType(installPath: string): string {
    if (installPath.includes('node_modules')) {
        if (installPath.includes('.claude')) return 'npm-local';
        return 'npm-global';
    }
    if (installPath.includes('bin') || installPath.includes('Applications')) return 'native';
    return 'development';
}

/**
 * Gets diagnostics information for the Claude Code installation.
 * Based on n4A in chunk_601.ts and related logic in chunk_588.ts.
 */
export async function getDiagnosticsInfo(): Promise<DiagnosticsInfo> {
    const installPath = process.argv[1] || '';
    const invokedBinary = process.argv[0] || '';
    const settings = getSettings('userSettings');

    const info: DiagnosticsInfo = {
        version: getVersion(),
        installationType: getInstallationType(installPath),
        installationPath: installPath,
        invokedBinary,
        packageManager: null,
        configInstallMethod: settings.installMethod || "not set",
        autoUpdates: settings.autoUpdates === false ? "disabled" : "enabled",
        hasUpdatePermissions: null,
        ripgrepStatus: {
            working: false,
            mode: "vendor"
        },
        multipleInstallations: [],
        warnings: []
    };

    // 1. Detect package manager
    if (installPath.includes("pnpm")) info.packageManager = "pnpm";
    else if (installPath.includes("yarn")) info.packageManager = "yarn";
    else if (installPath.includes("npm")) info.packageManager = "npm";

    // 2. Check ripgrep (Logic from chunk_601)
    try {
        const { stdout } = await execFileAsync("rg", ["--version"]);
        info.ripgrepStatus.working = true;
        info.ripgrepStatus.mode = "system";
        // parse version if needed
    } catch {
        // Fallback to vendor check
        const vendorPath = path.join(path.dirname(info.installationPath), 'vendor', 'rg');
        if (fs.existsSync(vendorPath)) {
            info.ripgrepStatus.working = true;
            info.ripgrepStatus.mode = "vendor";
        }
    }

    // 3. Check for multiple installations (hJ7 in chunk_588)
    // For now, returning empty as it requires scanning PATH

    // 4. Update permissions check
    if (platform() !== "win32") {
        try {
            const dir = path.dirname(installPath);
            fs.accessSync(dir, fs.constants.W_OK);
            info.hasUpdatePermissions = true;
        } catch {
            info.hasUpdatePermissions = false;
        }
    }

    // 5. Warnings (gJ7 in chunk_588)
    if (info.installationType === 'npm-global' && info.hasUpdatePermissions === false) {
        info.warnings.push({
            issue: "Insufficient permissions for global npm update",
            fix: "Consider using 'claude install' for a native installation in your home directory, or fix npm permissions."
        });
    }

    if (info.configInstallMethod !== 'native' && info.installationType === 'native') {
        info.warnings.push({
            issue: "Installation type mismatch",
            fix: "Configuration is not set to 'native' but you are running the native binary. Run 'claude update' to sync."
        });
    }

    return info;
}
