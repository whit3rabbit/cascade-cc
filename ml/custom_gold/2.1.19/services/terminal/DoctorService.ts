import { exec } from 'node:child_process';
import { promisify } from 'node:util';
import { platform } from 'node:os';
import { readdirSync, readFileSync } from 'node:fs';
import { join as joinPath } from 'node:path';
import { EnvService } from '../config/EnvService.js';
import { getSettings } from '../config/SettingsService.js';
import { getClaudePaths } from '../../utils/shared/runtimeAndEnv.js';
import { isProcessRunning } from '../../utils/process/ProcessLock.js';
import { UpdaterService } from '../updater/UpdaterService.js';

const execAsync = promisify(exec);

export interface DiagnosticInfo {
    installationType: string;
    version: string;
    installationPath: string;
    invokedBinary: string;
    configInstallMethod: string;
    autoUpdates: string;
    updateChannel: string;
    autoUpdatesChannel: string;
    hasUpdatePermissions: boolean | null;
    multipleInstallations: Array<{ type: string; path: string }>;
    warnings: Array<{ issue: string; fix: string }>;
    ripgrepStatus: {
        workingDirectory: boolean;
        mode: string;
        systemPath: string | null;
    };
    ghStatus: {
        installed: boolean;
        authenticated: boolean;
        scopes: string[];
    };
    gitStatus: {
        isRepo: boolean;
        originUrl: string | null;
    };
    envVars: Array<{ name: string; message: string; status: 'ok' | 'capped' | 'error' }>;
    packageManager?: string;
    stableVersion?: string;
    latestVersion?: string;
    versionLocks: Array<{ version: string; pid: number; isProcessRunning: boolean }>;
    sessionMetrics?: {
        totalCostUSD: number;
        inputTokens: number;
        outputTokens: number;
    };
}

export interface DiagnosticSummary {
    ok: number;
    warn: number;
    errorCount: number;
}

export function getDiagnosticSummary(info: DiagnosticInfo): DiagnosticSummary {
    let ok = 0;
    let warn = 0;
    let errorCount = 0;

    const bumpStatus = (status: 'ok' | 'capped' | 'error' | 'not-set') => {
        if (status === 'ok') ok++;
        else if (status === 'capped' || status === 'not-set') warn++;
        else if (status === 'error') errorCount++;
    };

    info.envVars.forEach(v => bumpStatus(v.status));
    if (info.ripgrepStatus.workingDirectory) ok++; else errorCount++;
    if (info.ghStatus.installed && info.ghStatus.authenticated) ok++; else warn++;

    return { ok, warn, errorCount };
}

export function formatDoctorReport(info: DiagnosticInfo): string {
    let report = `Diagnostics\n`;
    report += `└ Currently running: ${info.installationType} (${info.version})\n`;
    if (info.packageManager) report += `└ Package manager: ${info.packageManager}\n`;
    report += `└ Path: ${info.installationPath}\n`;
    report += `└ Invoked: ${info.invokedBinary}\n`;
    report += `└ Config install method: ${info.configInstallMethod}\n`;
    report += `└ Search: ${info.ripgrepStatus.workingDirectory ? "OK" : "Not working"} (${info.ripgrepStatus.mode === 'builtin' ? 'bundled' : (info.ripgrepStatus.systemPath || 'system')})\n`;

    report += `\nUpdates\n`;
    report += `└ Auto-updates: ${info.packageManager ? "Managed by package manager" : info.autoUpdates}\n`;
    report += `└ Auto-update channel: ${info.updateChannel}\n`;
    report += `└ Stable version: ${info.stableVersion || 'unknown'}\n`;
    report += `└ Latest version: ${info.latestVersion || 'unknown'}\n`;

    report += `\nVersion Locks\n`;
    if (info.versionLocks.length === 0) {
        report += `└ No active version locks\n`;
    } else {
        for (const lock of info.versionLocks) {
            report += `└ ${lock.version}: PID ${lock.pid} ${lock.isProcessRunning ? '(running)' : '(stale)'}\n`;
        }
    }

    report += `\nPress Enter to continue…`;
    return report;
}

/**
 * Service to gather system diagnostics, aligned with 2.1.19 gold reference.
 * Logic mirrors Zt() from chunk1138 and GitHub checks from chunk1271.
 */
export class DoctorService {
    static async getDiagnosticInfo(): Promise<DiagnosticInfo> {
        const version = UpdaterService.getCurrentVersion() || "unknown";
        const installationPath = process.argv[1] || "unknown";
        const invokedBinary = process.argv[0] || "node";
        const installationType = process.env.npm_config_global ? "npm-global" : "native";
        const configInstallMethod = installationType === "native" ? "native" : "npm";

        const warnings: Array<{ issue: string; fix: string }> = [];

        // Ripgrep status
        const useBuiltinRipgrep = EnvService.get("CLAUDE_CODE_USE_BUILTIN_RIPGREP") === "true";
        let rgStatus = { workingDirectory: true, mode: useBuiltinRipgrep ? "builtin" : "system", systemPath: null as string | null };
        try {
            const { stdout } = await execAsync('rg --version');
            if (!stdout.includes('ripgrep')) {
                rgStatus.workingDirectory = false;
            } else {
                try {
                    const { stdout: pathStdout } = await execAsync(platform() === 'win32' ? 'where rg' : 'which rg');
                    rgStatus.systemPath = pathStdout.trim();
                } catch {
                    rgStatus.systemPath = "system";
                }
            }
        } catch {
            rgStatus.workingDirectory = false;
            warnings.push({
                issue: "ripgrep not found in PATH",
                fix: "Install ripgrep for faster file searches: brew install ripgrep (macOS), apt install ripgrep (Linux/WSL)"
            });
        }

        // GitHub CLI status
        let ghStatus = { installed: false, authenticated: false, scopes: [] as string[] };
        try {
            await execAsync('gh --version');
            ghStatus.installed = true;
            try {
                const { stdout: authStdout } = await execAsync('gh auth status -a');
                ghStatus.authenticated = true;
                const scopesMatch = authStdout.match(/Token scopes: (.*)$/m);
                if (scopesMatch) {
                    ghStatus.scopes = scopesMatch[1].split(',').map(s => s.trim());
                }
            } catch {
                ghStatus.authenticated = false;
                warnings.push({
                    issue: "GitHub CLI not authenticated",
                    fix: "Run 'gh auth login' to enable integration features like /pr-comments"
                });
            }
        } catch {
            ghStatus.installed = false;
            warnings.push({
                issue: "GitHub CLI not found",
                fix: "Install GitHub CLI (gh) from https://cli.github.com/ for PR review context"
            });
        }

        // Git status
        let gitStatus = { isRepo: false, originUrl: null as string | null };
        try {
            await execAsync('git rev-parse --is-inside-work-tree');
            gitStatus.isRepo = true;
            try {
                const { stdout: originStdout } = await execAsync('git remote get-url origin');
                gitStatus.originUrl = originStdout.trim();
            } catch { }
        } catch {
            gitStatus.isRepo = false;
        }

        // Environment variables
        const envVars: Array<{ name: string; message: string; status: 'ok' | 'capped' | 'error' }> = [];
        const apiKey = EnvService.get("ANTHROPIC_API_KEY");
        if (!apiKey) {
            envVars.push({ name: "ANTHROPIC_API_KEY", message: "Missing", status: "error" });
            warnings.push({
                issue: "ANTHROPIC_API_KEY not found",
                fix: "Set the ANTHROPIC_API_KEY environment variable or run 'claude login' if implemented"
            });
        } else {
            envVars.push({ name: "ANTHROPIC_API_KEY", message: "Configured (masked)", status: "ok" });
        }

        // Update channels
        const settings = getSettings();
        const updateChannel = (settings.updateChannel || settings.autoUpdatesChannel || 'latest') as string;
        const updateInfo = await UpdaterService.getUpdateChannelInfo();

        // Version locks
        const lockDir = getClaudePaths().locks;
        let versionLocks: Array<{ version: string; pid: number; isProcessRunning: boolean }> = [];
        try {
            const entries = readdirSync(lockDir);
            for (const entry of entries) {
                if (!entry.endsWith('.lock')) continue;
                const lockPath = joinPath(lockDir, entry);
                try {
                    const raw = readFileSync(lockPath, 'utf8');
                    if (!raw.trim()) continue;
                    const parsed = JSON.parse(raw) as { pid?: number; version?: string };
                    if (typeof parsed.pid !== 'number' || !parsed.version) continue;
                    versionLocks.push({
                        pid: parsed.pid,
                        version: parsed.version,
                        isProcessRunning: isProcessRunning(parsed.pid)
                    });
                } catch {
                    continue;
                }
            }
        } catch {
            versionLocks = [];
        }

        // Session metrics
        const { costService } = await import('./CostService.js');
        const usage = costService.getUsage();
        const sessionMetrics = {
            totalCostUSD: costService.calculateCost(),
            inputTokens: usage.inputTokens,
            outputTokens: usage.outputTokens
        };

        return {
            installationType,
            version,
            installationPath,
            invokedBinary,
            configInstallMethod,
            autoUpdates: updateInfo?.latestVersion && updateInfo?.latestVersion !== version ? 'available' : 'latest',
            updateChannel,
            autoUpdatesChannel: updateChannel,
            hasUpdatePermissions: await DoctorService.checkUpdatePermissions(),
            multipleInstallations: [],
            warnings,
            ripgrepStatus: rgStatus,
            ghStatus,
            gitStatus,
            envVars,
            stableVersion: updateInfo?.stableVersion,
            latestVersion: updateInfo?.latestVersion,
            versionLocks,
            sessionMetrics
        };
    }

    private static async checkUpdatePermissions(): Promise<boolean> {
        try {
            const { access } = await import('fs/promises');
            const { constants } = await import('fs');
            const installPath = process.execPath;
            await access(installPath, constants.W_OK);
            return true;
        } catch {
            return false;
        }
    }
}
