import { exec } from 'node:child_process';
import { promisify } from 'node:util';
import { platform } from 'node:os';
import { EnvService } from '../config/EnvService.js';

const execAsync = promisify(exec);

export interface DiagnosticInfo {
    installationType: string;
    version: string;
    installationPath: string;
    invokedBinary: string;
    configInstallMethod: string;
    autoUpdates: string;
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
    sessionMetrics?: {
        totalCostUSD: number;
        inputTokens: number;
        outputTokens: number;
    };
}

/**
 * Service to gather system diagnostics, aligned with 2.1.19 gold reference.
 * Logic mirrors Zt() from chunk1138 and GitHub checks from chunk1271.
 */
export class DoctorService {
    static async getDiagnosticInfo(): Promise<DiagnosticInfo> {
        const version = "2.1.19-deob";
        const installationPath = process.argv[1] || "unknown";
        const invokedBinary = process.argv[0] || "node";
        const installationType = process.env.npm_config_global ? "npm-global" : "native";

        const warnings: Array<{ issue: string; fix: string }> = [];

        // Ripgrep status
        let rgStatus = { workingDirectory: true, mode: "system", systemPath: null as string | null };
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
        } catch (e) {
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
            configInstallMethod: "direct",
            autoUpdates: "enabled",
            hasUpdatePermissions: true,
            multipleInstallations: [],
            warnings,
            ripgrepStatus: rgStatus,
            ghStatus,
            gitStatus,
            envVars,
            sessionMetrics
        };
    }
}
