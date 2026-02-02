/**
 * File: src/services/terminal/DoctorService.ts
 * Role: Performs health checks on dependencies, network, and environment.
 */

import { execSync } from 'node:child_process';
import { EnvService } from '../config/EnvService.js';
import https from 'node:https';
import { resolve } from 'node:path';
import { existsSync, statSync } from 'node:fs';
import { getToolSettings } from '../config/SettingsService.js';
import { listAgents } from '../agents/AgentPersistence.js';

export interface HealthCheckResult {
    name: string;
    status: 'ok' | 'warn' | 'error';
    message: string;
    details?: string;
}

export interface InstallationInfo {
    type: 'development' | 'native' | 'npm-global' | 'npm-local' | 'package-manager' | 'unknown';
    version: string;
    path: string;
    invokedBinary: string;
    installMethod: string;
    autoUpdates: 'enabled' | 'disabled' | 'error';
    ripgrep: {
        ok: boolean;
        mode: 'system' | 'builtin' | 'missing';
        details?: string;
    };
}

export interface DoctorDiagnostics {
    installation: InstallationInfo;
    healthChecks: HealthCheckResult[];
    versionLock?: {
        locked: boolean;
        version?: string;
        source?: string;
    };
    environmentVariables: {
        name: string;
        value: string;
        status: 'valid' | 'invalid' | 'capped' | 'not-set';
        message?: string;
    }[];
    permissions: {
        unreachableRules: string[];
    };
    contextUsage: {
        claudeMdSize?: number;
        agentCount: number;
        mcpTokenCount: number;
    };
}

export class DoctorService {
    static async getDiagnostics(): Promise<DoctorDiagnostics> {
        const installation = await this.getInstallationInfo();
        const healthChecks: HealthCheckResult[] = [];

        // Dependency Checks
        healthChecks.push(this.checkBinary('git', '--version', 'Git'));
        healthChecks.push(this.checkBinary('tmux', '-V', 'tmux'));

        // Network Checks
        healthChecks.push(await this.checkConnectivity('api.anthropic.com'));
        healthChecks.push(await this.checkConnectivity('statsig.anthropic.com'));

        // Path & Conflicts
        healthChecks.push(this.checkPathHealth());
        const conflictCheck = await this.checkMultipleInstallations();
        if (conflictCheck.status !== 'ok') {
            healthChecks.push(conflictCheck);
        }

        // Settings Health
        healthChecks.push(await this.checkSettingsHealth());

        return {
            installation,
            healthChecks,
            versionLock: this.getVersionLockInfo(),
            environmentVariables: this.checkEnvironmentVariables(),
            permissions: {
                unreachableRules: this.getUnreachablePermissionRules()
            },
            contextUsage: await this.getContextUsage()
        };
    }

    private static async getInstallationInfo(): Promise<InstallationInfo> {
        const version = "2.1.19";
        const type = await this.detectInstallationType();
        const invokedBinary = process.argv[1] || 'unknown';
        const installMethod = EnvService.get("CLAUDE_CODE_INSTALL_METHOD") || "not set";
        const rgStatus = this.checkRipgrep();

        // Check for multiple installations (simple version)
        const multiple = await this.checkMultipleInstallations();

        return {
            type: type as any,
            version,
            path: process.execPath,
            invokedBinary,
            installMethod,
            autoUpdates: this.getAutoUpdatesStatus(),
            ripgrep: rgStatus
        };
    }

    private static getAutoUpdatesStatus(): 'enabled' | 'disabled' | 'error' {
        const disableReason = EnvService.get("CLAUDE_CODE_DISABLE_AUTO_UPDATE");
        if (disableReason) return 'disabled';
        return 'enabled';
    }

    private static async detectInstallationType(): Promise<string> {
        const invoked = process.argv[1] || "";

        // Match gold reference detection patterns
        if (invoked.includes('ts-node') || process.env.NODE_ENV === 'development') return 'development';
        if (invoked.includes('.local/bin/claude')) return 'native';
        if (invoked.match(/node_modules[\\\/]@anthropic-ai[\\\/]claude-code/)) {
            // Check if it's a global install or local
            try {
                const npmPrefix = execSync('npm config get prefix', { stdio: 'pipe' }).toString().trim();
                if (invoked.startsWith(npmPrefix)) return 'npm-global';
                return 'npm-local';
            } catch {
                return 'npm-global';
            }
        }

        const installMethod = EnvService.get("CLAUDE_CODE_INSTALL_METHOD");
        if (installMethod === 'homebrew') return 'package-manager';

        return 'unknown';
    }

    private static checkRipgrep(): { ok: boolean; mode: 'system' | 'builtin' | 'missing'; details?: string } {
        // Match gold reference ripgrep detection
        const useBuiltin = EnvService.get("CLAUDE_CODE_USE_BUILTIN_RIPGREP");

        if (useBuiltin === "true" || useBuiltin === "1") {
            return { ok: true, mode: 'builtin', details: 'Using bundled ripgrep' };
        }

        try {
            const output = execSync(`rg --version`, { stdio: 'pipe' }).toString().trim();
            return { ok: true, mode: 'system', details: output.split('\n')[0] };
        } catch (e) {
            return { ok: false, mode: 'missing', details: 'ripgrep (rg) not found in PATH.' };
        }
    }

    private static checkBinary(cmd: string, args: string, displayName: string): HealthCheckResult {
        try {
            const output = execSync(`${cmd} ${args}`, { stdio: 'pipe' }).toString().trim();
            return {
                name: displayName,
                status: 'ok',
                message: output.split('\n')[0]
            };
        } catch (error) {
            return {
                name: displayName,
                status: displayName === 'tmux' ? 'warn' : 'error',
                message: `${displayName} not found. Some features may be limited.`
            };
        }
    }

    private static async checkConnectivity(hostname: string): Promise<HealthCheckResult> {
        return new Promise((resolveResolve) => {
            const startTime = Date.now();
            const req = https.get(`https://${hostname}`, (res) => {
                const duration = Date.now() - startTime;
                resolveResolve({
                    name: `Network: ${hostname}`,
                    status: 'ok',
                    message: `Connected in ${duration}ms (Status: ${res.statusCode})`
                });
                res.resume();
            });

            req.on('error', (err) => {
                resolveResolve({
                    name: `Network: ${hostname}`,
                    status: 'error',
                    message: `Connection failed: ${err.message}`
                });
            });

            req.setTimeout(5000, () => {
                req.destroy();
                resolveResolve({
                    name: `Network: ${hostname}`,
                    status: 'error',
                    message: 'Connection timed out'
                });
            });
        });
    }

    private static checkPathHealth(): HealthCheckResult {
        let inPath = false;
        try {
            const which = process.platform === 'win32' ? 'where' : 'which';
            execSync(`${which} claude`, { stdio: 'pipe' });
            inPath = true;
        } catch (e) { }

        if (inPath) {
            return {
                name: "Binary Path",
                status: 'ok',
                message: "'claude' is correctly installed in your PATH."
            };
        } else {
            return {
                name: "Binary Path",
                status: 'warn',
                message: "'claude' binary not found in PATH. You may need to run 'npm install -g @anthropic-ai/claude-code'."
            };
        }
    }

    private static async checkMultipleInstallations(): Promise<HealthCheckResult> {
        const installations: string[] = [];
        const home = process.env.HOME || process.env.USERPROFILE || "";
        const nativePath = resolve(home, ".local/bin/claude");
        if (existsSync(nativePath)) installations.push(`Native: ${nativePath}`);

        try {
            const npmPrefix = execSync('npm config get prefix', { stdio: 'pipe' }).toString().trim();
            const npmPath = process.platform === 'win32'
                ? resolve(npmPrefix, "claude.cmd")
                : resolve(npmPrefix, "bin", "claude");
            if (existsSync(npmPath)) installations.push(`NPM Global: ${npmPath}`);
        } catch (e) { }

        if (installations.length > 1) {
            return {
                name: 'Multiple Installations',
                status: 'warn',
                message: `Found ${installations.length} installations. This might cause version conflicts.`,
                details: installations.join('\n')
            };
        }

        return {
            name: 'Multiple Installations',
            status: 'ok',
            message: 'No conflicting installations found.'
        };
    }

    private static async checkSettingsHealth(): Promise<HealthCheckResult> {
        const home = process.env.HOME || process.env.USERPROFILE || "";
        const settingsFiles = [
            resolve(home, ".claude", "settings.json"),
            resolve(process.cwd(), ".claude", "settings.json"),
            resolve(process.cwd(), ".claude", "settings.local.json")
        ];

        const errors: string[] = [];
        for (const file of settingsFiles) {
            if (existsSync(file)) {
                try {
                    const content = execSync(process.platform === 'win32' ? `type "${file}"` : `cat "${file}"`, { stdio: 'pipe' }).toString();
                    JSON.parse(content);
                } catch (e: any) {
                    errors.push(`${file}: ${e.message}`);
                }
            }
        }

        if (errors.length > 0) {
            return {
                name: 'Settings Health',
                status: 'error',
                message: `Found ${errors.length} invalid settings file(s).`,
                details: errors.join('\n')
            };
        }

        return {
            name: 'Settings Health',
            status: 'ok',
            message: 'All settings files are valid.'
        };
    }

    private static getVersionLockInfo(): { locked: boolean; version?: string; source?: string } {
        const lockedVersion = EnvService.get("CLAUDE_CODE_VERSION");
        if (lockedVersion) {
            return {
                locked: true,
                version: lockedVersion,
                source: 'Environment (CLAUDE_CODE_VERSION)'
            };
        }
        return {
            locked: false
        };
    }

    private static checkEnvironmentVariables() {
        const varsToCheck = [
            'BASH_MAX_OUTPUT_LENGTH',
            'TASK_MAX_OUTPUT_LENGTH',
            'CLAUDE_CODE_MAX_OUTPUT_TOKENS'
        ];

        return varsToCheck.map(name => {
            const value = EnvService.get(name);
            if (value === undefined) {
                return { name, value: 'not set', status: 'not-set' as const };
            }

            const numValue = parseInt(String(value), 10);
            if (isNaN(numValue)) {
                return { name, value: String(value), status: 'invalid' as const, message: 'Must be a number' };
            }

            if (name === 'BASH_MAX_OUTPUT_LENGTH' && numValue > 150000) {
                return { name, value: String(value), status: 'capped' as const, message: 'Capped at 150000' };
            }

            return { name, value: String(value), status: 'valid' as const };
        });
    }

    private static getUnreachablePermissionRules(): string[] {
        const unreachable: string[] = [];
        const settings = getToolSettings('userSettings');
        const allowed = new Set(settings.permissions?.allow || []);
        const denied = new Set(settings.permissions?.deny || []);

        for (const rule of allowed) {
            if (denied.has(rule)) {
                unreachable.push(`Conflict: '${rule}' is in both allow and deny lists.`);
            }
        }

        return unreachable;
    }

    private static async getContextUsage() {
        let claudeMdSize = 0;
        const claudeMdPath = resolve(process.cwd(), 'CLAUDE.md');
        if (existsSync(claudeMdPath)) {
            try {
                claudeMdSize = statSync(claudeMdPath).size;
            } catch (e) { }
        }

        const agents = listAgents();
        const agentCount = agents.length;

        // Placeholder for MCP token count - in real implementation this would query McpClientManager
        const mcpTokenCount = 0;

        return {
            claudeMdSize,
            agentCount,
            mcpTokenCount
        };
    }
}
