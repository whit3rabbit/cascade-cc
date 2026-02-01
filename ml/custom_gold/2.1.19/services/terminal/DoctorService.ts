/**
 * File: src/services/terminal/DoctorService.ts
 * Role: Performs health checks on dependencies, network, and environment.
 */

import { execSync } from 'node:child_process';
import { EnvService } from '../config/EnvService.js';
import https from 'node:https';
import { resolve } from 'node:path';
import { existsSync } from 'node:fs';

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
    autoUpdates: string;
}

export class DoctorService {
    static async runChecks(): Promise<HealthCheckResult[]> {
        const results: HealthCheckResult[] = [];

        // 1. Installation Info
        const installInfo = await this.getInstallationInfo();
        results.push({
            name: 'Installation',
            status: 'ok',
            message: `${installInfo.type} (${installInfo.version})`,
            details: `Path: ${installInfo.path}\nInvoked: ${installInfo.invokedBinary}\nInstall Method: ${installInfo.installMethod}`
        });

        // 2. Dependency Checks
        results.push(this.checkBinary('git', '--version', 'Git'));

        // Ripgrep check
        const rgStatus = this.checkRipgrep();
        results.push({
            name: 'Search (ripgrep)',
            status: rgStatus.ok ? 'ok' : 'error',
            message: rgStatus.ok ? `OK (${rgStatus.mode})` : 'Not working',
            details: rgStatus.details
        });

        results.push(this.checkBinary('tmux', '-V', 'tmux'));

        // 3. Network Checks
        results.push(await this.checkConnectivity('api.anthropic.com'));
        results.push(await this.checkConnectivity('statsig.anthropic.com'));

        // 4. Path & Environment
        results.push(this.checkPathHealth());

        // 5. Conflicts
        results.push(await this.checkMultipleInstallations());

        // 6. Settings
        results.push(await this.checkSettingsHealth());

        return results;
    }

    private static async getInstallationInfo(): Promise<InstallationInfo> {
        const version = "2.1.19";
        const type = await this.detectInstallationType();
        const invokedBinary = process.argv[1] || 'unknown';
        const installMethod = EnvService.get("CLAUDE_CODE_INSTALL_METHOD") || "not set";

        return {
            type: type as any,
            version,
            path: process.execPath,
            invokedBinary,
            installMethod,
            autoUpdates: "enabled" // Mocking for now as we don't have the full autoupdater deobfuscated
        };
    }

    private static async detectInstallationType(): Promise<string> {
        const invoked = process.argv[1] || "";
        if (invoked.includes('ts-node') || invoked.includes('node_modules/.bin/claude')) return 'development';
        if (invoked.includes('.local/bin/claude')) return 'native';
        if (invoked.match(/node_modules[\\\/]@anthropic-ai[\\\/]claude-code/)) return 'npm-global';
        return 'unknown';
    }

    private static async checkMultipleInstallations(): Promise<HealthCheckResult> {
        const installations: string[] = [];
        const home = process.env.HOME || process.env.USERPROFILE || "";

        // Check native path
        const nativePath = resolve(home, ".local/bin/claude");
        if (existsSync(nativePath)) installations.push(`Native: ${nativePath}`);

        // Check npm global (rough check)
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

    private static checkRipgrep(): { ok: boolean; mode: string; details?: string } {
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
                status: displayName === 'tmux' ? 'warn' : 'error', // tmux is often optional but nice to have
                message: `${displayName} not found. Some features may be limited.`
            };
        }
    }

    private static async checkConnectivity(hostname: string): Promise<HealthCheckResult> {
        return new Promise((resovleResolve) => {
            const startTime = Date.now();
            const req = https.get(`https://${hostname}`, (res) => {
                const duration = Date.now() - startTime;
                resovleResolve({
                    name: `Network: ${hostname}`,
                    status: 'ok',
                    message: `Connected in ${duration}ms (Status: ${res.statusCode})`
                });
                res.resume();
            });

            req.on('error', (err) => {
                resovleResolve({
                    name: `Network: ${hostname}`,
                    status: 'error',
                    message: `Connection failed: ${err.message}`
                });
            });

            req.setTimeout(5000, () => {
                req.destroy();
                resovleResolve({
                    name: `Network: ${hostname}`,
                    status: 'error',
                    message: 'Connection timed out'
                });
            });
        });
    }

    private static checkPathHealth(): HealthCheckResult {
        // Check if 'claude' is in PATH
        let inPath = false;
        try {
            const which = process.platform === 'win32' ? 'where' : 'which';
            execSync(`${which} claude`, { stdio: 'pipe' });
            inPath = true;
        } catch (e) {
            // Not in path
        }

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
}
