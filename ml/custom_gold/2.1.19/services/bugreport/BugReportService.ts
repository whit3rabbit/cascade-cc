/**
 * File: src/services/bugreport/BugReportService.ts
 * Role: Collect logs and system info for bug reporting.
 */

import { zip, Zippable, strToU8 } from 'fflate';
import { promises as fs, createWriteStream } from 'fs';
import { join } from 'path';
import { homedir, platform, release, arch, cpus, totalmem } from 'os';
import { EnvService } from '../config/EnvService.js';

export class BugReportService {
    static async generateReport(outputDir: string = process.cwd()): Promise<string> {
        try {
            const systemInfo = this.getSystemInfo();
            const logs = await this.getRecentLogs();

            const zipData: Zippable = {
                'system-info.json': strToU8(JSON.stringify(systemInfo, null, 2)),
                'logs.txt': strToU8(logs),
            };

            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const filename = `claude-bugreport-${timestamp}.zip`;
            const outputPath = join(outputDir, filename);

            return new Promise((resolve, reject) => {
                zip(zipData, { level: 9 }, (err, data) => {
                    if (err) return reject(err);
                    fs.writeFile(outputPath, data)
                        .then(() => resolve(outputPath))
                        .catch(reject);
                });
            });

        } catch (error) {
            throw new Error(`Failed to generate bug report: ${error}`);
        }
    }

    private static getSystemInfo() {
        return {
            platform: platform(),
            release: release(),
            arch: arch(),
            cpus: cpus().length,
            totalMem: totalmem(),
            nodeVersion: process.version,
            env: {
                TERM: EnvService.get("TERM"),
                SHELL: EnvService.get("SHELL")
            }
        };
    }

    private static async getRecentLogs(): Promise<string> {
        // Attempt to read recent logs from standard location
        // Assuming ./claude/logs or similar
        // For now, return a placeholder or in-memory logs
        return "Log collection implementation pending.\n[INFO] Bug report generation started.";
    }
}
