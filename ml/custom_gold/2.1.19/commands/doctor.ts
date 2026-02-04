/**
 * File: src/commands/doctor.ts
 * Role: Implementation of the /doctor command.
 */

import { CommandDefinition, createCommandHelper } from './helpers.js';
import { DoctorService } from '../services/terminal/DoctorService.js';

/**
 * Command definition for /doctor, diagnosing installation health.
 * Aligned with 2.1.19 gold reference format.
 */
export const doctorCommandDefinition: CommandDefinition = createCommandHelper(
    "doctor",
    "Diagnose and verify your Claude Code installation and settings",
    {
        type: "local",
        isEnabled: () => !process.env.DISABLE_DOCTOR_COMMAND,
        async call(onDone: (result: string) => void) {
            const info = await DoctorService.getDiagnosticInfo();

            let report = `## Diagnostics\n`;
            report += `└ Currently running: ${info.installationType} (${info.version})\n`;
            if (info.packageManager) report += `└ Package manager: ${info.packageManager}\n`;
            report += `└ Path: ${info.installationPath}\n`;
            report += `└ Invoked: ${info.invokedBinary}\n`;
            report += `└ Search (ripgrep): ${info.ripgrepStatus.workingDirectory ? "OK" : "Not working"} (${info.ripgrepStatus.mode})\n\n`;

            if (info.warnings.length > 0) {
                report += `### ⚠️ Warnings\n`;
                for (const w of info.warnings) {
                    report += `- **${w.issue}**\n  *Fix:* ${w.fix}\n`;
                }
                report += `\n`;
            }

            report += `### Environment Variables\n`;
            for (const ev of info.envVars) {
                const statusIcon = ev.status === 'ok' ? '✅' : (ev.status === 'capped' ? '⚠️' : '❌');
                report += `└ ${statusIcon} ${ev.name}: ${ev.message}\n`;
            }

            onDone(report);
        }
    }
);
