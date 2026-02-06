/**
 * File: src/commands/doctor.ts
 * Role: Implementation of the /doctor command.
 */

import { CommandDefinition, createCommandHelper } from './helpers.js';
import { DoctorService, formatDoctorReport } from '../services/terminal/DoctorService.js';

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
            onDone(formatDoctorReport(info));
        }
    }
);
