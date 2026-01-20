import { z } from "zod";
import { logTelemetryEvent } from "../../services/telemetry/telemetryInit.js";

const maxOutputSchema = z.coerce.number().int().positive().max(2000000).default(200000);

function parsePositiveInt(envValue: string | undefined): number | null {
    if (!envValue) return null;
    const parsed = parseInt(envValue, 10);
    if (Number.isNaN(parsed) || parsed <= 0) return null;
    return parsed;
}

export function getBashTimeout(): number {
    const envValue = parsePositiveInt(process.env.BASH_DEFAULT_TIMEOUT_MS);
    if (envValue !== null) return envValue;
    return 120000;
}

export function getMaxBashTimeout(): number {
    const envValue = parsePositiveInt(process.env.BASH_MAX_TIMEOUT_MS);
    if (envValue !== null) return Math.max(envValue, getBashTimeout());
    return Math.max(600000, getBashTimeout());
}

export function getMaxBashOutputLength(): number {
    const parsed = maxOutputSchema.safeParse(process.env.BASH_MAX_OUTPUT_LENGTH);
    if (!parsed.success) {
        logTelemetryEvent("tengu_bash_max_output_length_invalid", {
            input: process.env.BASH_MAX_OUTPUT_LENGTH
        });
        return maxOutputSchema.parse(undefined);
    }
    return parsed.data;
}
