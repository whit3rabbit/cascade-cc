/**
 * File: src/utils/shared/loggingUtils.ts
 * Role: Standardized severity levels and logging helpers.
 */

export type SeverityLevel = "fatal" | "error" | "warning" | "log" | "info" | "debug";

const VALID_SEVERITY_LEVELS: SeverityLevel[] = ["fatal", "error", "warning", "log", "info", "debug"];

/**
 * Normalizes a severity string to a valid SeverityLevel.
 */
export function severityLevelFromString(severity: string | undefined): SeverityLevel {
    if (!severity) return "log";
    const s = severity.toLowerCase();
    if (s === "warn") return "warning";
    if ((VALID_SEVERITY_LEVELS as string[]).includes(s)) return s as SeverityLevel;
    return "log";
}

/**
 * Backward compatible alias.
 */
export const severityFromString = severityLevelFromString;
