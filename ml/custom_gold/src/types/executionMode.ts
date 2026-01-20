/**
 * Execution modes for the CLI.
 * Deobfuscated from CT and nA1 in chunk_224.ts.
 */

export type ExecutionMode =
    | "default"
    | "plan"
    | "delegate"
    | "acceptEdits"
    | "bypassPermissions"
    | "dontAsk";

/**
 * Validates and normalizes an execution mode string.
 */
export function normalizeExecutionMode(mode: string): ExecutionMode {
    switch (mode) {
        case "plan": return "plan";
        case "delegate": return "delegate";
        case "acceptEdits": return "acceptEdits";
        case "bypassPermissions": return "bypassPermissions";
        case "dontAsk": return "dontAsk";
        case "default": return "default";
        default: return "default";
    }
}

/**
 * Returns a human-readable label for a mode.
 */
export function getExecutionModeLabel(mode: ExecutionMode): string {
    switch (mode) {
        case "plan": return "Plan Mode";
        case "delegate": return "Delegate Mode";
        case "acceptEdits": return "Accept edits";
        case "bypassPermissions": return "Bypass Permissions";
        case "dontAsk": return "Don't Ask";
        case "default": return "Default";
    }
}

/**
 * Returns an icon representing the mode.
 */
export function getExecutionModeIcon(mode: ExecutionMode): string {
    switch (mode) {
        case "plan": return "⏸";
        case "delegate": return "⇢";
        case "acceptEdits":
        case "bypassPermissions":
        case "dontAsk": return "⏵⏵";
        default: return "";
    }
}

/**
 * Returns the color/style key for the mode.
 */
export function getExecutionModeStyle(mode: ExecutionMode): string {
    switch (mode) {
        case "plan": return "planMode";
        case "delegate": return "delegateMode";
        case "acceptEdits": return "autoAccept";
        case "bypassPermissions":
        case "dontAsk": return "error";
        default: return "text";
    }
}
