/**
 * Constants and validators for various tool and output limits.
 * Deobfuscated from chunk_2.ts.
 */

export interface LimitValidationResult {
    effective: number;
    status: "valid" | "invalid" | "capped";
    message?: string;
}

/**
 * Configuration for max bash output length.
 * Deobfuscated from PhA in chunk_2.ts.
 */
export const bashMaxOutputLength = {
    name: "BASH_MAX_OUTPUT_LENGTH",
    default: 30000,
    validate: (value?: string): LimitValidationResult => {
        if (!value) return { effective: 30000, status: "valid" };
        const parsed = parseInt(value, 10);
        if (isNaN(parsed) || parsed <= 0) {
            return {
                effective: 30000,
                status: "invalid",
                message: `Invalid value "${value}" (using default: 30000)`
            };
        }
        if (parsed > 150000) {
            return {
                effective: 150000,
                status: "capped",
                message: `Capped from ${parsed} to 150000`
            };
        }
        return { effective: parsed, status: "valid" };
    }
};

/**
 * Configuration for max output tokens.
 * Deobfuscated from ShA in chunk_2.ts.
 */
export const claudeCodeMaxOutputTokens = {
    name: "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
    default: 32000,
    validate: (value?: string): LimitValidationResult => {
        if (!value) return { effective: 32000, status: "valid" };
        const parsed = parseInt(value, 10);
        if (isNaN(parsed) || parsed <= 0) {
            return {
                effective: 32000,
                status: "invalid",
                message: `Invalid value "${value}" (using default: 32000)`
            };
        }
        if (parsed > 64000) {
            return {
                effective: 64000,
                status: "capped",
                message: `Capped from ${parsed} to 64000`
            };
        }
        return { effective: parsed, status: "valid" };
    }
};

/**
 * Determines memory limit based on context.
 * Deobfuscated from NO in chunk_2.ts.
 */
export function getMemoryLimit(tags: string[]): number {
    if (tags.includes("[1m]")) return 1000000;
    return 200000;
}

export const DEFAULT_OUTPUT_LIMIT = 20000;
