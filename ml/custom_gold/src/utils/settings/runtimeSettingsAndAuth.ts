import { join } from "path";
import { homedir } from "os";

/**
 * Returns the directory where Claude configuration is stored.
 * Defaults to ~/.claude unless CLAUDE_CONFIG_DIR is set.
 */
export function getConfigDir(): string {
    return process.env.CLAUDE_CONFIG_DIR ?? join(homedir(), ".claude");
}

/**
 * Converts a string or boolean value to a boolean.
 * Recognizes "1", "true", "yes", "on" as true (case-insensitive).
 */
export function toBoolean(value: any): boolean {
    if (!value) return false;
    if (typeof value === "boolean") return value;
    const trimmed = value.toLowerCase().trim();
    return ["1", "true", "yes", "on"].includes(trimmed);
}

/**
 * Converts a string or boolean value to a boolean (negated/falsey check).
 * Recognizes "0", "false", "no", "off" as false (case-insensitive).
 */
export function fromBoolean(value: any): boolean {
    if (value === undefined) return false;
    if (typeof value === "boolean") return !value;
    if (!value) return false;
    const trimmed = value.toLowerCase().trim();
    return ["0", "false", "no", "off"].includes(trimmed);
}

/**
 * Parses an array of environment variable strings (e.g., ["KEY=VALUE"]) into an object.
 * Throws an error if the format is invalid.
 */
export function parseEnvVars(vars?: string[]): Record<string, string> {
    const result: Record<string, string> = {};
    if (vars) {
        for (const v of vars) {
            const [key, ...rest] = v.split("=");
            if (!key || rest.length === 0) {
                throw new Error(`Invalid environment variable format: ${v}, environment variables should be added as: -e KEY1=value1 -e KEY2=value2`);
            }
            result[key] = rest.join("=");
        }
    }
    return result;
}

/**
 * Returns the AWS region from environment variables, defaulting to 'us-east-1'.
 */
export function getAwsRegion(): string {
    return process.env.AWS_REGION || process.env.AWS_DEFAULT_REGION || "us-east-1";
}

/**
 * Returns the Cloud ML region from environment variables, defaulting to 'us-east5'.
 */
export function getCloudMlRegion(): string {
    return process.env.CLOUD_ML_REGION || "us-east5";
}

/**
 * Checks if the project working directory should be maintained for bash.
 */
export function getBashMaintainProjectWorkingDir(): boolean {
    return toBoolean(process.env.CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR);
}

/**
 * Returns the appropriate Vertex region for a given model.
 */
export function getVertexRegionForModel(modelName?: string): string {
    if (modelName?.startsWith("claude-haiku-4-5")) return process.env.VERTEX_REGION_CLAUDE_HAIKU_4_5 || getCloudMlRegion();
    if (modelName?.startsWith("claude-3-5-haiku")) return process.env.VERTEX_REGION_CLAUDE_3_5_HAIKU || getCloudMlRegion();
    if (modelName?.startsWith("claude-3-5-sonnet")) return process.env.VERTEX_REGION_CLAUDE_3_5_SONNET || getCloudMlRegion();
    if (modelName?.startsWith("claude-3-7-sonnet")) return process.env.VERTEX_REGION_CLAUDE_3_7_SONNET || getCloudMlRegion();
    if (modelName?.startsWith("claude-opus-4-1")) return process.env.VERTEX_REGION_CLAUDE_4_1_OPUS || getCloudMlRegion();
    if (modelName?.startsWith("claude-opus-4")) return process.env.VERTEX_REGION_CLAUDE_4_0_OPUS || getCloudMlRegion();
    if (modelName?.startsWith("claude-sonnet-4-5")) return process.env.VERTEX_REGION_CLAUDE_4_5_SONNET || getCloudMlRegion();
    if (modelName?.startsWith("claude-sonnet-4")) return process.env.VERTEX_REGION_CLAUDE_4_0_SONNET || getCloudMlRegion();
    return getCloudMlRegion();
}
