/**
 * File: src/constants/keys.ts
 * Role: Unique keys for context, storage, and telemetry.
 */

/**
 * OpenTelemetry and internal context keys.
 */
export const CONTEXT_KEYS = {
    SUPPRESS_TRACING: Symbol.for("OpenTelemetry SDK Context Key SUPPRESS_TRACING"),
    SMITHY_CONTEXT: "__smithy_context"
} as const;

/**
 * Storage keys for local settings and persistence.
 */
export const STORAGE_KEYS = {
    ONBOARDING_COMPLETED: "hasCompletedOnboarding",
    LAST_VERSION: "lastOnboardingVersion",
    GITHUB_REPO_PATHS: "githubRepoPaths"
} as const;

/**
 * Analytics and tracking event keys.
 */
export const ANALYTICS_KEYS = {
    MCP_CLI_STATUS: "tengu_mcp_cli_status",
    ACCEPT_FEEDBACK_MODE: "tengu_accept_feedback_mode_entered"
} as const;

/**
 * Identity and endpoint keys.
 */
export const IDENTITY_ENDPOINT = "IDENTITY_ENDPOINT";
export const IMDS_ENDPOINT = "IMDS_ENDPOINT";
