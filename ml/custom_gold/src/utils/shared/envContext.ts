import { getAgentContext } from "./sessionContext.js";
import { toBoolean } from "../settings/runtimeSettingsAndAuth.js";

// Placeholders for platform/terminal detection - these likely live in other chunks
const platformUtils = {
    platform: process.platform,
    arch: process.arch,
    nodeVersion: process.version,
    getPackageManagers: async () => ["npm"],
    getRuntimes: async () => ["node"],
    isRunningWithBun: () => false,
    isConductor: () => false,
    detectDeploymentEnvironment: () => "production"
};

const terminalUtils = {
    terminal: process.env.TERM || "unknown"
};

/**
 * Gathers detailed environment context.
 */
export async function getEnvironmentContext() {
    const [packageManagers, runtimes] = await Promise.all([
        platformUtils.getPackageManagers(),
        platformUtils.getRuntimes()
    ]);

    return {
        platform: platformUtils.platform,
        arch: platformUtils.arch,
        nodeVersion: platformUtils.nodeVersion,
        terminal: terminalUtils.terminal,
        packageManagers: packageManagers.join(","),
        runtimes: runtimes.join(","),
        isRunningWithBun: platformUtils.isRunningWithBun(),
        isCi: toBoolean(process.env.CI),
        isClaubbit: process.env.CLAUBBIT === "true",
        isClaudeCodeRemote: process.env.CLAUDE_CODE_REMOTE === "true",
        isConductor: platformUtils.isConductor(),
        remoteEnvironmentType: process.env.CLAUDE_CODE_REMOTE_ENVIRONMENT_TYPE,
        claudeCodeContainerId: process.env.CLAUDE_CODE_CONTAINER_ID,
        claudeCodeRemoteSessionId: process.env.CLAUDE_CODE_REMOTE_SESSION_ID,
        tags: process.env.CLAUDE_CODE_TAGS,
        isGithubAction: process.env.GITHUB_ACTIONS === "true",
        isClaudeCodeAction: process.env.CLAUDE_CODE_ACTION === "1" || process.env.CLAUDE_CODE_ACTION === "true",
        // isClaudeAiAuth: isOauthEnabled(), // We'll need a better way to check this
        version: "2.0.76",
        buildTime: "2025-12-22T23:56:12Z",
        deploymentEnvironment: platformUtils.detectDeploymentEnvironment(),
        ...(process.env.GITHUB_ACTIONS === "true" && {
            githubEventName: process.env.GITHUB_EVENT_NAME,
            githubActionsRunnerEnvironment: process.env.RUNNER_ENVIRONMENT,
            githubActionsRunnerOs: process.env.RUNNER_OS,
            githubActionRef: process.env.GITHUB_ACTION_PATH?.includes("claude-code-action/")
                ? process.env.GITHUB_ACTION_PATH.split("claude-code-action/")[1]
                : undefined
        })
    };
}

/**
 * Returns core context for telemetry events.
 */
export async function getTelemetryContext(options: { model?: string } = {}) {
    const envContext = await getEnvironmentContext();
    const agentContext = getAgentContext();

    return {
        model: options.model || process.env.ANTHROPIC_MODEL || "unknown",
        sessionId: process.env.CLAUDE_CODE_SESSION_ID || "unknown",
        userType: "external",
        envContext,
        isInteractive: String(toBoolean(process.env.CLAUDE_CODE_INTERACTIVE)),
        clientType: process.env.CLAUDE_CODE_CLIENT_TYPE || "cli",
        sweBenchRunId: process.env.SWE_BENCH_RUN_ID || "",
        sweBenchInstanceId: process.env.SWE_BENCH_INSTANCE_ID || "",
        sweBenchTaskId: process.env.SWE_BENCH_TASK_ID || "",
        ...agentContext
    };
}

/**
 * Encodes telemetry context/metadata for logging.
 */
export function encodeTelemetryMetadata(context: any, metadata: Record<string, any> = {}): Record<string, string> {
    const result: Record<string, string> = {};
    for (const [key, value] of Object.entries(metadata)) {
        if (value !== undefined) result[key] = String(value);
    }
    for (const [key, value] of Object.entries(context)) {
        if (value === undefined) continue;
        if (key === "envContext") result.env = JSON.stringify(value);
        else if (key === "processMetrics") result.process = JSON.stringify(value);
        else result[key] = String(value);
    }
    return result;
}

/**
 * Prepares an event for Statsig logging.
 */
export function prepareTelemetryEvent(context: any, metadata: Record<string, any> = {}) {
    const { envContext, processMetrics, ...rest } = context;
    return {
        ...metadata,
        ...rest,
        env: envContext,
        ...(processMetrics && { process: processMetrics }),
        surface: "claude-code"
    };
}
