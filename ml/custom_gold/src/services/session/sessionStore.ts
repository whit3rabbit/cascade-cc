import { getGlobalState, setGlobalState, resetGlobalState } from "./globalState.js";
import type { SessionState } from "./globalState.js";
export { getGlobalState, setGlobalState, resetGlobalState };
export type { SessionState };
import { randomUUID } from "node:crypto";

export function getSessionId(): string {
    let state = getGlobalState();
    if (!state.sessionId) {
        state.sessionId = randomUUID();
    }
    return state.sessionId;
}

export function setSessionId(id: string) {
    const state = getGlobalState();
    state.sessionId = id;
    if (process.env.CLAUDE_CODE_SESSION_ID !== undefined) {
        process.env.CLAUDE_CODE_SESSION_ID = id;
    }
}

export function generateSessionId(): string {
    const id = randomUUID();
    setSessionId(id);
    return id;
}

export function getOriginalCwd(): string {
    return getGlobalState().originalCwd;
}

export function setOriginalCwd(cwd: string) {
    getGlobalState().originalCwd = cwd;
}

export function getCwd(): string {
    return getGlobalState().cwd;
}

export function setCwd(cwd: string) {
    getGlobalState().cwd = cwd;
}

export function addApiDuration(duration: number, durationWithoutRetries: number) {
    const state = getGlobalState();
    state.totalAPIDuration += duration;
    state.totalAPIDurationWithoutRetries += durationWithoutRetries;
}

export function addModelUsage(cost: number, usage: any, model: string) {
    const state = getGlobalState();
    state.totalCostUSD += cost;

    // Default struct
    const modelStats = state.modelUsage[model] ?? {
        inputTokens: 0,
        outputTokens: 0,
        cacheReadInputTokens: 0,
        cacheCreationInputTokens: 0,
        webSearchRequests: 0,
        costUSD: 0,
        contextWindow: 0 // Placeholder logic for now
    };

    modelStats.inputTokens += usage.input_tokens || 0;
    modelStats.outputTokens += usage.output_tokens || 0;
    modelStats.cacheReadInputTokens += usage.cache_read_input_tokens ?? 0;
    modelStats.cacheCreationInputTokens += usage.cache_creation_input_tokens ?? 0;
    modelStats.webSearchRequests += usage.server_tool_use?.web_search_requests ?? 0;
    modelStats.costUSD += cost;

    state.modelUsage[model] = modelStats;
}

export function getTotalCost(): number {
    return getGlobalState().totalCostUSD;
}

export function getTotalApiDuration(): number {
    return getGlobalState().totalAPIDuration;
}

export function getSessionDuration(): number {
    return Date.now() - getGlobalState().startTime;
}

export function addLinesChanged(added: number, removed: number) {
    const state = getGlobalState();
    state.totalLinesAdded += added;
    state.totalLinesRemoved += removed;
}

export function getTotalLinesAdded(): number {
    return getGlobalState().totalLinesAdded;
}

export function getTotalLinesRemoved(): number {
    return getGlobalState().totalLinesRemoved;
}

export function updateLastInteractionTime() {
    getGlobalState().lastInteractionTime = Date.now();
}

export function getLastInteractionTime(): number {
    return getGlobalState().lastInteractionTime;
}

export function getClientType(): string {
    return getGlobalState().clientType;
}

export function setClientType(type: string) {
    getGlobalState().clientType = type;
}

export function isInteractive(): boolean {
    return getGlobalState().isInteractive;
}

export function setInteractive(value: boolean) {
    getGlobalState().isInteractive = value;
}

export function getAuthToken(): string | undefined {
    return getGlobalState().oauthTokenFromFd;
}

export function setAuthToken(token: string) {
    getGlobalState().oauthTokenFromFd = token;
}

export function getModelUsage(): Record<string, any> {
    return getGlobalState().modelUsage;
}

export function resetSessionMetrics() {
    const state = getGlobalState();
    // Keep identifiers, reset counters
    resetGlobalState();
}

export function setTelemetryMeters(meter: any, factory: any) {
    const state = getGlobalState();
    state.meter = meter;

    state.sessionCounter = factory("claude_code.session.count", { description: "Count of CLI sessions started" });
    state.locCounter = factory("claude_code.lines_of_code.count", { description: "Count of lines of code modified" });
    state.prCounter = factory("claude_code.pull_request.count", { description: "Number of pull requests created" });
    state.commitCounter = factory("claude_code.commit.count", { description: "Number of git commits created" });
    state.costCounter = factory("claude_code.cost.usage", { description: "Cost of the Claude Code session", unit: "USD" });
    state.tokenCounter = factory("claude_code.token.usage", { description: "Number of tokens used", unit: "tokens" });
    state.codeEditToolDecisionCounter = factory("claude_code.code_edit_tool.decision", { description: "Count of code editing tool permission decisions" });
    state.activeTimeCounter = factory("claude_code.active_time.total", { description: "Total active time in seconds", unit: "s" });
}

export function getTelemetryMeters() {
    const state = getGlobalState();
    return {
        sessionCounter: state.sessionCounter,
        locCounter: state.locCounter,
        prCounter: state.prCounter,
        commitCounter: state.commitCounter,
        costCounter: state.costCounter,
        tokenCounter: state.tokenCounter,
        decisionCounter: state.codeEditToolDecisionCounter,
        activeTimeCounter: state.activeTimeCounter
    };
}
