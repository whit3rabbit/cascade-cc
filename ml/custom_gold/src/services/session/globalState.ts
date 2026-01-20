export interface ModelUsageStats {
    inputTokens: number;
    outputTokens: number;
    cacheReadInputTokens: number;
    cacheCreationInputTokens: number;
    webSearchRequests: number;
    costUSD: number;
    contextWindow: number;
}

export interface SessionState {
    originalCwd: string;
    totalCostUSD: number;
    totalAPIDuration: number;
    totalAPIDurationWithoutRetries: number;
    totalToolDuration: number;
    startTime: number;
    lastInteractionTime: number;
    totalLinesAdded: number;
    totalLinesRemoved: number;
    hasUnknownModelCost: boolean;
    cwd: string;
    modelUsage: Record<string, ModelUsageStats>;
    mainLoopModelOverride?: string;
    initialMainLoopModel: string | null;
    modelStrings: any;
    isInteractive: boolean;
    clientType: string;
    sessionIngressToken?: string;
    oauthTokenFromFd?: string;
    apiKeyFromFd?: string;
    flagSettingsPath?: string;
    allowedSettingSources: string[];
    meter: any;
    sessionCounter: any;
    locCounter: any;
    prCounter: any;
    commitCounter: any;
    costCounter: any;
    tokenCounter: any;
    codeEditToolDecisionCounter: any;
    activeTimeCounter: any;
    sessionId: string;
    loggerProvider: any;
    eventLogger: any;
    meterProvider: any;
    tracerProvider: any;
    agentColorMap: Map<string, string>;
    agentColorIndex: number;
    envVarValidators: any[];
    lastAPIRequest: any;
    inMemoryErrorLog: any[];
    inlinePlugins: any[];
    sessionBypassPermissionsMode: boolean;
    sessionPersistenceDisabled: boolean;
    hasExitedPlanMode: boolean;
    needsPlanModeExitAttachment: boolean;
    hasExitedDelegateMode: boolean;
    needsDelegateModeExitAttachment: boolean;
    lspRecommendationShownThisSession: boolean;
    initJsonSchema: any;
    registeredHooks: any;
    planSlugCache: Map<string, string>;
    teleportedSessionInfo: any;
    invokedSkills: Map<string, any>;
    toolPermissionContext: any;
    sessionHooks: any;

}

export const initialSessionState: SessionState = {
    originalCwd: "",
    totalCostUSD: 0,
    totalAPIDuration: 0,
    totalAPIDurationWithoutRetries: 0,
    totalToolDuration: 0,
    startTime: Date.now(),
    lastInteractionTime: Date.now(),
    totalLinesAdded: 0,
    totalLinesRemoved: 0,
    hasUnknownModelCost: false,
    cwd: "",
    modelUsage: {},
    mainLoopModelOverride: undefined,
    initialMainLoopModel: null,
    modelStrings: null,
    isInteractive: false,
    clientType: "cli",
    sessionIngressToken: undefined,
    oauthTokenFromFd: undefined,
    apiKeyFromFd: undefined,
    flagSettingsPath: undefined,
    allowedSettingSources: ["userSettings", "projectSettings", "localSettings", "flagSettings", "policySettings"],
    meter: null,
    sessionCounter: null,
    locCounter: null,
    prCounter: null,
    commitCounter: null,
    costCounter: null,
    tokenCounter: null,
    codeEditToolDecisionCounter: null,
    activeTimeCounter: null,
    sessionId: "",
    loggerProvider: null,
    eventLogger: null,
    meterProvider: null,
    tracerProvider: null,
    agentColorMap: new Map(),
    agentColorIndex: 0,
    envVarValidators: [],
    lastAPIRequest: null,
    inMemoryErrorLog: [],
    inlinePlugins: [],
    sessionBypassPermissionsMode: false,
    sessionPersistenceDisabled: false,
    hasExitedPlanMode: false,
    needsPlanModeExitAttachment: false,
    hasExitedDelegateMode: false,
    needsDelegateModeExitAttachment: false,
    lspRecommendationShownThisSession: false,
    initJsonSchema: null,
    registeredHooks: null,
    planSlugCache: new Map(),
    teleportedSessionInfo: null,
    invokedSkills: new Map(),
    toolPermissionContext: {},
    sessionHooks: {}
};

let globalState: SessionState = { ...initialSessionState };

export const getGlobalState = () => globalState;
export const setGlobalState = (newState: SessionState) => { globalState = newState; };
export const resetGlobalState = () => { globalState = { ...initialSessionState, sessionId: globalState.sessionId, cwd: globalState.cwd, originalCwd: globalState.originalCwd }; };

export const setHasExitedPlanMode = (value: boolean) => {
    globalState.hasExitedPlanMode = value;
};

export const setNeedsPlanModeExitAttachment = (value: boolean) => {
    globalState.needsPlanModeExitAttachment = value;
};

export const updatePlanModeExitAttachment = (previousMode: string, nextMode: string) => {
    if (nextMode === "plan" && previousMode !== "plan") {
        globalState.needsPlanModeExitAttachment = false;
    }
    if (previousMode === "plan" && nextMode !== "plan") {
        globalState.needsPlanModeExitAttachment = true;
    }
};

export const getSessionId = () => globalState.sessionId;
export const setSessionId = (id: string) => {
    globalState.sessionId = id;
    process.env.CLAUDE_CODE_SESSION_ID = id;
};

export const getOriginalCwd = () => globalState.originalCwd;
export const setOriginalCwd = (path: string) => { globalState.originalCwd = path; };

export const getCurrentCwd = () => globalState.cwd;
export const setCurrentCwd = (path: string) => { globalState.cwd = path; };

export const getInMemoryErrorLog = () => [...globalState.inMemoryErrorLog];
export const addToInMemoryErrorLog = (error: any) => {
    if (globalState.inMemoryErrorLog.length >= 100) globalState.inMemoryErrorLog.shift();
    globalState.inMemoryErrorLog.push(error);
};

