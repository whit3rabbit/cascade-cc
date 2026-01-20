
// Logic from chunk_582.ts (Global AppState and Permissions)

import React, { createContext, useContext, useState, useCallback, useMemo, useEffect } from "react";
import { PermissionState } from "../services/permissions/permissionManager.js";

// --- AppState Schema & Initial State (gs) ---
export interface AppState {
    settings: any;
    tasks: Record<string, any>;
    verbose: boolean;
    mainLoopModel: string | null;
    statusLineText?: string;
    toolPermissionContext: PermissionState;
    agentDefinitions: {
        activeAgents: any[];
        allAgents: any[];
    };
    mcp: {
        clients: any[];
        tools: any[];
        resources: Record<string, any>;
    };
    plugins: {
        enabled: any[];
        errors: any[];
        installationStatus: any;
        [key: string]: any;
    };
    notifications: {
        current: any | null;
        queue: any[];
    };
    thinkingEnabled: boolean;
    [key: string]: any;
}

export function getDefaultAppState(): AppState {
    return {
        settings: {},
        tasks: {},
        verbose: false,
        mainLoopModel: null,
        statusLineText: undefined,
        toolPermissionContext: {
            mode: "ask",
            alwaysAllowRules: {},
            alwaysDenyRules: {},
            alwaysAskRules: {},
            additionalWorkingDirectories: new Map()
        },
        agentDefinitions: {
            activeAgents: [],
            allAgents: []
        },
        mcp: {
            clients: [],
            tools: [],
            resources: {}
        },
        plugins: {
            enabled: [],
            errors: [],
            installationStatus: { marketplaces: [], plugins: [] }
        },
        notifications: {
            current: null,
            queue: []
        },
        thinkingEnabled: true,
        gitDiff: { stats: null, perFileStats: new Map(), lastUpdated: 0 },
        promptSuggestion: {
            text: null,
            promptId: null,
            shownAt: 0,
            acceptedAt: 0,
            generationRequestId: null
        },
        queuedCommands: [],
        teamContext: null,
        pendingWorkerRequest: false,
        pendingSandboxRequest: false,
        elicitation: { queue: [] },
        promptSuggestionEnabled: true,
    };
}

// --- AppState Context & Provider (C5) ---
const AppStateContext = createContext<[AppState, (updater: (s: AppState) => AppState) => void] | undefined>(undefined);

// --- Global State Access (Compatibility) ---
let globalAppState: AppState = getDefaultAppState();
let globalStateUpdater: ((updater: (s: AppState) => AppState) => void) | null = null;

export function getAppState() {
    return globalAppState;
}

export function updateAppState(updater: (s: AppState) => AppState) {
    if (globalStateUpdater) {
        globalStateUpdater(updater);
    } else {
        globalAppState = updater(globalAppState);
    }
}

export function AppStateProvider({ children, initialState }: { children: React.ReactNode, initialState?: AppState }) {
    const [state, setState] = useState<AppState>(initialState || getDefaultAppState());

    const updateState = useCallback((updater: (s: AppState) => AppState) => {
        setState(prev => {
            const next = updater(prev);
            // In real app, this might trigger side effects or persist to disk
            return next;
        });
    }, []);

    useEffect(() => {
        globalAppState = state;
    }, [state]);

    useEffect(() => {
        globalStateUpdater = updateState;
        return () => { globalStateUpdater = null; };
    }, [updateState]);

    const contextValue = useMemo(() => [state, updateState] as [AppState, typeof updateState], [state, updateState]);

    return (
        <AppStateContext.Provider value={contextValue}>
            {children}
        </AppStateContext.Provider>
    );
}

export function useAppState() {
    const context = useContext(AppStateContext);
    if (!context) {
        throw new Error("useAppState must be used within an AppStateProvider");
    }
    return context;
}

// --- Permission Decision Logic (zJ7) ---
export interface PermissionDecision {
    behavior: "allow" | "deny" | "ask" | "passthrough";
    decisionReason?: any;
    message?: string;
    updatedInput?: any;
    suggestions?: any[];
}

export async function checkToolPermission(tool: any, input: any, appState: AppState): Promise<PermissionDecision> {
    const { toolPermissionContext } = appState;

    // Helper to check if tool is in a rule set
    const isListed = (rulesMap: Record<string, string[]>): boolean => {
        for (const source of Object.keys(rulesMap)) {
            const rules = rulesMap[source] || [];
            // Simple check: matches tool name exactly or starts with "toolname:"
            if (rules.some(r => r === tool.name || r.startsWith(`${tool.name}:`))) {
                return true;
            }
        }
        return false;
    };

    // 1. Check for explicit deny rules
    if (isListed(toolPermissionContext.alwaysDenyRules)) {
        return { behavior: "deny", message: `Permission to use ${tool.name} has been denied.` };
    }

    // 2. Check for bypass/plan modes
    // PermissionMode is "allow" | "deny" | "ask"
    if (toolPermissionContext.mode === "allow") {
        return { behavior: "allow", updatedInput: input };
    }

    // 3. Check for explicit allow rules
    if (isListed(toolPermissionContext.alwaysAllowRules)) {
        return { behavior: "allow", updatedInput: input };
    }

    // 4. Fallback to asking or passthrough
    return { behavior: "ask", message: `Allow ${tool.name}?` };
}
