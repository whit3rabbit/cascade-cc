import React, { createContext, useContext, useState, useCallback, useMemo } from 'react';

export interface AppState {
    verbose: boolean;
    toolPermissionContext: {
        mode: string;
        isBypassPermissionsModeAvailable: boolean;
    };
    tasks: Record<string, any>;
    viewingAgentTaskId: string | null;
    promptSuggestion: {
        text: string | null;
        promptId: string | null;
        shownAt: number;
        acceptedAt: number;
        generationRequestId: string | null;
    };
    speculation: {
        status: 'idle' | 'active';
        boundary: any;
        startTime: number;
    };
    queuedCommands: any[];
    promptCoaching: {
        tip: string | null;
    };
    // Add other fields as needed
}

const initialAppState: AppState = {
    verbose: false,
    toolPermissionContext: {
        mode: 'default',
        isBypassPermissionsModeAvailable: false
    },
    tasks: {},
    viewingAgentTaskId: null,
    promptSuggestion: {
        text: null,
        promptId: null,
        shownAt: 0,
        acceptedAt: 0,
        generationRequestId: null
    },
    speculation: {
        status: 'idle',
        boundary: null,
        startTime: 0
    },
    queuedCommands: [],
    promptCoaching: {
        tip: null
    }
};

type AppStateContextType = [AppState, (updater: (prev: AppState) => AppState) => void];

const AppStateContext = createContext<AppStateContextType | undefined>(undefined);

export const AppStateProvider: React.FC<{ children: React.ReactNode; initialState?: Partial<AppState> }> = ({ children, initialState }) => {
    const [state, setState] = useState<AppState>({ ...initialAppState, ...initialState });

    const updateState = useCallback((updater: (prev: AppState) => AppState) => {
        setState(prev => updater(prev));
    }, []);

    const value = useMemo<AppStateContextType>(() => [state, updateState], [state, updateState]);

    return (
        <AppStateContext.Provider value={value}>
            {children}
        </AppStateContext.Provider>
    );
};

export const useAppState = (): AppStateContextType => {
    const context = useContext(AppStateContext);
    if (!context) {
        throw new Error("useAppState must be used within an AppStateProvider");
    }
    return context;
};
