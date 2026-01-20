
// Based on chunk_477.ts:269-400
export type SessionHookType = 'function' | 'bash' | 'json';

export interface SessionHook {
    type: SessionHookType;
    id: string;
    timeout: number;
    callback?: (...args: any[]) => any;
    errorMessage?: string;
    matcher: string;
}

export interface SessionHooksState {
    [sessionId: string]: {
        hooks: {
            [eventName: string]: Array<{
                matcher: string;
                hooks: Array<{
                    hook: SessionHook;
                    onHookSuccess?: (result: any) => void;
                }>;
            }>;
        };
    };
}

// These would likely be part of a global state or context
let sessionHooksState: SessionHooksState = {};

export function registerFunctionHook(
    updateState: (fn: (prev: SessionHooksState) => SessionHooksState) => void,
    sessionId: string,
    eventName: string,
    matcher: string,
    callback: (...args: any[]) => any,
    errorMessage?: string,
    options?: { id?: string; timeout?: number }
): string {
    const id = options?.id || `function-hook-${Date.now()}-${Math.random()}`;
    const hook: SessionHook = {
        type: 'function',
        id,
        timeout: options?.timeout || 5000,
        callback,
        errorMessage,
        matcher
    };

    updateState((prev) => {
        const session = prev[sessionId] || { hooks: {} };
        const eventHooks = session.hooks[eventName] || [];
        const matcherIndex = eventHooks.findIndex((h) => h.matcher === matcher);

        let newEventHooks;
        if (matcherIndex >= 0) {
            newEventHooks = [...eventHooks];
            newEventHooks[matcherIndex] = {
                ...newEventHooks[matcherIndex],
                hooks: [...newEventHooks[matcherIndex].hooks, { hook }]
            };
        } else {
            newEventHooks = [...eventHooks, { matcher, hooks: [{ hook }] }];
        }

        return {
            ...prev,
            [sessionId]: {
                ...session,
                hooks: {
                    ...session.hooks,
                    [eventName]: newEventHooks
                }
            }
        };
    });

    return id;
}

export function getHooksForEvent(state: SessionHooksState, sessionId: string, eventName: string) {
    const session = state[sessionId];
    if (!session) return new Map();

    const hooks = session.hooks[eventName];
    if (!hooks) return new Map();

    return new Map(hooks.map(h => [h.matcher, h.hooks.map(hh => hh.hook)]));
}

export function clearSessionHooks(updateState: (fn: (prev: SessionHooksState) => SessionHooksState) => void, sessionId: string) {
    updateState((prev) => {
        const newState = { ...prev };
        delete newState[sessionId];
        return newState;
    });
}
