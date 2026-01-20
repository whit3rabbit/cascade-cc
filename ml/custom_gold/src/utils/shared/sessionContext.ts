import { AsyncLocalStorage } from "node:async_hooks";

export interface SessionContext {
    agentId?: string;
    parentSessionId?: string;
    agentType?: string;
    [key: string]: any;
}

/**
 * AsyncLocalStorage instance for storing session-specific data.
 */
export const sessionStorage = new AsyncLocalStorage<SessionContext>();

/**
 * Returns the current session context from storage.
 */
export function getSessionContext(): SessionContext | undefined {
    return sessionStorage.getStore();
}

/**
 * Runs a function within a specified session context.
 */
export function runInSessionContext<T>(context: SessionContext, fn: () => T): T {
    return sessionStorage.run(context, fn);
}

/**
 * Retrieves agent-related context, falling back to environment variables if storage is empty.
 */
export function getAgentContext(): SessionContext {
    const context = getSessionContext();
    if (context) {
        return {
            agentId: context.agentId,
            parentSessionId: context.parentSessionId,
            agentType: context.agentType
        };
    }

    return {
        ...(process.env.CLAUDE_CODE_PARENT_SESSION_ID ? {
            parentSessionId: process.env.CLAUDE_CODE_PARENT_SESSION_ID
        } : {})
    };
}
