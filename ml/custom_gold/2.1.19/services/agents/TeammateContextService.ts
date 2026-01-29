/**
 * File: src/services/agents/TeammateContextService.ts
 * Role: Manages context for multi-agent teams (swarms) using AsyncLocalStorage.
 */

import { AsyncLocalStorage } from "node:async_hooks";

export interface Teammate {
    id: string;
    name: string;
    [key: string]: any;
}

export interface TeamContext {
    agentId?: string;
    leadAgentId?: string;
    teammates?: Record<string, Teammate>;
    [key: string]: any;
}

const contextStorage = new AsyncLocalStorage<TeamContext>();
let globalTeamContext: TeamContext | null = null;

/**
 * Service for accessing and managing teammate/agent contexts.
 */
export const TeammateContextService = {
    /**
     * Gets the current teammate context from async storage or global fallback.
     */
    getContext(): TeamContext | null {
        return contextStorage.getStore() || globalTeamContext;
    },

    /**
     * Executes a callback with a specific teammate context.
     */
    runWithContext<T>(context: TeamContext, callback: () => T): T {
        return contextStorage.run(context, callback);
    },

    /**
     * Sets the global fallback team context.
     */
    setGlobalContext(context: TeamContext | null): void {
        globalTeamContext = context;
    },

    /**
     * Determines the name of the lead agent from a team context.
     */
    getLeadAgentName(context: TeamContext | null): string {
        if (!context || !context.teammates || !context.leadAgentId) return "team-lead";
        const leadId = context.leadAgentId;
        return context.teammates[leadId]?.name || "team-lead";
    },

    /**
     * Checks if the currently active agent is the team lead.
     */
    isCurrentAgentLead(): boolean {
        const ctx = this.getContext();
        if (!ctx?.leadAgentId) return false;
        return ctx.agentId === ctx.leadAgentId;
    },

    /**
     * Returns true if there is an active teammate context.
     */
    isActive(): boolean {
        return !!contextStorage.getStore();
    }
};
