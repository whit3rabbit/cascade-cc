/**
 * File: src/services/teams/TeamIdentity.ts
 * Role: Helpers for swarm agent and request identifiers.
 */

import { sanitizeTeamName } from "./TeamManager.js";

export function createAgentId(agentName: string, teamName: string): string {
    return `${agentName}@${teamName}`;
}

export function parseAgentId(agentId: string): { agentName: string; teamName: string } | null {
    const index = agentId.indexOf("@");
    if (index === -1) return null;
    return {
        agentName: agentId.slice(0, index),
        teamName: agentId.slice(index + 1)
    };
}

export function createRequestId(prefix: string, teamName: string): string {
    return `${prefix}-${Date.now()}@${sanitizeTeamName(teamName)}`;
}

export function normalizeTeamName(teamName?: string): string {
    return sanitizeTeamName(teamName || "default");
}
