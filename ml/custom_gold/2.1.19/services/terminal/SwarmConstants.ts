/**
 * File: src/services/terminal/SwarmConstants.ts
 * Role: Shared constants for swarm/tmux coordination.
 */

export const SWARM_SESSION_NAME = "claude-swarm";
export const SWARM_WINDOW_NAME = "swarm-view";
export const HIDDEN_TMUX_SESSION = "claude-hidden";
export const TEAM_LEAD_NAME = "team-lead";

export function getSwarmSessionNameForPid(pid = process.pid): string {
    return `claude-swarm-${pid}`;
}
