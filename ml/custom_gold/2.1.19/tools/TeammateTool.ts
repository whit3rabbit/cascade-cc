/**
 * File: src/tools/TeammateTool.ts
 * Role: Tool for coordinating swarm teammates and team operations.
 */

import {
    createTeam,
    listTeams,
    readTeamConfig,
    removeTeam,
    writeTeamConfig,
    generateMemberName,
    getTeamConfigPath
} from "../services/teams/TeamManager.js";
import {
    writeToMailbox,
    writeToPendingInbox,
    readPendingInbox,
    TeammateMailboxMessage
} from "../services/teams/TeammateMailbox.js";
import {
    buildJoinApproved,
    buildJoinRejected,
    buildJoinRequest,
    buildPlanApprovalResponse,
    buildShutdownApproved,
    buildShutdownRejected,
    buildShutdownRequest
} from "../services/teams/TeamMessages.js";
import { createAgentId, createRequestId } from "../services/teams/TeamIdentity.js";
import { TEAM_LEAD_NAME } from "../services/terminal/SwarmConstants.js";
import { getAgentId, getSessionId } from "../utils/shared/runtimeAndEnv.js";
import { TmuxBackend } from "../services/terminal/TmuxBackend.js";
import { notificationQueue } from "../services/terminal/NotificationService.js";

export interface TeammateToolInput {
    operation:
    | "spawnTeam"
    | "cleanup"
    | "write"
    | "broadcast"
    | "requestShutdown"
    | "approveShutdown"
    | "rejectShutdown"
    | "approvePlan"
    | "rejectPlan"
    | "discoverTeams"
    | "requestJoin"
    | "approveJoin"
    | "rejectJoin"
    | "hide"
    | "show"
    | "spawn";
    team_name?: string;
    description?: string;
    name?: string;
    key?: string;
    value?: string;
    target_agent_id?: string;
    request_id?: string;
    feedback?: string;
    reason?: string;
    proposed_name?: string;
    capabilities?: string;
    assigned_name?: string;
    timeout_ms?: number;
}

export interface TeammateToolResult {
    success: boolean;
    message: string;
    [key: string]: any;
}

import { EnvService } from '../services/config/EnvService.js';

const DEFAULT_COLORS = ["cyan", "green", "yellow", "magenta", "blue", "red"];

function getTeamName(input?: TeammateToolInput): string | undefined {
    return input?.team_name || EnvService.get("CLAUDE_CODE_TEAM_NAME") || undefined;
}

function getSenderName(input?: TeammateToolInput): string {
    return EnvService.get("CLAUDE_CODE_AGENT_NAME") || input?.name || TEAM_LEAD_NAME;
}

function pickColor(seed: string): string {
    let hash = 0;
    for (let i = 0; i < seed.length; i++) {
        hash = (hash * 31 + seed.charCodeAt(i)) | 0;
    }
    const idx = Math.abs(hash) % DEFAULT_COLORS.length;
    return DEFAULT_COLORS[idx];
}

function buildMessage(text: string, sender: string): TeammateMailboxMessage {
    return {
        from: sender,
        text,
        timestamp: new Date().toISOString()
    };
}

function requireTeamName(teamName?: string): string {
    if (!teamName) {
        throw new Error("team_name is required for this operation. Set CLAUDE_CODE_TEAM_NAME or provide team_name in input.");
    }
    return teamName;
}

function parseRequestTarget(requestId: string): string | null {
    const match = requestId.match(/@([^@]+)$/);
    return match?.[1] ?? null;
}

function isTeamLead(teamName: string): boolean {
    const agentName = EnvService.get("CLAUDE_CODE_AGENT_NAME") || TEAM_LEAD_NAME;
    const agentId = getAgentId();
    const config = readTeamConfig(teamName);
    if (!config) return agentName === TEAM_LEAD_NAME;
    if (config.leadAgentId && agentId) return config.leadAgentId === agentId;
    return agentName === TEAM_LEAD_NAME;
}

export const TeammateTool = {
    name: "Teammate",
    description: "Manage teams and coordinate teammates in a swarm.",
    prompt: "Manage teams and coordinate teammates in a swarm. Use this tool for team operations, communication, and task assignment.",
    inputSchema: {
        operation: { type: "string" },
        team_name: { type: "string" },
        description: { type: "string" },
        name: { type: "string" },
        key: { type: "string" },
        value: { type: "string" },
        target_agent_id: { type: "string" },
        request_id: { type: "string" },
        feedback: { type: "string" },
        reason: { type: "string" },
        proposed_name: { type: "string" },
        capabilities: { type: "string" },
        assigned_name: { type: "string" },
        timeout_ms: { type: "number" }
    },
    outputSchema: {
        success: { type: "boolean" },
        message: { type: "string" }
    },
    async call(input: TeammateToolInput): Promise<{ data: TeammateToolResult }> {
        const teamName = getTeamName(input);

        switch (input.operation) {
            case "spawnTeam": {
                if (!input.team_name || input.team_name.trim().length === 0) {
                    throw new Error("team_name is required for spawnTeam operation");
                }
                const name = input.team_name.trim();
                const senderName = getSenderName(input);
                const agentId = getAgentId() || createAgentId(senderName, name);
                const memberColor = pickColor(agentId);
                const config = createTeam({
                    name,
                    description: input.description,
                    leadAgentId: agentId,
                    members: [
                        {
                            name: senderName,
                            agentId,
                            color: memberColor,
                            joinedAt: Date.now(),
                            tmuxPaneId: "",
                            cwd: process.cwd(),
                            subscriptions: []
                        }
                    ]
                });
                notificationQueue.add({
                    text: `Team "${config.name}" spawned`,
                    type: 'success',
                    timeoutMs: 3000
                });
                return {
                    data: {
                        success: true,
                        message: `Team "${config.name}" created`,
                        team_name: config.name,
                        teamFilePath: getTeamConfigPath(config.name)
                    }
                };
            }
            case "cleanup": {
                const resolved = requireTeamName(teamName);
                const config = readTeamConfig(resolved);
                if (config && config.members && config.members.length > 1) {
                    return {
                        data: {
                            success: false,
                            message: "cleanup failed: team still has active members"
                        }
                    };
                }
                removeTeam(resolved);
                return {
                    data: {
                        success: true,
                        message: `Team "${resolved}" removed`
                    }
                };
            }
            case "write": {
                const resolved = requireTeamName(teamName);
                if (!input.target_agent_id) {
                    throw new Error("target_agent_id is required for write operation");
                }
                const senderName = getSenderName(input);
                const messageText = input.key ? `[${input.key}] ${input.value ?? ""}` : String(input.value ?? "");
                writeToMailbox(input.target_agent_id, buildMessage(messageText, senderName), resolved);
                return {
                    data: {
                        success: true,
                        message: `Message sent to ${input.target_agent_id}'s inbox`
                    }
                };
            }
            case "broadcast": {
                const resolved = requireTeamName(teamName);
                const config = readTeamConfig(resolved);
                if (!config) {
                    throw new Error(`Team "${resolved}" does not exist`);
                }
                if (input.value === undefined) {
                    throw new Error("value is required for broadcast operation");
                }
                const senderName = getSenderName(input);
                const messageText = input.key ? `[${input.key}] ${input.value}` : String(input.value);
                const recipients = config.members
                    .map(member => member.name)
                    .filter(name => name && name.toLowerCase() !== senderName.toLowerCase());
                if (recipients.length === 0) {
                    return {
                        data: {
                            success: true,
                            message: "No teammates to broadcast to (you are the only team member)",
                            notifiedCount: 0,
                            recipients: []
                        }
                    };
                }
                for (const recipient of recipients) {
                    writeToMailbox(recipient, buildMessage(messageText, senderName), resolved);
                }
                return {
                    data: {
                        success: true,
                        message: `Message broadcast to ${recipients.length} teammate(s): ${recipients.join(", ")}`,
                        notifiedCount: recipients.length,
                        recipients
                    }
                };
            }
            case "requestShutdown": {
                const resolved = requireTeamName(teamName);
                if (!input.target_agent_id) {
                    throw new Error("target_agent_id is required for requestShutdown operation");
                }
                const senderName = getSenderName(input);
                const requestId = createRequestId("shutdown", resolved);
                const payload = buildShutdownRequest({
                    requestId,
                    from: senderName,
                    reason: input.reason
                });
                writeToMailbox(input.target_agent_id, buildMessage(JSON.stringify(payload), senderName), resolved);
                return {
                    data: {
                        success: true,
                        message: `Shutdown request sent to ${input.target_agent_id}. Request ID: ${requestId}`,
                        requestId,
                        target: input.target_agent_id
                    }
                };
            }
            case "approveShutdown": {
                const resolved = requireTeamName(teamName);
                if (!input.request_id) {
                    throw new Error("request_id is required for approveShutdown");
                }
                const senderName = getSenderName(input);
                const payload = buildShutdownApproved({
                    requestId: input.request_id,
                    from: senderName
                });
                writeToMailbox(TEAM_LEAD_NAME, buildMessage(JSON.stringify(payload), senderName), resolved);
                return {
                    data: {
                        success: true,
                        message: `Shutdown approved. Sent confirmation to ${TEAM_LEAD_NAME}.`,
                        requestId: input.request_id
                    }
                };
            }
            case "rejectShutdown": {
                const resolved = requireTeamName(teamName);
                if (!input.request_id) {
                    throw new Error("request_id is required for rejectShutdown");
                }
                if (!input.reason) {
                    throw new Error("reason is required for rejectShutdown");
                }
                const senderName = getSenderName(input);
                const payload = buildShutdownRejected({
                    requestId: input.request_id,
                    from: senderName,
                    reason: input.reason
                });
                writeToMailbox(TEAM_LEAD_NAME, buildMessage(JSON.stringify(payload), senderName), resolved);
                return {
                    data: {
                        success: true,
                        message: `Shutdown rejected. Reason: "${input.reason}".`,
                        requestId: input.request_id
                    }
                };
            }
            case "approvePlan": {
                const resolved = requireTeamName(teamName);
                if (!isTeamLead(resolved)) {
                    throw new Error("Only the team lead can approve plans.");
                }
                if (!input.target_agent_id || !input.request_id) {
                    throw new Error("target_agent_id and request_id are required for approvePlan");
                }
                const payload = buildPlanApprovalResponse({
                    requestId: input.request_id,
                    approvedApiKeys: true
                });
                writeToMailbox(input.target_agent_id, buildMessage(JSON.stringify(payload), TEAM_LEAD_NAME), resolved);
                return {
                    data: {
                        success: true,
                        message: `Plan approved for ${input.target_agent_id}.`
                    }
                };
            }
            case "rejectPlan": {
                const resolved = requireTeamName(teamName);
                if (!isTeamLead(resolved)) {
                    throw new Error("Only the team lead can reject plans.");
                }
                if (!input.target_agent_id || !input.request_id) {
                    throw new Error("target_agent_id and request_id are required for rejectPlan");
                }
                const payload = buildPlanApprovalResponse({
                    requestId: input.request_id,
                    approvedApiKeys: false,
                    feedback: input.feedback || "Plan needs revision"
                });
                writeToMailbox(input.target_agent_id, buildMessage(JSON.stringify(payload), TEAM_LEAD_NAME), resolved);
                return {
                    data: {
                        success: true,
                        message: `Plan rejected for ${input.target_agent_id} with feedback: "${input.feedback || "Plan needs revision"}"`
                    }
                };
            }
            case "discoverTeams": {
                const currentTeam = teamName;
                const teams = listTeams()
                    .filter(team => team.name !== currentTeam)
                    .map(team => ({
                        name: team.name,
                        description: team.description,
                        leadAgentId: team.leadAgentId,
                        memberCount: team.members?.length ?? 0
                    }));
                return {
                    data: {
                        success: true,
                        teams,
                        message: teams.length > 0
                            ? `Found ${teams.length} team(s) available to join`
                            : "No teams available to join"
                    }
                };
            }
            case "requestJoin": {
                if (!input.team_name) {
                    throw new Error("team_name is required for requestJoin operation");
                }
                if (!input.proposed_name) {
                    return {
                        data: {
                            success: false,
                            message: "proposed_name is required for requestJoin operation. Please provide a name to join the team.",
                            status: "error"
                        }
                    };
                }
                const targetTeam = input.team_name;
                const config = readTeamConfig(targetTeam);
                if (!config) {
                    return {
                        data: {
                            success: false,
                            message: `Team "${targetTeam}" not found`,
                            status: "team_not_found"
                        }
                    };
                }
                const sessionId = getSessionId();
                const requestId = input.request_id || createRequestId("join", targetTeam);
                const payload = buildJoinRequest({
                    requestId,
                    sessionId,
                    proposedName: input.proposed_name,
                    capabilities: input.capabilities,
                    cwd: process.cwd()
                });
                const leaderName = config.members.find(member => member.agentId === config.leadAgentId)?.name || TEAM_LEAD_NAME;
                writeToMailbox(leaderName, buildMessage(JSON.stringify(payload), input.proposed_name), targetTeam);

                if (input.timeout_ms && input.timeout_ms > 0) {
                    const deadline = Date.now() + input.timeout_ms;
                    while (Date.now() < deadline) {
                        const pending = readPendingInbox(targetTeam, sessionId);
                        for (const message of pending) {
                            try {
                                const parsed = JSON.parse(message.text);
                                if (parsed?.type === "join_approved" && parsed.requestId === requestId) {
                                    return {
                                        data: {
                                            success: true,
                                            message: `Successfully joined team "${targetTeam}" as ${parsed.agentName}`,
                                            status: "approved",
                                            requestId,
                                            teamName: targetTeam,
                                            agentId: parsed.agentId,
                                            agentName: parsed.agentName
                                        }
                                    };
                                }
                                if (parsed?.type === "join_rejected" && parsed.requestId === requestId) {
                                    return {
                                        data: {
                                            success: false,
                                            message: `Join request rejected: ${parsed.reason || "No reason provided"}`,
                                            status: "rejected",
                                            requestId,
                                            teamName: targetTeam
                                        }
                                    };
                                }
                            } catch {
                                continue;
                            }
                        }
                        await new Promise(resolve => setTimeout(resolve, 1000));
                    }
                    return {
                        data: {
                            success: false,
                            message: `Join request timed out after ${input.timeout_ms / 1000} seconds. The team leader may be busy or unavailable.`,
                            status: "timeout",
                            requestId,
                            teamName: targetTeam
                        }
                    };
                }

                return {
                    data: {
                        success: true,
                        message: `Join request sent to ${leaderName} for team "${targetTeam}"`,
                        status: "pending",
                        requestId,
                        teamName: targetTeam
                    }
                };
            }
            case "approveJoin": {
                const resolved = requireTeamName(teamName);
                if (!isTeamLead(resolved)) {
                    return {
                        data: {
                            success: false,
                            message: "Only the team leader can approve join requests. Use spawn to create teammates.",
                            member_name: "",
                            member_agent_id: ""
                        }
                    };
                }
                if (!input.target_agent_id || !input.request_id) {
                    throw new Error("target_agent_id and request_id are required for approveJoin operation");
                }
                const config = readTeamConfig(resolved);
                if (!config) {
                    return {
                        data: {
                            success: false,
                            message: `Team "${resolved}" config not found`,
                            member_name: "",
                            member_agent_id: ""
                        }
                    };
                }
                const sessionId = parseRequestTarget(input.request_id);
                if (!sessionId) {
                    return {
                        data: {
                            success: false,
                            message: `Invalid request_id format: ${input.request_id}`,
                            member_name: "",
                            member_agent_id: ""
                        }
                    };
                }
                const assignedName = input.assigned_name || input.target_agent_id;
                const uniqueName = generateMemberName(assignedName, resolved);
                const agentId = createAgentId(uniqueName, resolved);
                const color = pickColor(agentId);

                config.members.push({
                    name: uniqueName,
                    agentId,
                    color,
                    joinedAt: Date.now(),
                    tmuxPaneId: "",
                    cwd: "",
                    subscriptions: []
                });
                writeTeamConfig(resolved, config);

                const payload = buildJoinApproved({
                    requestId: input.request_id,
                    teamName: resolved,
                    teamFilePath: getTeamConfigPath(resolved),
                    leadAgentId: config.leadAgentId || "",
                    agentId,
                    agentName: uniqueName,
                    color,
                    planModeRequired: false
                });
                writeToPendingInbox(resolved, sessionId, buildMessage(JSON.stringify(payload), TEAM_LEAD_NAME));
                notificationQueue.add({
                    text: `Teammate ${uniqueName} joined`,
                    type: 'info',
                    timeoutMs: 3000
                });
                return {
                    data: {
                        success: true,
                        message: `Successfully approved ${uniqueName} to join the team`,
                        member_name: uniqueName,
                        member_agent_id: agentId
                    }
                };
            }
            case "rejectJoin": {
                const resolved = requireTeamName(teamName);
                if (!isTeamLead(resolved)) {
                    return {
                        data: {
                            success: false,
                            message: "Only the team leader can reject join requests",
                            requestId: input.request_id
                        }
                    };
                }
                if (!input.target_agent_id || !input.request_id) {
                    throw new Error("target_agent_id and request_id are required for rejectJoin operation");
                }
                const sessionId = parseRequestTarget(input.request_id);
                if (!sessionId) {
                    return {
                        data: {
                            success: false,
                            message: `Invalid request_id format: ${input.request_id}`,
                            requestId: input.request_id
                        }
                    };
                }
                const payload = buildJoinRejected({
                    requestId: input.request_id,
                    reason: input.reason || "Join request was rejected by the team leader"
                });
                writeToPendingInbox(resolved, sessionId, buildMessage(JSON.stringify(payload), TEAM_LEAD_NAME));
                return {
                    data: {
                        success: true,
                        message: `Successfully rejected join request from ${input.target_agent_id}`,
                        requestId: input.request_id
                    }
                };
            }
            case "hide": {
                const resolved = requireTeamName(teamName);
                if (!input.target_agent_id) {
                    throw new Error("target_agent_id is required for hide operation");
                }
                const config = readTeamConfig(resolved);
                if (!config) throw new Error(`Team "${resolved}" not found`);
                const member = config.members.find(m => m.agentId === input.target_agent_id || m.name === input.target_agent_id);
                if (!member || !member.tmuxPaneId) {
                    throw new Error(`Member ${input.target_agent_id} not found or has no active pane`);
                }
                const tmux = new TmuxBackend();
                const success = await tmux.hidePane(member.tmuxPaneId);
                return {
                    data: {
                        success,
                        message: success ? `Pane for ${member.name} hidden` : `Failed to hide pane for ${member.name}`
                    }
                };
            }
            case "show": {
                const resolved = requireTeamName(teamName);
                if (!input.target_agent_id) {
                    throw new Error("target_agent_id is required for show operation");
                }
                const config = readTeamConfig(resolved);
                if (!config) throw new Error(`Team "${resolved}" not found`);
                const member = config.members.find(m => m.agentId === input.target_agent_id || m.name === input.target_agent_id);
                if (!member || !member.tmuxPaneId) {
                    throw new Error(`Member ${input.target_agent_id} not found or has no active pane`);
                }
                const tmux = new TmuxBackend();
                const targetWindow = await tmux.getCurrentWindowTarget();
                if (!targetWindow) throw new Error("Could not determine current tmux window");
                const success = await tmux.showPane(member.tmuxPaneId, targetWindow);
                return {
                    data: {
                        success,
                        message: success ? `Pane for ${member.name} reshown` : `Failed to show pane for ${member.name}`
                    }
                };
            }
            case "spawn": {
                const resolved = requireTeamName(teamName);
                if (!input.proposed_name) {
                    throw new Error("proposed_name is required for spawn operation");
                }
                const senderName = getSenderName(input);
                if (!isTeamLead(resolved)) {
                    throw new Error("Only the team lead can spawn new teammates.");
                }

                const uniqueName = generateMemberName(input.proposed_name, resolved);
                const agentId = createAgentId(uniqueName, resolved);
                const color = pickColor(agentId);

                // Initialize Tmux Backend
                const tmux = new TmuxBackend();
                if (!(await tmux.isAvailable())) {
                    throw new Error("tmux is not available in this environment. Cannot spawn teammate.");
                }

                // Create Pane
                let paneInfo;
                try {
                    paneInfo = await tmux.createTeammatePaneInSwarmView(uniqueName, color);
                } catch (err: any) {
                    throw new Error(`Failed to create tmux pane: ${err.message}`);
                }

                // Construct Command
                const isDev = EnvService.get("NODE_ENV") === 'development' || process.argv[1].includes('.ts') || process.argv[1].includes('cli.tsx');
                const requestId = createRequestId("join", resolved);
                let command;
                if (isDev) {
                    command = `npm run dev -- --agent ${uniqueName} --auto-join ${requestId}`;
                } else {
                    command = `claude --agent ${uniqueName} --auto-join ${requestId}`;
                }

                await tmux.sendCommandToPane(paneInfo.paneId, command);

                // Update Config
                const config = readTeamConfig(resolved);
                if (config) {
                    config.members.push({
                        name: uniqueName,
                        agentId,
                        color,
                        joinedAt: Date.now(),
                        tmuxPaneId: paneInfo.paneId,
                        cwd: process.cwd(),
                        subscriptions: []
                    });
                    writeTeamConfig(resolved, config);
                }

                notificationQueue.add({
                    text: `Teammate ${uniqueName} spawned in new pane`,
                    type: 'success',
                    timeoutMs: 5000
                });

                return {
                    data: {
                        success: true,
                        message: `Spawned teammate ${uniqueName} in new pane`,
                        member_name: uniqueName,
                        member_agent_id: agentId,
                        pane_id: paneInfo.paneId
                    }
                };
            }
            default:
                throw new Error(`Unknown teammate operation: ${input.operation}`);
        }
    }
};
