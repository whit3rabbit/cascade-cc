/**
 * File: src/services/teams/TeamMessages.ts
 * Role: Structured swarm message helpers for mailbox communication.
 */

export interface ShutdownRequestPayload {
    requestId: string;
    from: string;
    reason?: string;
}

export interface ShutdownApprovedPayload {
    requestId: string;
    from: string;
    paneId?: string;
    backendType?: string;
}

export interface ShutdownRejectedPayload {
    requestId: string;
    from: string;
    reason: string;
}

export interface PlanApprovalResponsePayload {
    requestId: string;
    approvedApiKeys: boolean;
    feedback?: string;
    permissionMode?: string;
}

export interface JoinRequestPayload {
    requestId: string;
    sessionId: string;
    proposedName: string;
    capabilities?: string;
    cwd?: string;
}

export interface JoinApprovedPayload {
    requestId: string;
    teamName: string;
    teamFilePath: string;
    leadAgentId: string;
    agentId: string;
    agentName: string;
    color: string;
    planModeRequired: boolean;
}

export interface JoinRejectedPayload {
    requestId: string;
    reason: string;
}

export function buildShutdownRequest(payload: ShutdownRequestPayload) {
    return {
        type: "shutdown_request",
        requestId: payload.requestId,
        from: payload.from,
        reason: payload.reason,
        timestamp: new Date().toISOString()
    };
}

export function buildShutdownApproved(payload: ShutdownApprovedPayload) {
    return {
        type: "shutdown_approved",
        requestId: payload.requestId,
        from: payload.from,
        timestamp: new Date().toISOString(),
        paneId: payload.paneId,
        backendType: payload.backendType
    };
}

export function buildShutdownRejected(payload: ShutdownRejectedPayload) {
    return {
        type: "shutdown_rejected",
        requestId: payload.requestId,
        from: payload.from,
        reason: payload.reason,
        timestamp: new Date().toISOString()
    };
}

export function buildPlanApprovalResponse(payload: PlanApprovalResponsePayload) {
    return {
        type: "plan_approval_response",
        requestId: payload.requestId,
        approvedApiKeys: payload.approvedApiKeys,
        feedback: payload.feedback,
        permissionMode: payload.permissionMode,
        timestamp: new Date().toISOString()
    };
}

export function buildJoinRequest(payload: JoinRequestPayload) {
    return {
        type: "join_request",
        requestId: payload.requestId,
        sessionId: payload.sessionId,
        proposedName: payload.proposedName,
        capabilities: payload.capabilities,
        cwd: payload.cwd,
        timestamp: new Date().toISOString()
    };
}

export function buildJoinApproved(payload: JoinApprovedPayload) {
    return {
        type: "join_approved",
        requestId: payload.requestId,
        teamName: payload.teamName,
        teamFilePath: payload.teamFilePath,
        leadAgentId: payload.leadAgentId,
        agentId: payload.agentId,
        agentName: payload.agentName,
        color: payload.color,
        planModeRequired: payload.planModeRequired,
        timestamp: new Date().toISOString()
    };
}

export function buildJoinRejected(payload: JoinRejectedPayload) {
    return {
        type: "join_rejected",
        requestId: payload.requestId,
        reason: payload.reason,
        timestamp: new Date().toISOString()
    };
}
