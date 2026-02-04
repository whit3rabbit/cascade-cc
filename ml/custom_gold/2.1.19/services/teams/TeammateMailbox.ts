/**
 * File: src/services/teams/TeammateMailbox.ts
 * Role: File-backed mailbox utilities for swarm teammates.
 */

import { existsSync, mkdirSync, readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import lockfile from "proper-lockfile";
import { getTeamsDir, sanitizeTeamName } from "./TeamManager.js";
import { SendMessageContentType, SendMessageSubtype } from "../../types/AgentTypes.js";

export interface TeammateMailboxMessage {
    from: string;
    text: string;
    timestamp: string;
    color?: string;
    read?: boolean;
    subtype?: SendMessageSubtype;
    contentType?: SendMessageContentType;
    requestId?: string;
    approve?: boolean;
    context?: any;
}

function normalizeMailboxId(value: string): string {
    return sanitizeTeamName(value || "unknown");
}

function getTeamsBaseDir(): string {
    return getTeamsDir();
}

export function getInboxPath(agentName: string, teamName?: string): string {
    const safeTeam = normalizeMailboxId(teamName || "default");
    const safeAgent = normalizeMailboxId(agentName);
    const inboxDir = join(getTeamsBaseDir(), safeTeam, "inboxes");
    return join(inboxDir, `${safeAgent}.json`);
}

export function getPendingInboxPath(teamName: string, sessionId: string): string {
    const safeTeam = normalizeMailboxId(teamName || "default");
    const safeSession = normalizeMailboxId(sessionId);
    const inboxDir = join(getTeamsBaseDir(), safeTeam, "inboxes");
    return join(inboxDir, `pending-${safeSession}.json`);
}

export function ensureInboxDir(teamName?: string): void {
    const safeTeam = normalizeMailboxId(teamName || "default");
    const inboxDir = join(getTeamsBaseDir(), safeTeam, "inboxes");
    if (!existsSync(inboxDir)) {
        mkdirSync(inboxDir, { recursive: true });
    }
}

export function readMailbox(agentName: string, teamName?: string): TeammateMailboxMessage[] {
    const inboxPath = getInboxPath(agentName, teamName);
    if (!existsSync(inboxPath)) return [];
    try {
        const content = readFileSync(inboxPath, "utf-8");
        const parsed = JSON.parse(content);
        return Array.isArray(parsed) ? parsed : [];
    } catch {
        return [];
    }
}

export function readUnreadMessages(agentName: string, teamName?: string): TeammateMailboxMessage[] {
    return readMailbox(agentName, teamName).filter(message => !message.read);
}

export function writeToMailbox(agentName: string, message: TeammateMailboxMessage, teamName?: string): void {
    ensureInboxDir(teamName);
    const inboxPath = getInboxPath(agentName, teamName);
    const lockPath = `${inboxPath}.lock`;

    if (!existsSync(inboxPath)) {
        writeFileSync(inboxPath, "[]", "utf-8");
    }

    let release: (() => void) | undefined;
    try {
        release = lockfile.lockSync(inboxPath, { lockfilePath: lockPath });
        const messages = readMailbox(agentName, teamName);
        messages.push({ ...message, read: false });
        writeFileSync(inboxPath, JSON.stringify(messages, null, 2), "utf-8");
    } finally {
        if (release) {
            try {
                release();
            } catch {
                // Ignore lock release errors.
            }
        }
    }
}

export function markMessageAsReadByIndex(agentName: string, index: number, teamName?: string): void {
    const inboxPath = getInboxPath(agentName, teamName);
    if (!existsSync(inboxPath)) return;
    const lockPath = `${inboxPath}.lock`;
    let release: (() => void) | undefined;

    try {
        release = lockfile.lockSync(inboxPath, { lockfilePath: lockPath });
        const messages = readMailbox(agentName, teamName);
        if (index < 0 || index >= messages.length) return;
        if (messages[index]?.read) return;
        messages[index] = { ...messages[index], read: true };
        writeFileSync(inboxPath, JSON.stringify(messages, null, 2), "utf-8");
    } finally {
        if (release) {
            try {
                release();
            } catch {
                // Ignore lock release errors.
            }
        }
    }
}

export function markMessagesAsRead(agentName: string, teamName?: string): void {
    const inboxPath = getInboxPath(agentName, teamName);
    if (!existsSync(inboxPath)) return;
    const lockPath = `${inboxPath}.lock`;
    let release: (() => void) | undefined;

    try {
        release = lockfile.lockSync(inboxPath, { lockfilePath: lockPath });
        const messages = readMailbox(agentName, teamName).map(message => ({
            ...message,
            read: true
        }));
        writeFileSync(inboxPath, JSON.stringify(messages, null, 2), "utf-8");
    } finally {
        if (release) {
            try {
                release();
            } catch {
                // Ignore lock release errors.
            }
        }
    }
}

export function clearMailbox(agentName: string, teamName?: string): void {
    const inboxPath = getInboxPath(agentName, teamName);
    if (!existsSync(inboxPath)) return;
    try {
        unlinkSync(inboxPath);
    } catch {
        // Ignore clear errors.
    }
}

export function readPendingInbox(teamName: string, sessionId: string): TeammateMailboxMessage[] {
    const inboxPath = getPendingInboxPath(teamName, sessionId);
    if (!existsSync(inboxPath)) return [];
    try {
        const content = readFileSync(inboxPath, "utf-8");
        const parsed = JSON.parse(content);
        return Array.isArray(parsed) ? parsed : [];
    } catch {
        return [];
    }
}

export function writeToPendingInbox(teamName: string, sessionId: string, message: TeammateMailboxMessage): void {
    ensureInboxDir(teamName);
    const inboxPath = getPendingInboxPath(teamName, sessionId);
    const lockPath = `${inboxPath}.lock`;

    if (!existsSync(inboxPath)) {
        writeFileSync(inboxPath, "[]", "utf-8");
    }

    let release: (() => void) | undefined;
    try {
        release = lockfile.lockSync(inboxPath, { lockfilePath: lockPath });
        const messages = readPendingInbox(teamName, sessionId);
        messages.push({ ...message, read: false });
        writeFileSync(inboxPath, JSON.stringify(messages, null, 2), "utf-8");
    } finally {
        if (release) {
            try {
                release();
            } catch {
                // Ignore lock release errors.
            }
        }
    }
}

export function clearPendingInbox(teamName: string, sessionId: string): void {
    const inboxPath = getPendingInboxPath(teamName, sessionId);
    if (!existsSync(inboxPath)) return;
    try {
        unlinkSync(inboxPath);
    } catch {
        // Ignore clear errors.
    }
}
