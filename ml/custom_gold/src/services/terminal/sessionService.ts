
import { v4 as uuidv4 } from 'uuid';

/**
 * Service for managing session state, deobfuscated from chunk_3.ts.
 */

interface SessionState {
    originalCwd: string;
    cwd: string;
    sessionId: string;
    startTime: number;
    // Add other fields from chunk_3 if needed
}

const state: SessionState = {
    originalCwd: process.cwd(),
    cwd: process.cwd(),
    sessionId: uuidv4(),
    startTime: Date.now()
};

export function getSessionId(): string {
    return state.sessionId;
}

export function resetSessionId(): string {
    state.sessionId = uuidv4();
    return state.sessionId;
}

export function setSessionId(id: string): void {
    state.sessionId = id;
    if (process.env.CLAUDE_CODE_SESSION_ID !== undefined) {
        process.env.CLAUDE_CODE_SESSION_ID = id;
    }
}

export function getOriginalCwd(): string {
    return state.originalCwd;
}

export function setOriginalCwd(cwd: string): void {
    state.originalCwd = cwd;
}

export function getCwd(): string {
    return state.cwd;
}

export function setCwd(cwd: string): void {
    state.cwd = cwd;
}

export function getSessionDuration(): number {
    return Date.now() - state.startTime;
}

export const sessionService = {
    getSessionId,
    resetSessionId,
    setSessionId,
    getOriginalCwd,
    setOriginalCwd,
    getCwd,
    setCwd,
    getSessionDuration
};
