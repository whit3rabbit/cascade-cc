import * as fs from "node:fs";
import * as path from "node:path";
import { getSessionId } from "./sessionStore.js";
import { getProjectRoot } from "../../utils/shared/pathUtils.js";

import { SessionIndexManager } from "./SessionIndex.js";

// Types for log entries
export interface SessionLogEntry {
    type: string;
    uuid?: string;
    parentUuid?: string;
    sessionId?: string;
    timestamp?: string;
    [key: string]: any;
}

export class SessionPersistence {
    private static instance: SessionPersistence | null = null;
    private sessionFile: string | null = null;

    private constructor() { }

    static getInstance(): SessionPersistence {
        if (!SessionPersistence.instance) {
            SessionPersistence.instance = new SessionPersistence();
        }
        return SessionPersistence.instance;
    }

    private ensureSessionFile(): string {
        if (this.sessionFile) return this.sessionFile;

        const sessionId = getSessionId();
        const root = getProjectRoot();
        const sessionDir = path.join(root, ".claude", "sessions");

        if (!fs.existsSync(sessionDir)) {
            fs.mkdirSync(sessionDir, { recursive: true });
        }

        this.sessionFile = path.join(sessionDir, `${sessionId}.jsonl`);
        if (!fs.existsSync(this.sessionFile)) {
            fs.writeFileSync(this.sessionFile, "", { mode: 0o600 });
        }

        return this.sessionFile;
    }

    async appendEntry(entry: SessionLogEntry) {
        const file = this.ensureSessionFile();
        const line = JSON.stringify(entry) + "\n";
        fs.appendFileSync(file, line, { mode: 0o600 });
    }

    async insertQueueOperation(operation: any) {
        await this.appendEntry(operation);
    }

    async listLocalSessions(): Promise<any[]> {
        const root = getProjectRoot();
        const sessionDir = path.join(root, ".claude", "sessions");
        const index = SessionIndexManager.readIndex(sessionDir);
        return index ? index.entries : [];
    }
}

export const getSessionPersistence = () => SessionPersistence.getInstance();

export async function logQueueOperationToSession(operation: any) {
    await getSessionPersistence().insertQueueOperation(operation);
}
