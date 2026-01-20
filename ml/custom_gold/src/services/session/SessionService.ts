import { z } from "zod";
import axios, { AxiosInstance } from "axios";
import { getAuthToken } from "./sessionStore.js";

export const SessionSchema = z.object({
    id: z.string(),
    title: z.string(),
    description: z.string(),
    status: z.enum(["idle", "working", "waiting", "completed", "archived", "cancelled", "rejected"]),
    repo: z.object({
        name: z.string(),
        owner: z.object({
            login: z.string()
        }),
        default_branch: z.string().optional()
    }).nullable(),
    turns: z.array(z.string()),
    created_at: z.string(),
    updated_at: z.string()
});

export type Session = z.infer<typeof SessionSchema>;

export class SessionService {
    private baseUrl: string;

    constructor(baseUrl: string = "https://api.claude.ai/api") {
        this.baseUrl = baseUrl;
    }

    private async getHeaders() {
        const token = getAuthToken();
        if (!token) {
            throw new Error("No authentication token available. Please login first.");
        }
        return {
            "Authorization": `Bearer ${token}`,
            "Content-Type": "application/json"
        };
    }

    async listSessions(): Promise<Session[]> {
        // Implementation of aA2
        return [];
    }

    async getSession(sessionId: string): Promise<Session> {
        // Implementation of RTA
        throw new Error("Not implemented");
    }

    async persistSessionLog(sessionId: string, logEntry: any, options: any) {
        // Implementation of Rs3
    }
}
