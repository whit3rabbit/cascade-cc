
export class SessionManager {
    private currentSessionId: string;
    private startTime: number;

    constructor() {
        this.currentSessionId = this.generateSessionId();
        this.startTime = Date.now();
    }

    private generateSessionId(): string {
        return `sess_${Math.random().toString(36).substr(2, 9)}`;
    }

    public getSession(): object {
        return {
            id: this.currentSessionId,
            duration: Date.now() - this.startTime
        };
    }
}
