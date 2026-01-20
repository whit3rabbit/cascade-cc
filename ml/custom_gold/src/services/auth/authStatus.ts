export interface AuthStatus {
    isAuthenticating: boolean;
    output: string[];
    error?: Error | any;
}

/**
 * Singleton class managing authentication UI state and output.
 * Deobfuscated from iE in chunk_174.ts.
 */
export class AuthenticationStatusManager {
    private static instance: AuthenticationStatusManager | null = null;
    private status: AuthStatus = {
        isAuthenticating: false,
        output: []
    };
    private listeners: Set<(status: AuthStatus) => void> = new Set();

    private constructor() { }

    static getInstance(): AuthenticationStatusManager {
        if (!AuthenticationStatusManager.instance) {
            AuthenticationStatusManager.instance = new AuthenticationStatusManager();
        }
        return AuthenticationStatusManager.instance;
    }

    getStatus(): AuthStatus {
        return {
            ...this.status,
            output: [...this.status.output]
        };
    }

    startAuthentication(): void {
        this.status = {
            isAuthenticating: true,
            output: []
        };
        this.notifyListeners();
    }

    addOutput(text: string): void {
        this.status.output.push(text);
        this.notifyListeners();
    }

    setError(error: Error | any): void {
        this.status.error = error;
        this.notifyListeners();
    }

    endAuthentication(reset: boolean): void {
        if (reset) {
            this.status = {
                isAuthenticating: false,
                output: []
            };
        } else {
            this.status.isAuthenticating = false;
        }
        this.notifyListeners();
    }

    subscribe(listener: (status: AuthStatus) => void): () => void {
        this.listeners.add(listener);
        return () => {
            this.listeners.delete(listener);
        };
    }

    private notifyListeners(): void {
        const currentStatus = this.getStatus();
        this.listeners.forEach((listener) => listener(currentStatus));
    }

    static reset(): void {
        if (AuthenticationStatusManager.instance) {
            AuthenticationStatusManager.instance.listeners.clear();
            AuthenticationStatusManager.instance = null;
        }
    }
}
