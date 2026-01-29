/**
 * File: src/services/telemetry/SentryHub.ts
 * Role: Manages exception tracking, breadcrumbs, and user context (Sentry configuration).
 */

export interface Breadcrumb {
    message?: string;
    category?: string;
    level?: string;
    type?: string;
    data?: Record<string, any>;
    timestamp?: number;
}

export interface SentryUser {
    id?: string;
    username?: string;
    email?: string;
    [key: string]: any;
}

export interface SentryScope {
    breadcrumbs: Breadcrumb[];
    user: SentryUser | null;
    tags: Record<string, string>;
    addBreadcrumb(crumb: Breadcrumb): void;
    setUser(user: SentryUser | null): void;
    setTag(key: string, value: string): void;
}

/**
 * Hub manages the current scope and client for telemetry reporting.
 */
export class SentryHub {
    private client: any;
    private scope: SentryScope;

    constructor(client?: any, scope?: SentryScope) {
        this.client = client;
        this.scope = scope || {
            breadcrumbs: [],
            user: null,
            tags: {},
            addBreadcrumb(crumb: Breadcrumb) {
                this.breadcrumbs.push(crumb);
                if (this.breadcrumbs.length > 100) this.breadcrumbs.shift();
            },
            setUser(user: SentryUser | null) { this.user = user; },
            setTag(k: string, v: string) { this.tags[k] = v; }
        };
    }

    captureException(err: Error, hint?: any): void {
        console.error(`[Telemetry] Captured exception: ${err.message}`, hint);
        if (this.client && typeof this.client.captureException === 'function') {
            this.client.captureException(err, { ...this.scope, ...hint });
        }
    }

    addBreadcrumb(crumb: Breadcrumb): void {
        this.scope.addBreadcrumb({
            timestamp: Date.now() / 1000,
            ...crumb
        });
    }

    setUser(user: SentryUser | null): void {
        this.scope.setUser(user);
    }

    setTag(key: string, value: string): void {
        this.scope.setTag(key, value);
    }
}

const globalHub = new SentryHub();

/**
 * Returns the global telemetry hub.
 */
export function getCurrentHub(): SentryHub {
    return globalHub;
}
