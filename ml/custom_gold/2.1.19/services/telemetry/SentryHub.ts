/**
 * File: src/services/telemetry/SentryHub.ts
 * Role: Manages exception tracking, breadcrumbs, and user context using @sentry/node.
 */

import * as Sentry from '@sentry/node';
import { EnvService } from '../config/EnvService.js';

export interface Breadcrumb {
    message?: string;
    category?: string;
    level?: Sentry.SeverityLevel;
    data?: Record<string, any>;
    timestamp?: number;
    type?: string;
}

export interface SentryUser {
    id?: string;
    username?: string;
    email?: string;
    [key: string]: any;
}

/**
 * Hub manages the current scope and client for telemetry reporting.
 */
export class SentryHub {
    private static instance: SentryHub;
    private initialized: boolean = false;

    constructor() {
        this.initialize();
    }

    public static getInstance(): SentryHub {
        if (!SentryHub.instance) {
            SentryHub.instance = new SentryHub();
        }
        return SentryHub.instance;
    }

    private initialize() {
        if (this.initialized) return;

        const dsn = EnvService.get('SENTRY_DSN'); // Fallback or empty if not provided
        const environment = EnvService.get('NODE_ENV') || 'production';
        const release = '2.1.19'; // Matching the deobfuscated version

        if (dsn) {
            Sentry.init({
                dsn,
                environment,
                release,
                // Disable automatic instrumentation that might interfere with CLI
                defaultIntegrations: false,
                integrations: [
                    Sentry.onUncaughtExceptionIntegration(),
                    Sentry.onUnhandledRejectionIntegration(),
                ],
            });
            this.initialized = true;
        }
    }

    captureException(err: Error | unknown, hint?: any): void {
        const message = err instanceof Error ? err.message : String(err);
        if (EnvService.isTruthy('DEBUG_TELEMETRY')) {
            console.error(`[Telemetry] Captured exception: ${message}`, hint);
        }

        if (this.initialized) {
            Sentry.captureException(err, hint);
        }
    }

    captureMessage(message: string, level?: Sentry.SeverityLevel): void {
        if (EnvService.isTruthy('DEBUG_TELEMETRY')) {
            console.log(`[Telemetry] Captured message: ${message}`);
        }
        if (this.initialized) {
            Sentry.captureMessage(message, level);
        }
    }

    addBreadcrumb(crumb: Breadcrumb): void {
        if (this.initialized) {
            Sentry.addBreadcrumb(crumb);
        }
    }

    setUser(user: SentryUser | null): void {
        if (this.initialized) {
            Sentry.setUser(user);
        }
    }

    setTag(key: string, value: string): void {
        if (this.initialized) {
            Sentry.setTag(key, value);
        }
    }

    setContext(key: string, context: Record<string, any> | null): void {
        if (this.initialized) {
            Sentry.setContext(key, context);
        }
    }
}

const globalHub = SentryHub.getInstance();

/**
 * Returns the global telemetry hub.
 */
export function getCurrentHub(): SentryHub {
    return globalHub;
}
