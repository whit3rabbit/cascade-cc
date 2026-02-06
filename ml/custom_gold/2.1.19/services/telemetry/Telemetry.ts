/**
 * File: src/services/telemetry/Telemetry.ts
 * Role: Telemetry Service Interface using Statsig
 */

import StatsigModule from 'statsig-js';
// Handle CJS/ESM interop
const Statsig = (StatsigModule as any).default || StatsigModule;

import { EnvService } from '../config/EnvService.js';
import { Log } from '../statsig/StatsigService.js';

let isInitialized = false;

// Default to a placeholder if not provided, though typically we'd want a real key.
// In the 2.1.19 code, the key might be obscured or fetched remotely.
// We'll allow it to be set via env var STATSIG_CLIENT_KEY.
const STATSIG_CLIENT_KEY = EnvService.get("STATSIG_CLIENT_KEY") || "client-claude-code-placeholder";

export async function initializeTelemetry() {
    if (isInitialized) return;

    try {
        const environment = {
            tier: EnvService.get('NODE_ENV') || 'production',
        };

        // Initialize Statsig
        await Statsig.initialize(
            STATSIG_CLIENT_KEY,
            {
                userID: EnvService.get('USER') || 'unknown',
                email: 'unknown',
            },
            {
                environment,
                localMode: !EnvService.get("STATSIG_CLIENT_KEY"), // Run in local mode if no key provided to avoid errors
            }
        );

        isInitialized = true;
        Log.info("Telemetry Service Initialized via Statsig");

        if (EnvService.isTruthy("DEBUG_TELEMETRY")) {
            console.log(`[Telemetry] Statsig initialized with key: ${STATSIG_CLIENT_KEY.substring(0, 8)}...`);
        }

    } catch (error) {
        console.error('[Telemetry] Failed to initialize Statsig:', error);
    }
}

export function track(eventName: string, properties: Record<string, any> = {}) {
    if (!isInitialized) return;

    try {
        Statsig.logEvent(eventName, null, properties);
        if (EnvService.isTruthy("DEBUG_TELEMETRY")) {
            console.log(`[Telemetry] Tracked event: ${eventName}`, properties);
        }
    } catch (error) {
        if (EnvService.isTruthy("DEBUG_TELEMETRY")) {
            console.error(`[Telemetry] Failed to track event ${eventName}:`, error);
        }
    }
}

export async function shutdownTelemetry() {
    if (isInitialized) {
        Statsig.shutdown();
        isInitialized = false;
        Log.info(`Telemetry Service Shutdown.`);
    }
}

/**
 * Legacy flush sync method - Statsig handles flushing automatically on shutdown, but we can stub this.
 */
export function flushSync() {
    // Statsig-js doesn't expose a flushSync for node process exit reliably in the browser SDK flavor,
    // but typically shutdown() handles pending events.
    if (isInitialized) {
        Statsig.shutdown();
        isInitialized = false;
    }
}

export class Telemetry {
    private static instance: Telemetry | null = null;

    static getInstance(): Telemetry {
        if (!Telemetry.instance) {
            Telemetry.instance = new Telemetry();
        }
        return Telemetry.instance;
    }

    add(durationSeconds: number, properties: Record<string, any> = {}): void {
        track("tengu_wait_time", {
            duration_seconds: durationSeconds,
            ...properties
        });
    }
}

export { Log } from '../statsig/StatsigService.js';
