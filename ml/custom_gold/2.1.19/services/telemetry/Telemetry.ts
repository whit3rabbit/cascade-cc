/**
 * File: src/services/telemetry/Telemetry.ts
 * Role: Telemetry Service Interface
 */

import { Log } from '../statsig/StatsigService.js';

// Internal telemetry queue or buffer
interface TelemetryEvent {
    name: string;
    properties: Record<string, any>;
    timestamp: number;
}

const telemetryBuffer: TelemetryEvent[] = [];
let isInitialized = false;

export function track(eventName: string, properties: Record<string, any> = {}) {
    const event: TelemetryEvent = {
        name: eventName,
        properties,
        timestamp: Date.now()
    };

    // In a real implementation, this would send data to a backend (Statsig, Segment, etc.)
    // For now, we log it or buffer it.
    if (process.env.DEBUG_TELEMETRY) {
        console.log(`[Telemetry] ${eventName}`, properties);
    }

    telemetryBuffer.push(event);
}

export function initializeTelemetry() {
    if (isInitialized) return;
    isInitialized = true;

    // Flush buffer or start background sender
    Log.info("Telemetry Service Initialized");
}

export function shutdownTelemetry() {
    // Flush remaining events
    const count = telemetryBuffer.length;
    telemetryBuffer.length = 0; // Clear
    Log.info(`Telemetry Service Shutdown. Flushed ${count} events.`);
}

// Re-export for convenience if needed, though direct import is preferred
export { Log } from '../statsig/StatsigService.js';
