/**
 * File: src/services/telemetry/Telemetry.ts
 * Role: Telemetry Service Interface
 */

import { join } from 'node:path';
import { homedir } from 'node:os';
import { appendFile, mkdir, readFile, unlink, readdir } from 'node:fs/promises';
import { Log } from '../statsig/StatsigService.js';
import { request } from 'undici';

// Internal telemetry queue or buffer
interface TelemetryEvent {
    name: string;
    properties: Record<string, any>;
    timestamp: number;
}

const telemetryBuffer: TelemetryEvent[] = [];
let isInitialized = false;
let flushInterval: any = null;

import { EnvService } from '../config/EnvService.js';

const TELEMETRY_DIR = join(EnvService.get('CLAUDE_CONFIG_DIR'), 'telemetry');
const TELEMETRY_FILE = join(TELEMETRY_DIR, 'events.jsonl');

async function flushEvents() {
    if (telemetryBuffer.length === 0) return;

    const eventsToWrite = [...telemetryBuffer];
    telemetryBuffer.length = 0; // Clear buffer immediately

    try {
        await mkdir(TELEMETRY_DIR, { recursive: true });
        const data = eventsToWrite.map(e => JSON.stringify(e)).join('\n') + '\n';
        await appendFile(TELEMETRY_FILE, data, 'utf8');

        if (EnvService.isTruthy("DEBUG_TELEMETRY")) {
            console.log(`[Telemetry] Flushed ${eventsToWrite.length} events to disk`);
        }

        // Attempt to upload everything from disk
        await uploadEvents();
    } catch (error) {
        console.error('[Telemetry] Failed to flush events:', error);
    }
}

/**
 * Synchronously flushes events to disk. Safe for use in exit handlers.
 */
export function flushSync() {
    if (telemetryBuffer.length === 0) return;

    try {
        const { appendFileSync, mkdirSync } = require('node:fs');
        const eventsToWrite = [...telemetryBuffer];
        telemetryBuffer.length = 0;

        mkdirSync(TELEMETRY_DIR, { recursive: true });
        const data = eventsToWrite.map(e => JSON.stringify(e)).join('\n') + '\n';
        appendFileSync(TELEMETRY_FILE, data, 'utf8');

        if (EnvService.isTruthy("DEBUG_TELEMETRY")) {
            console.log(`[Telemetry] Synchronously flushed ${eventsToWrite.length} events`);
        }
    } catch (error) {
        console.error('[Telemetry] Failed to sync flush:', error);
    }
}

const TELEMETRY_ENDPOINT = EnvService.get("CLAUDE_TELEMETRY_URL") || "https://statsigapi.net/v1/log_event";

async function uploadEvents() {
    try {
        const data = await readFile(TELEMETRY_FILE, 'utf8');
        if (!data.trim()) return;

        const events = data.trim().split('\n').map(l => JSON.parse(l));

        const response = await request(TELEMETRY_ENDPOINT, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'STATSIG-API-KEY': EnvService.get("STATSIG_CLIENT_KEY") || "client-claude-key"
            },
            body: JSON.stringify({
                events,
                statsigMetadata: {
                    sdkType: 'node-lite',
                    sdkVersion: '1.0.0'
                }
            })
        });

        if (response.statusCode >= 200 && response.statusCode < 300) {
            await unlink(TELEMETRY_FILE);
            if (EnvService.isTruthy("DEBUG_TELEMETRY")) {
                console.log(`[Telemetry] Successfully uploaded ${events.length} events`);
            }
        } else {
            if (EnvService.isTruthy("DEBUG_TELEMETRY")) {
                console.warn(`[Telemetry] Failed to upload events. Status: ${response.statusCode}`);
            }
        }
    } catch (error) {
        if (EnvService.isTruthy("DEBUG_TELEMETRY")) {
            console.error('[Telemetry] Upload error:', error);
        }
    }
}

export function track(eventName: string, properties: Record<string, any> = {}) {
    const event: TelemetryEvent = {
        name: eventName,
        properties,
        timestamp: Date.now()
    };

    if (EnvService.isTruthy("DEBUG_TELEMETRY")) {
        console.log(`[Telemetry] ${eventName}`, properties);
    }

    telemetryBuffer.push(event);
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

export function initializeTelemetry() {
    if (isInitialized) return;
    isInitialized = true;

    // Start background flusher (every 30s)
    flushInterval = setInterval(flushEvents, 30000);

    Log.info("Telemetry Service Initialized");
}

export async function shutdownTelemetry() {
    if (flushInterval) {
        clearInterval(flushInterval);
        flushInterval = null;
    }
    await flushEvents();
    Log.info(`Telemetry Service Shutdown.`);
}

export { Log } from '../statsig/StatsigService.js';
