/**
 * File: src/services/statsig/StatsigMetadataProvider.ts
 * Role: Provides environment and session metadata for Statsig events.
 */

import { randomUUID } from 'node:crypto';
import { BUILD_INFO } from '../../constants/build.js';

export interface StatsigMetadata {
    sdkType: string;
    sdkVersion: string;
    sessionID: string;
    language: string;
    platform: string;
    arch: string;
}

let sessionID = randomUUID();
let sessionStartTime = Date.now();

/**
 * Returns the current Statsig session metadata.
 */
export function getStatsigMetadata(): StatsigMetadata {
    return {
        sdkType: "claude-code-node",
        sdkVersion: BUILD_INFO.VERSION,
        sessionID,
        language: "en-US",
        platform: process.platform,
        arch: process.arch
    };
}

/**
 * Rotates the session ID (e.g., after long inactivity).
 */
export function rotateSession(): void {
    sessionID = randomUUID();
    sessionStartTime = Date.now();
}

/**
 * Returns the session duration in seconds.
 */
export function getSessionDuration(): number {
    return Math.floor((Date.now() - sessionStartTime) / 1000);
}
