/**
 * File: src/services/terminal/AppInitializer.ts
 * Role: Orchestrates the initialization of all services, commands, and tools.
 */

import { initializeCommandDefinitions } from './CommandRegistry.js';
import { initializeSettings } from '../config/SettingsService.js';
import { initializeTelemetry } from '../telemetry/OtelInit.js';
import { initializeLogging } from '../logging/LogManager.js';
import { terminalLog } from '../../utils/shared/runtime.js';

/**
 * Main entry point for initializing the Claude Code environment.
 */
export async function initializeApp(): Promise<void> {
    terminalLog("Initializing Claude Code...");

    try {
        // 1. Settings first
        await initializeSettings();

        // 2. Observability
        await initializeLogging();
        await initializeTelemetry();

        // 3. Command & Tool Registry
        await initializeCommandDefinitions();

        // 4. Integrations (MCP, etc.)
        // await initializeMcpServers();

        terminalLog("Initialization complete.");
    } catch (error) {
        console.error("Critical failure during initialization:", error);
        process.exit(1);
    }
}
