/**
 * File: src/services/terminal/AppInitializer.ts
 * Role: Orchestrates the initialization of all services, commands, and tools.
 */

import { initializeCommandDefinitions } from './CommandRegistry.js';
import { initializeSettings, getPlansDirectory } from '../config/SettingsService.js';
import { getCleanupThresholdDate, cleanupOldFiles } from '../../utils/fileCleanup.js';
import { initializeTelemetry } from '../telemetry/OtelInit.js';
import { initializeLogging } from '../logging/LogManager.js';
import { terminalLog } from '../../utils/shared/runtime.js';

/**
 * Main entry point for initializing the Claude Code environment.
 */
import { createShellSnapshot } from './ShellSnapshotService.js';
import { McpServerManager } from '../mcp/McpServerManager.js';
import { mcpClientManager } from '../mcp/McpClientManager.js';
import { initTreeSitter } from '../../utils/shared/treeSitter.js';
import { MailboxPollingService } from '../teams/MailboxPollingService.js';

async function initializeMcpServers() {
    const servers = await McpServerManager.getAllMcpServers();
    for (const [name, config] of Object.entries(servers)) {
        try {
            await mcpClientManager.connect(name, config as any);
        } catch (err) {
            console.error(`Failed to connect to MCP server ${name}:`, err);
        }
    }
}

import { EnvService } from '../config/EnvService.js';

/**
 * Main entry point for initializing the Claude Code environment.
 */
export async function initializeApp(): Promise<{ isFirstRun: boolean }> {
    terminalLog("Initializing Claude Code...");

    let isFirstRun = false;
    try {
        // 0. Shell Snapshot
        await createShellSnapshot(EnvService.get("SHELL"));

        // 1. Settings first
        const settings = await initializeSettings();
        isFirstRun = !settings.onboardingComplete;

        // 1.5. Cleanup
        try {
            const threshold = getCleanupThresholdDate();
            const plansDir = getPlansDirectory();
            // Fire and forget cleanup to not slow down startup
            cleanupOldFiles(plansDir, threshold, '.md').catch(() => { });
        } catch (e) {
            // ignore
        }

        // 2. Observability
        await initializeLogging();
        await initializeTelemetry();

        // 2.5 Tree Sitter
        try {
            await initTreeSitter();
        } catch (e) {
            console.warn("TreeSitter initialization failed, some parsing features may be limited.", e);
        }

        // 3. Command & Tool Registry
        await initializeCommandDefinitions();

        // 4. Integrations (MCP, etc.)
        await initializeMcpServers();

        // 5. Team Services
        MailboxPollingService.start();

        terminalLog("Initialization complete.");
        return { isFirstRun };
    } catch (error) {
        console.error("Critical failure during initialization:", error);
        process.exit(1);
    }
}
