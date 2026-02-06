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
import { mcpClientManager } from '../mcp/McpClientManager.js';
import { initTreeSitter } from '../../utils/shared/treeSitter.js';
import { MailboxPollingService } from '../teams/MailboxPollingService.js';
import { getLeaderPaneId } from '../../utils/terminal/terminalDetection.js';



import { MarketplaceService } from '../marketplace/MarketplaceService.js';

async function initializeMcpServers() {
    try {
        await mcpClientManager.initializeAllServers();
        // Fire and forget marketplace auto-install
        MarketplaceService.autoInstallOfficialMarketplace().catch(err =>
            console.error("Marketplace auto-install failed:", err)
        );
    } catch (err) {
        console.error("Failed to initialize MCP servers:", err);
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
        // Fire and forget cleanup to not slow down startup, and ensure it fails silently.
        (async () => {
            try {
                const threshold = getCleanupThresholdDate();
                const plansDir = getPlansDirectory();
                await cleanupOldFiles(plansDir, threshold, '.md');
            } catch {
                // Ignore any errors during cleanup
            }
        })().catch(() => {
            // "Fire and forget" - ignore unhandled promise rejections
        });

        // 2. Observability
        await initializeLogging();
        await initializeTelemetry();

        // 2.5 Tree Sitter
        try {
            await initTreeSitter();
        } catch (_e) {
            console.warn("TreeSitter initialization failed, some parsing features may be limited.", _e);
        }

        // 3. Command & Tool Registry
        await initializeCommandDefinitions();

        // 4. Integrations (MCP, etc.)
        // Fire and forget to not block TUI startup
        initializeMcpServers().catch(err => console.error("Error initializing MCP servers:", err));

        // 5. Team Services
        MailboxPollingService.start();

        // 6. Terminal Detection
        try {
            const leaderPane = await getLeaderPaneId();
            if (leaderPane) {
                terminalLog(`Leader Pane ID detected: ${leaderPane}`);
                // Store in EnvService or similar if needed, or it's just available for other services to query dynamically 
                // if they are in the same process/pane. 
                // Note: If we spawn new processes, they inherit env but not this JS state. 
                // Ideally we'd set an env var for children.
                process.env.CLAUDE_LEADER_PANE_ID = leaderPane;
            }
        } catch {
            // ignore
        }

        terminalLog("Initialization complete.");
        return { isFirstRun };
    } catch (error) {
        console.error("Critical failure during initialization:", error);
        process.exit(1);
    }
}
