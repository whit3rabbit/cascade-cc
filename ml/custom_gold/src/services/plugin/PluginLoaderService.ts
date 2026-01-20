
import { getEnabledPlugins } from "./PluginManager.js";
import { loadPluginCommands, loadPluginSkills, loadPluginAgents } from "./PluginCommandLoader.js";
import { cleanupOrphanedPlugins } from "./PluginCleanupService.js";
import { logEvent } from "../telemetry/TelemetryService.js";

// Logic from chunk_362.ts (F2A, sG0, E2A, Vo, eG0) wrappers

export class PluginLoaderService {
    private static commandsCache: any[] | null = null;
    private static skillsCache: any[] | null = null;
    private static agentsCache: any[] | null = null;
    private static hooksCache: any | null = null;
    private static outputStylesCache: any[] | null = null;

    static clearCaches() {
        this.commandsCache = null;
        this.skillsCache = null;
        this.agentsCache = null;
        this.hooksCache = null;
        this.outputStylesCache = null;
    }

    static async getPluginCommands() {
        if (this.commandsCache) return this.commandsCache;

        const { enabled, errors } = await getEnabledPlugins();
        if (errors.length > 0) {
            console.error("Plugin loading errors:", errors);
        }

        const commands: any[] = [];
        for (const plugin of enabled) {
            try {
                // Ignore ignored paths logic if complicated, simplified here
                const pluginCmds = await loadPluginCommands(plugin);
                commands.push(...pluginCmds);
            } catch (e) {
                console.error(`Failed to load commands from plugin ${plugin.name}`, e);
            }
        }

        this.commandsCache = commands;
        return this.commandsCache;
    }

    static async getPluginSkills() {
        if (this.skillsCache) return this.skillsCache;

        const { enabled } = await getEnabledPlugins();
        const skills: any[] = [];

        for (const plugin of enabled) {
            try {
                const pluginSkills = await loadPluginSkills(plugin);
                skills.push(...pluginSkills);
            } catch (e) {
                console.error(`Failed to load skills from plugin ${plugin.name}`, e);
            }
        }

        this.skillsCache = skills;
        return this.skillsCache;
    }

    static async getPluginAgents() {
        if (this.agentsCache) return this.agentsCache;

        const { enabled } = await getEnabledPlugins();
        const agents: any[] = [];

        for (const plugin of enabled) {
            try {
                const pluginAgents = await loadPluginAgents(plugin);
                agents.push(...pluginAgents);
            } catch (e) {
                console.error(`Failed to load agents from plugin ${plugin.name}`, e);
            }
        }

        this.agentsCache = agents;
        return this.agentsCache;
    }

    static async getPluginHooks() {
        if (this.hooksCache) return this.hooksCache;

        const { enabled } = await getEnabledPlugins();
        // hooksConfig is already loaded by loadPluginComponents in PluginManifestLoader

        // Aggregate hooks by event type
        const aggregatedHooks: any = {
            PreToolUse: [],
            PostToolUse: [],
            PostToolUseFailure: [],
            Notification: [],
            UserPromptSubmit: [],
            SessionStart: [],
            SessionEnd: [],
            Stop: [],
            SubagentStart: [],
            SubagentStop: [],
            PreCompact: [],
            PermissionRequest: []
        };

        for (const plugin of enabled) {
            if (plugin.hooksConfig) {
                for (const [eventType, hookEntries] of Object.entries(plugin.hooksConfig)) {
                    if (aggregatedHooks[eventType] && Array.isArray(hookEntries)) {
                        // Transform hook entries to executable/runtime format if needed
                        // For now pushing raw config
                        aggregatedHooks[eventType].push(...(hookEntries as any[]).map(h => ({ ...h, pluginName: plugin.name })));
                    }
                }
            }
        }

        this.hooksCache = aggregatedHooks;
        return this.hooksCache;
    }

    static async getPluginOutputStyles() {
        if (this.outputStylesCache) return this.outputStylesCache;

        // Logic from eG0 (chunk_362)
        // Similar to commands but for output styles
        // Stub for now as less critical
        this.outputStylesCache = [];
        return this.outputStylesCache;
    }

    static async refreshPluginSystem() {
        this.clearCaches();
        await cleanupOrphanedPlugins();
    }
}
