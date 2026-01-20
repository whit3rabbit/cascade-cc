
import * as fs from "fs";
import * as path from "path";
import { PluginManifestSchema } from "../marketplace/MarketplaceSchemas.js";
import { logEvent } from "../telemetry/TelemetryService.js";

// Logic from chunk_363.ts (u51, V82, X82)

export function readPluginManifest(manifestPath: string, pluginName: string, sourceDesc: any): any {
    if (!fs.existsSync(manifestPath)) {
        return {
            name: pluginName,
            description: `Plugin from ${sourceDesc}`
        };
    }

    try {
        const content = fs.readFileSync(manifestPath, "utf-8");
        const json = JSON.parse(content);
        const result = PluginManifestSchema.safeParse(json);
        if (result.success) return result.data;

        const validationError: any = result.error;
        const errors = validationError.errors.map((e: any) => `${e.path.join(".")}: ${e.message}`).join(", ");
        throw new Error(`Validation errors: ${errors}`);
    } catch (error: any) {
        throw new Error(`Invalid manifest for ${pluginName} at ${manifestPath}: ${error.message}`);
    }
}

export function loadPluginComponents(pluginPath: string, source: any, enabled: boolean) {
    const errors: any[] = [];
    const manifestPath = path.join(pluginPath, ".claude-plugin", "plugin.json");

    // Try legacy location
    const legacyManifestPath = path.join(pluginPath, "plugin.json");
    const finalManifestPath = fs.existsSync(manifestPath) ? manifestPath : legacyManifestPath;

    let manifest: any;
    try {
        manifest = readPluginManifest(finalManifestPath, path.basename(pluginPath), source);
    } catch (e: any) {
        errors.push({
            type: "manifest-error",
            source,
            plugin: path.basename(pluginPath),
            error: e.message
        });
        // Create stub manifest to proceed with caution
        manifest = {
            name: path.basename(pluginPath),
            description: "Failed to load manifest"
        };
    }

    const plugin: any = {
        name: manifest.name,
        manifest,
        path: pluginPath,
        source,
        repository: source, // simplified
        enabled
    };

    // Check 'commands'
    const commandsDir = path.join(pluginPath, "commands");
    if (!manifest.commands && fs.existsSync(commandsDir)) {
        plugin.commandsPath = commandsDir;
    }

    if (manifest.commands) {
        const commands = Array.isArray(manifest.commands) ? manifest.commands : [manifest.commands];
        // Simple handling of list of strings (paths) or objects
        // The original code handles objects with 'source' or 'content' keys
        // Simplified here for brevity, assuming mostly directory structure or file paths
        const resolvedPaths: string[] = [];

        const commandItems = Array.isArray(manifest.commands) ? manifest.commands : typeof manifest.commands === 'object' ? Object.values(manifest.commands) : [manifest.commands];

        for (const cmd of commandItems) {
            if (typeof cmd === 'string') {
                const p = path.join(pluginPath, cmd);
                if (fs.existsSync(p)) resolvedPaths.push(p);
                else errors.push({ type: "path-not-found", component: "commands", path: p });
            } else if (typeof cmd === 'object' && cmd && 'source' in cmd) {
                const p = path.join(pluginPath, (cmd as any).source);
                if (fs.existsSync(p)) resolvedPaths.push(p);
            }
        }
        if (resolvedPaths.length > 0) plugin.commandsPaths = resolvedPaths;
    }

    // Check 'agents'
    const agentsDir = path.join(pluginPath, "agents");
    if (!manifest.agents && fs.existsSync(agentsDir)) plugin.agentsPath = agentsDir;
    if (manifest.agents) {
        const agents = Array.isArray(manifest.agents) ? manifest.agents : [manifest.agents];
        const resolved: string[] = [];
        for (const a of agents) {
            if (typeof a === 'string') {
                const p = path.join(pluginPath, a);
                if (fs.existsSync(p)) resolved.push(p);
            }
        }
        if (resolved.length > 0) plugin.agentsPaths = resolved;
    }

    // Check 'skills'
    const skillsDir = path.join(pluginPath, "skills");
    if (!manifest.skills && fs.existsSync(skillsDir)) plugin.skillsPath = skillsDir;
    if (manifest.skills) {
        const skills = Array.isArray(manifest.skills) ? manifest.skills : [manifest.skills];
        const resolved: string[] = [];
        for (const s of skills) {
            if (typeof s === 'string') {
                const p = path.join(pluginPath, s);
                if (fs.existsSync(p)) resolved.push(p);
            }
        }
        if (resolved.length > 0) plugin.skillsPaths = resolved;
    }

    // Check 'output-styles'
    const stylesDir = path.join(pluginPath, "output-styles");
    if (!manifest.outputStyles && fs.existsSync(stylesDir)) plugin.outputStylesPath = stylesDir;
    // ... logic for manifest.outputStyles (omitted for brevity)

    // Check 'hooks'
    // Stubbbed for now, simple check
    const hooksFile = path.join(pluginPath, "hooks", "hooks.json");
    if (fs.existsSync(hooksFile)) {
        try {
            const content = fs.readFileSync(hooksFile, 'utf-8');
            // Validate and attach
            const hooks = JSON.parse(content);
            plugin.hooksConfig = hooks;
        } catch (e) {
            // error
        }
    }

    return { plugin, errors };
}
