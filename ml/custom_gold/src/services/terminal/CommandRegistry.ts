
// Logic from chunk_578.ts (Command & Skill Registry)

import { getSkillDirectoryCommands } from "./SkillDirectoryService.js"; // Mocking import
import { PluginLoaderService } from "../plugin/PluginLoaderService.js";

// --- Skills Loader (xY7) ---
export async function loadSkills(path?: string) {
    try {
        const [skillDirCommands, pluginSkills] = await Promise.all([
            getSkillDirectoryCommands(path).catch((err: any) => {
                console.error("Failed to load skill directory commands:", err);
                return [];
            }),
            PluginLoaderService.getPluginSkills().catch((err: any) => {
                console.error("Failed to load plugin skills:", err);
                return [];
            })
        ]);

        return {
            skillDirCommands,
            pluginSkills
        };
    } catch (err: any) {
        console.error("Unexpected error loading skills:", err);
        return {
            skillDirCommands: [],
            pluginSkills: []
        };
    }
}

// --- Command Finder (wS) ---
export function findCommand(name: string, commands: any[]) {
    const command = commands.find(c =>
        c.name === name ||
        c.userFacingName?.() === name ||
        c.aliases?.includes(name)
    );

    if (!command) {
        const available = commands.map(c => {
            const uName = c.userFacingName?.() || c.name;
            return c.aliases?.length ? `${uName} (aliases: ${c.aliases.sort().join(", ")})` : uName;
        }).sort().join(", ");

        throw new ReferenceError(`Command ${name} not found. Available commands: ${available}`);
    }

    return command;
}

// --- System Prompt Fragments (vY7, kY7) ---

/**
 * Generates reasoning effort instructions based on user settings.
 */
export function getReasoningEffortFragment(level: string | undefined): string {
    if (!level) return "";

    // Map human friendly levels to numeric values if needed
    const effortMap: Record<string, number> = {
        low: 10,
        medium: 50,
        high: 100
    };

    const value = effortMap[level.toLowerCase()] ?? level;

    return `
<reasoning_effort>${value}</reasoning_effort>

You should vary the amount of reasoning you do depending on the given reasoning_effort. reasoning_effort varies between 0 and 100. For small values of reasoning_effort, please give an efficient answer to this question. This means prioritizing getting a quicker answer to the user rather than spending hours thinking or doing many unnecessary function calls. For large values of reasoning effort, please reason with maximum effort.`;
}

/**
 * Generates allowed tools list for the system prompt.
 */
export function getAllowedToolsFragment(allowedTools: any[]): string {
    if (!allowedTools || allowedTools.length === 0) return "";

    return `
You can use the following tools without requiring user approval: ${allowedTools.map(t => t.name).join(", ")}
`;
}
