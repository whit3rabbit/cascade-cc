/**
 * File: src/services/plugin/SkillLoader.ts
 * Role: Discovers and loads "Skill" commands (agentic capabilities) from local and plugin directories.
 */

import { readdirSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';
import { getProjectRoot } from '../../utils/fs/paths.js';

export interface SkillCommand {
    name: string;
    source: 'local_skill' | 'plugin_skill';
    path: string;
}

/**
 * Loads skill commands from the .claude/skills directory in the project or home dir.
 */
export async function getSkillDirectoryCommands(): Promise<SkillCommand[]> {
    const locations = [
        join(getProjectRoot(), '.claude', 'skills'),
        join(homedir(), '.claude', 'skills')
    ];

    const commands: SkillCommand[] = [];
    for (const dir of locations) {
        if (!existsSync(dir)) {
            continue;
        }

        try {
            const files = readdirSync(dir);
            for (const file of files) {
                if (file.endsWith('.js') || file.endsWith('.md')) {
                    commands.push({
                        name: file.replace(/\.(js|md)$/, ''),
                        source: 'local_skill',
                        path: join(dir, file)
                    });
                }
            }
        } catch (error) {
            console.error(`[SkillLoader] Failed to read skills from ${dir}:`, error);
        }
    }
    return commands;
}

/**
 * Loads skills provided by MCP plugins or other external sources.
 * 
 * TODO: Interface with McpServerManager to retrieve plugin-provided skills.
 */
export async function getPluginSkills(): Promise<SkillCommand[]> {
    // Placeholder for plugin skill loading logic
    return [];
}
