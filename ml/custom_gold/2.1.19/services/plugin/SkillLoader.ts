import { readdirSync, existsSync } from 'node:fs';
import { join, relative } from 'node:path';
import { homedir } from 'node:os';
import { getProjectRoot } from '../../utils/fs/paths.js';
import { mcpClientManager } from '../mcp/McpClientManager.js';

export interface SkillCommand {
    name: string;
    source: 'local_skill' | 'plugin_skill';
    path: string;
    description?: string;
    serverId?: string;
}

/**
 * Recursively finds skill files in a directory.
 */
function discoverSkillsRecursively(baseDir: string, currentDir: string): SkillCommand[] {
    const commands: SkillCommand[] = [];
    if (!existsSync(currentDir)) {
        return commands;
    }

    try {
        const items = readdirSync(currentDir, { withFileTypes: true });
        for (const item of items) {
            const itemPath = join(currentDir, item.name);
            if (item.isDirectory()) {
                commands.push(...discoverSkillsRecursively(baseDir, itemPath));
            } else if (item.isFile() && (item.name.endsWith('.js') || item.name.endsWith('.md'))) {
                // For skills in subdirectories, use the relative path as name (e.g., "nested/myskill")
                const relativePath = relative(baseDir, itemPath);
                const skillName = relativePath.replace(/\.(js|md)$/, '');

                commands.push({
                    name: skillName,
                    source: 'local_skill',
                    path: itemPath
                });
            }
        }
    } catch (error) {
        console.error(`[SkillLoader] Failed to discover skills in ${currentDir}:`, error);
    }
    return commands;
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
        if (existsSync(dir)) {
            commands.push(...discoverSkillsRecursively(dir, dir));
        }
    }
    return commands;
}

/**
 * Loads skills provided by MCP plugins or other external sources.
 * Integrates with McpClientManager to retrieve tools/capabilities.
 */
export async function getPluginSkills(): Promise<SkillCommand[]> {
    try {
        const tools = await mcpClientManager.getTools();
        return tools.map(tool => ({
            name: tool.name,
            source: 'plugin_skill',
            path: `mcp://${tool.serverId}/${tool.name}`,
            description: tool.description,
            serverId: tool.serverId
        }));
    } catch (error) {
        console.error(`[SkillLoader] Failed to load plugin skills:`, error);
        return [];
    }
}
