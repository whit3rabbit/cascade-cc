/**
 * File: src/services/plugin/AgentLoader.ts
 * Role: Discovers and loads agent definitions from Markdown files in plugin directories.
 */

import { readFileSync, readdirSync } from 'node:fs';
import { join, basename } from 'node:path';

export interface AgentDefinition {
    name: string;
    plugin: string;
    type: string;
    systemPrompt: string;
    source: 'plugin';
}

/**
 * Loads agents from a given directory recursively.
 */
export function loadAgentsFromDirectory(directoryPath: string, pluginName: string): AgentDefinition[] {
    const agents: AgentDefinition[] = [];
    try {
        const entries = readdirSync(directoryPath, { withFileTypes: true });
        for (const entry of entries) {
            const fullPath = join(directoryPath, entry.name);
            if (entry.isDirectory()) {
                agents.push(...loadAgentsFromDirectory(fullPath, pluginName));
            } else if (entry.isFile() && entry.name.endsWith('.md')) {
                const agent = loadAgentFromFile(fullPath, pluginName);
                if (agent) {
                    agents.push(agent);
                }
            }
        }
    } catch (err: any) {
        console.error(`[AgentLoader] Failed to scan ${directoryPath}: ${err.message}`);
    }
    return agents;
}

/**
 * Parses an individual Markdown file into an agent definition.
 * 
 * TODO: Implement real frontmatter parsing if needed.
 */
function loadAgentFromFile(filePath: string, pluginName: string): AgentDefinition | null {
    try {
        const content = readFileSync(filePath, 'utf-8');
        const agentName = basename(filePath).replace('.md', '');

        return {
            name: agentName,
            plugin: pluginName,
            type: `${pluginName}:${agentName}`,
            systemPrompt: content,
            source: 'plugin'
        };
    } catch (err: any) {
        console.error(`[AgentLoader] Failed to load agent from ${filePath}: ${err.message}`);
        return null;
    }
}
