/**
 * File: src/services/plugin/AgentLoader.ts
 * Role: Discovers and loads agent definitions from Markdown files in plugin directories.
 */

import { readFileSync, readdirSync } from 'node:fs';
import { join, basename } from 'node:path';
import matter from 'gray-matter';

export interface AgentDefinition {
    name: string;
    plugin: string;
    type: string;
    systemPrompt: string;
    source: 'plugin';
    description?: string;
    tools?: string[];
    model?: string;
    userInvocable?: boolean;
    // Add other fields as needed based on SKILL.md structure
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
 */
function loadAgentFromFile(filePath: string, pluginName: string): AgentDefinition | null {
    try {
        const content = readFileSync(filePath, 'utf-8');
        const parsed = matter(content);
        const agentName = basename(filePath).replace('.md', '');
        const data = parsed.data || {};

        return {
            name: data.name || agentName,
            plugin: pluginName,
            type: `${pluginName}:${agentName}`,
            systemPrompt: parsed.content.trim(),
            source: 'plugin',
            description: data.description,
            tools: data.tools ? (Array.isArray(data.tools) ? data.tools : data.tools.split(',').map((t: string) => t.trim())) : [],
            model: data.model,
            userInvocable: data['user-invocable'] !== false // Default to true if not specified
        };
    } catch (err: any) {
        console.error(`[AgentLoader] Failed to load agent from ${filePath}: ${err.message}`);
        return null;
    }
}
