/**
 * File: src/services/agents/AgentPersistence.ts
 * Role: Handles saving, updating, and deleting custom agent markdown files.
 */

import { join, basename } from 'node:path';
import { writeFileSync, unlinkSync, existsSync, mkdirSync, readFileSync, readdirSync, statSync } from 'node:fs';
import { getBaseConfigDir } from '../../utils/shared/runtimeAndEnv.js';
import matter from 'gray-matter';

export interface AgentData {
    name: string;
    description: string;
    agentType: string;
    tools?: string[];
    systemPrompt: string;
    color?: string;
    model?: string;
    scope?: 'user' | 'project' | 'builtin';
    [key: string]: any;
}

/**
 * Formats agent metadata and system instructions into a Markdown file with YAML frontmatter.
 * 
 * @param {AgentData} agent - The agent configuration.
 * @returns {string} The formatted Markdown content.
 */
export function formatAgentMarkdown({ name, description, tools = [], systemPrompt, color, model }: AgentData): string {
    const escapedDesc = description.replace(/"/g, '\\"');
    const frontmatter = [
        "---",
        `name: ${name}`,
        `description: "${escapedDesc}"`,
        tools.length ? `tools: ${tools.join(", ")}` : null,
        model ? `model: ${model}` : null,
        color ? `color: ${color}` : null,
        "---",
        "",
        systemPrompt
    ].filter(Boolean).join("\n");
    return frontmatter;
}

/**
 * Loads and parses a single agent file.
 */
export function loadAgent(filePath: string, scope: 'user' | 'project'): AgentData | null {
    try {
        if (!existsSync(filePath)) return null;

        const content = readFileSync(filePath, 'utf8');
        const parsed = matter(content);
        const data = parsed.data as any;

        // Agent Type is derived from filename
        const agentType = basename(filePath, '.md');

        return {
            name: data.name || agentType,
            description: data.description || '',
            agentType,
            tools: data.tools ? (typeof data.tools === 'string' ? data.tools.split(',').map((t: string) => t.trim()) : data.tools) : [],
            systemPrompt: parsed.content.trim(),
            color: data.color,
            model: data.model,
            scope,
            ...data
        };
    } catch (error) {
        console.error(`Failed to load agent from ${filePath}:`, error);
        return null;
    }
}

/**
 * Lists all available custom agents (User and Project).
 */
export function listAgents(): AgentData[] {
    const agents: AgentData[] = [];

    // Project Agents
    const projectAgentsDir = join(process.cwd(), '.claude', 'agents');
    if (existsSync(projectAgentsDir)) {
        try {
            const files = readdirSync(projectAgentsDir).filter(f => f.endsWith('.md'));
            for (const file of files) {
                const agent = loadAgent(join(projectAgentsDir, file), 'project');
                if (agent) agents.push(agent);
            }
        } catch (err) {
            // Ignore directory read errors
        }
    }

    // User Agents
    const userAgentsDir = join(getBaseConfigDir(), 'agents');
    if (existsSync(userAgentsDir)) {
        try {
            const files = readdirSync(userAgentsDir).filter(f => f.endsWith('.md'));
            for (const file of files) {
                const agent = loadAgent(join(userAgentsDir, file), 'user');
                if (agent) agents.push(agent);
            }
        } catch (err) {
            // Ignore directory read errors
        }
    }

    return agents;
}

/**
 * Saves a custom agent to the configuration directory.
 * @param {AgentData} agent - The agent configuration.
 * @param {'user'|'project'} scope - The scope to save the agent to.
 */
export async function saveAgent(agent: AgentData, scope: 'user' | 'project' = 'user'): Promise<string> {
    let agentsDir: string;

    if (scope === 'project') {
        agentsDir = join(process.cwd(), '.claude', 'agents');
    } else {
        agentsDir = join(getBaseConfigDir(), 'agents');
    }

    if (!existsSync(agentsDir)) mkdirSync(agentsDir, { recursive: true });

    const filePath = join(agentsDir, `${agent.agentType}.md`);
    const content = formatAgentMarkdown(agent);

    writeFileSync(filePath, content, 'utf8');
    return filePath;
}

/**
 * Deletes a custom agent file.
 */
export async function deleteAgent(agentType: string, scope: 'user' | 'project' = 'user'): Promise<boolean> {
    let filePath: string;

    if (scope === 'project') {
        filePath = join(process.cwd(), '.claude', 'agents', `${agentType}.md`);
    } else {
        filePath = join(getBaseConfigDir(), 'agents', `${agentType}.md`);
    }

    if (existsSync(filePath)) {
        unlinkSync(filePath);
        return true;
    }
    return false;
}
