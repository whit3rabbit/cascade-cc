/**
 * File: src/services/agents/AgentPersistence.ts
 * Role: Handles saving, updating, and deleting custom agent markdown files.
 */

import { join } from 'node:path';
import { writeFileSync, unlinkSync, existsSync, mkdirSync } from 'node:fs';
import { getBaseConfigDir } from '../../utils/shared/runtimeAndEnv.js';

export interface AgentData {
    name: string;
    description: string;
    agentType: string;
    tools?: string[];
    systemPrompt: string;
    color?: string;
    model?: string;
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
 * Saves a custom agent to the local configuration directory.
 */
export async function saveAgent(agent: AgentData): Promise<string> {
    const agentsDir = join(getBaseConfigDir(), 'agents');
    if (!existsSync(agentsDir)) mkdirSync(agentsDir, { recursive: true });

    const filePath = join(agentsDir, `${agent.agentType}.md`);
    const content = formatAgentMarkdown(agent);

    writeFileSync(filePath, content, 'utf8');
    return filePath;
}

/**
 * Deletes a custom agent file.
 */
export async function deleteAgent(agentType: string): Promise<boolean> {
    const filePath = join(getBaseConfigDir(), 'agents', `${agentType}.md`);
    if (existsSync(filePath)) {
        unlinkSync(filePath);
        return true;
    }
    return false;
}
