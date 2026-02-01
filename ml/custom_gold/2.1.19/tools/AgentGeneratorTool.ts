/**
 * File: src/tools/AgentGeneratorTool.ts
 * Role: Allows Claude to generate and save a new custom agent with specific instructions and tools.
 */

import { saveAgent, AgentData } from '../services/agents/AgentPersistence.js';
import { validateAgentConfig } from '../services/agents/AgentValidator.js';

export interface AgentGeneratorInput {
    name: string;
    description: string;
    systemPrompt: string;
    tools?: string[];
    model?: string;
    color?: string;
    scope?: 'user' | 'project';
}

export const AgentGeneratorTool = {
    name: "AgentGenerator",
    description: "Generates and saves a new custom agent. Use this when the user asks to create a specialized agent.",
    async call(input: AgentGeneratorInput) {
        const { name, description, systemPrompt, tools = [], model, color, scope = 'user' } = input;

        // Generate a slug-like agentType from the name
        const agentType = name.toLowerCase().replace(/[^a-z0-9]/g, '-').replace(/-+/g, '-');

        const agentData: AgentData = {
            name,
            description,
            agentType,
            systemPrompt,
            tools,
            model,
            color,
            scope
        };

        const validation = validateAgentConfig({
            agentType,
            whenToUse: description,
            systemPrompt,
            tools
        });

        if (!validation.isValid) {
            return {
                is_error: true,
                content: `Failed to generate agent: ${validation.errors.join(", ")}`
            };
        }

        try {
            const filePath = await saveAgent(agentData, scope);
            return {
                is_error: false,
                content: `Successfully generated and saved agent "${name}" to ${filePath}.\n\n` +
                    `The user can now use this agent via \`/agents\` or by specifying it in the session.`
            };
        } catch (error: any) {
            return {
                is_error: true,
                content: `Failed to save generated agent: ${error.message}`
            };
        }
    }
};
