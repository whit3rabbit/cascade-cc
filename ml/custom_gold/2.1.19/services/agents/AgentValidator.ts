/**
 * File: src/services/agents/AgentValidator.ts
 * Role: Validates custom agent configurations.
 */

export interface AgentConfig {
    agentType: string;
    whenToUse: string;
    tools?: string[];
    systemPrompt?: string;
    getSystemPrompt?: () => string;
    [key: string]: any;
}

export interface ValidationResult {
    isValid: boolean;
    errors: string[];
    warnings: string[];
}

/**
 * Validates an agent configuration object.
 */
export function validateAgentConfig(agentConfig: AgentConfig, _allTools: any[] = []): ValidationResult {
    const errors: string[] = [];
    const warnings: string[] = [];

    if (!agentConfig.agentType) {
        errors.push("Agent type is required");
    }

    if (!agentConfig.whenToUse) {
        errors.push("Description (whenToUse) is required");
    } else {
        if (agentConfig.whenToUse.length < 10) {
            warnings.push("Description should be more descriptive (at least 10 characters)");
        }
    }

    if (agentConfig.tools !== undefined && !Array.isArray(agentConfig.tools)) {
        errors.push("Tools must be an array");
    }

    const systemPrompt = typeof agentConfig.getSystemPrompt === 'function'
        ? agentConfig.getSystemPrompt()
        : agentConfig.systemPrompt;

    if (!systemPrompt) {
        errors.push("System prompt is required");
    } else if (systemPrompt.length < 20) {
        errors.push("System prompt is too short (minimum 20 characters)");
    }

    return {
        isValid: errors.length === 0,
        errors,
        warnings
    };
}
