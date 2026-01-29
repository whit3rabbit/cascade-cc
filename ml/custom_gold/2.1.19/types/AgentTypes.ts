/**
 * File: src/types/AgentTypes.ts
 * Role: Type definitions and constants for the Agent system.
 */

export type AgentMode = 'prompt' | 'bash' | 'message-selector' | 'help' | 'ask';

export interface AgentMetadata {
    name: string;
    description: string;
    agentType: string;
    tools?: string[];
    model?: string;
    color?: string;
    content: string;
}

export interface AgentMessage {
    type: 'user' | 'assistant' | 'system' | 'progress' | 'tool_result' | 'tool_use';
    message: string | any;
    uuid?: string;
    timestamp?: number;
    [key: string]: any;
}

export const DEFAULT_AGENT_MODEL = "claude-3-5-sonnet-20241022";
