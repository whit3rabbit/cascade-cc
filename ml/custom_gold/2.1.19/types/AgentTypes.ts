/**
 * File: src/types/AgentTypes.ts
 * Role: Type definitions and constants for the Agent system.
 */

// Agent Mode definitions
export type AgentMode = 'prompt' | 'bash' | 'message-selector' | 'help' | 'ask';

// Agent Metadata Interface
export interface AgentMetadata {
    name: string;
    description: string;
    agentType: string;
    tools?: string[];
    model?: string;
    color?: string;
    content?: string;
}

// Agent Message Interface
export type AgentMessageSubtype =
    | 'init'
    | 'informational'
    | 'stop_hook_summary'
    | 'turn_duration'
    | 'local_command'
    | 'compact_boundary'
    | 'microcompact_boundary'
    | 'api_error'
    | 'progress'
    | 'plan_approval'
    | 'shutdown';

export interface AgentMessage {
    type: 'user' | 'assistant' | 'system' | 'progress' | 'tool_result' | 'tool_use';
    role?: 'user' | 'assistant' | 'system';
    content?: string | any;
    message?: string | any;
    uuid?: string;
    timestamp?: number | string;
    subtype?: AgentMessageSubtype | string;
    isMeta?: boolean;
    level?: 'info' | 'warn' | 'error' | 'debug';
    [key: string]: any;
}

// Default Agent Model Constant
export const DEFAULT_AGENT_MODEL = "claude-3-5-sonnet-20241022";

// SendMessage Message Schema Types
export type SendMessageContentType = 'message' | 'broadcast' | 'request' | 'response';
export type SendMessageSubtype = 'shutdown' | 'plan_approval';

export interface SendMessagePayload {
    contentType: SendMessageContentType;
    recipient?: string;
    messageContent?: string;
    subtype?: SendMessageSubtype;
    requestId?: string;
    approve?: boolean;
}
