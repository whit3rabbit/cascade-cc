export type AgentMode = 'prompt' | 'bash' | 'message-selector' | 'help' | 'ask';

export interface AgentMessage {
    type: 'user' | 'assistant' | 'system' | 'progress';
    message: {
        content: string | any[];
        type?: string;
    };
    uuid?: string;
}
