/**
 * File: src/services/mcp/McpContentProcessor.ts
 * Role: Orchestrates content processing and hooks.
 */

// import { track } from '../telemetry/Telemetry.js'; // Not used in current deob logic

type PostSamplingHook = (context: any) => Promise<void>;

let postSamplingHooks: PostSamplingHook[] = [];

/**
 * Executes post-sampling hooks.
 */
export async function executePostSamplingHooks(context: any): Promise<void> {
    for (const hook of postSamplingHooks) {
        try {
            await hook(context);
        } catch (error) {
            console.error(`Post-sampling hook failed: ${error}`);
        }
    }
}

/**
 * Registers a new hook to be executed after sampling.
 */
export function registerPostSamplingHook(hook: PostSamplingHook): void {
    postSamplingHooks.push(hook);
}

// Re-export specific processors for convenience
export { transformToolInput, prepareToolForConversation } from './ToolProcessor.js';
export { logContextSize, logSystemPromptBlock } from '../telemetry/ContextMetrics.js';
