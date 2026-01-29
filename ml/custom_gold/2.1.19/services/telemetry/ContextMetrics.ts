/**
 * File: src/services/telemetry/ContextMetrics.ts
 * Role: Tracks and logs metrics related to conversation context size.
 */

import { createHash } from "node:crypto";
import { track } from './Telemetry.js';

export interface ContextMetricsData {
    tools?: any[];
    gitStatus?: string;
    claudeMd?: string;
}

export interface SystemPromptBlock {
    text: string;
}

/**
 * Logs context size information for telemetry.
 */
export async function logContextSize({ tools, gitStatus, claudeMd }: ContextMetricsData): Promise<void> {
    const gitStatusSize = gitStatus?.length ?? 0;
    const claudeMdSize = claudeMd?.length ?? 0;

    track("tengu_context_size", {
        git_status_size: gitStatusSize,
        claude_md_size: claudeMdSize,
        total_context_size: gitStatusSize + claudeMdSize,
        mcp_tools_count: tools?.length ?? 0,
    });
}

/**
 * Logs hash and length of system prompt blocks to track updates.
 */
export function logSystemPromptBlock(prompt: SystemPromptBlock): void {
    if (!prompt || !prompt.text) return;

    track("tengu_sysprompt_block", {
        snippet: prompt.text.slice(0, 20),
        length: prompt.text.length,
        hash: createHash("sha256").update(prompt.text).digest("hex")
    });
}
