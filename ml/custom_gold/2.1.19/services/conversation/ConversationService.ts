/**
 * File: src/services/conversation/ConversationService.ts
 * Role: High-level orchestrator for LLM interactions and tool execution.
 */

import { PromptManager } from "./PromptManager.js";
import { terminalLog } from "../../utils/shared/runtime.js";
import { ToolExecutionManager } from "../tools/ToolExecutionManager.js";
import { checkToolPermissions } from "../terminal/PermissionService.js";
import { Anthropic } from "../anthropic/AnthropicClient.js";
import { parseSseEvents } from "../../utils/http/SseParser.js";
import { parseBalancedJSON } from "../../utils/json/BalancedParser.js";
import { EnvService } from "../config/EnvService.js";
import { hookService } from "../hooks/HookService.js";
import { ModelResolver } from "./ModelResolver.js";

export interface ConversationOptions {
    commands: any[];
    tools: any[];
    mcpClients: any[];
    cwd: string;
    canUseTool?: (name: string, input: any, context: any) => Promise<any>;
    onPermissionRequest?: (request: any) => Promise<string>;
    planMode?: boolean;
    setPlanMode?: (enabled: boolean) => void;
    verbose?: boolean;
    maxTurns?: number;
    maxBudgetUsd?: number;
    model?: string;
    agent?: string;
    shellSnapshotPath?: string;
}

export interface ConversationResult {
    type: "success" | "error";
    result: string;
    turns: number;
    durationMs: number;
    usage: any;
}

export class ConversationService {
    static async *startConversation(prompt: string, options: ConversationOptions): AsyncGenerator<any> {
        terminalLog(`Starting conversation: "${prompt}"`, "debug");

        const startTime = Date.now();
        const messages: any[] = [];
        let turnCount = 0;

        const systemPrompt = await PromptManager.assembleSystemPrompt(options);
        const resolvedModel = ModelResolver.resolveModel(options.model || 'claude-3-5-sonnet-20241022', !!options.planMode);

        yield {
            type: "system",
            subtype: "init",
            tools: options.tools.map((t: any) => t.name),
            model: resolvedModel,
            session_id: EnvService.get("CLAUDE_SESSION_ID"),
        };

        if (prompt) {
            messages.push({ role: "user", content: prompt });
        }

        const generator = this.conversationLoop(messages, systemPrompt, options);

        for await (const event of generator) {
            yield event;

            if (options.maxTurns && turnCount >= options.maxTurns) {
                yield { type: "result", subtype: "error_max_turns" };
                return;
            }
        }

        yield {
            type: "result",
            subtype: "success",
            duration_ms: Date.now() - startTime,
            num_turns: turnCount,
            result: messages[messages.length - 1]?.content || ""
        };
    }

    private static async *conversationLoop(
        messages: any[],
        systemPrompt: string,
        options: any
    ): AsyncGenerator<any> {
        let keepGoing = true;

        while (keepGoing) {
            const responseStream = this.streamLLM(messages, systemPrompt, options);

            let assistantMessage: any = { role: "assistant", content: [], tool_use: [] as any[] };
            let currentToolUse: any = null;

            for await (const chunk of responseStream) {
                yield { type: "stream_event", event: chunk };

                switch (chunk.type) {
                    case "content_block_start":
                        if (chunk.content_block.type === "text") {
                            assistantMessage.content.push(chunk.content_block);
                        } else if (chunk.content_block.type === "tool_use" || chunk.content_block.type === "server_tool_use") {
                            currentToolUse = { ...chunk.content_block, input_json: "" };
                            assistantMessage.tool_use.push(currentToolUse);
                            assistantMessage.content.push(currentToolUse);
                        }
                        yield { type: "partial_assistant", message: { ...assistantMessage } };
                        break;
                    case "content_block_delta":
                        const block = assistantMessage.content[chunk.index];
                        if (chunk.delta.type === "text_delta") {
                            if (block && block.type === "text") {
                                block.text += chunk.delta.text;
                            }
                        } else if (chunk.delta.type === "input_json_delta") {
                            if (currentToolUse) {
                                currentToolUse.input_json += chunk.delta.partial_json;
                                currentToolUse.input = parseBalancedJSON(currentToolUse.input_json);
                            }
                        }
                        yield { type: "partial_assistant", message: { ...assistantMessage } };
                        break;
                    case "message_delta":
                        break;
                    case "message_stop":
                        break;
                }
            }

            yield { type: "assistant", message: assistantMessage };

            if (assistantMessage.tool_use && assistantMessage.tool_use.length > 0) {
                for (const toolUse of assistantMessage.tool_use) {
                    if (toolUse.input_json) {
                        try {
                            toolUse.input = JSON.parse(toolUse.input_json);
                        } catch (e) {
                            toolUse.input = parseBalancedJSON(toolUse.input_json);
                        }
                        delete toolUse.input_json;
                    }
                }

                const toolResultBlocks: any[] = [];
                const toolResultsGenerator = this.executeTools(assistantMessage.tool_use, options);

                for await (const event of toolResultsGenerator) {
                    if (event.type === 'tool_result') {
                        toolResultBlocks.push(event.message);
                    } else {
                        yield event; // Bubble up cwd_update etc
                    }
                }

                const toolResultMessage = { role: "user", content: toolResultBlocks };
                messages.push(toolResultMessage);
                yield { type: "user", message: toolResultMessage };
                keepGoing = true;
            } else {
                keepGoing = false;
            }
        }
    }

    private static async * streamLLM(messages: any[], system: string, options: any): AsyncGenerator<any> {
        const client = new Anthropic({
            baseUrl: EnvService.get("ANTHROPIC_BASE_URL")
        });

        const anthropicMessages = messages.map(m => ({
            role: m.role,
            content: m.content
        }));

        const stream = await client.messages.create({
            model: ModelResolver.resolveModel(options.model || "claude-3-5-sonnet-20241022", !!options.planMode),
            max_tokens: Number(EnvService.get("CLAUDE_CODE_MAX_OUTPUT_TOKENS")) || 4096,
            system,
            messages: anthropicMessages,
            stream: true,
            tools: options.tools.map((t: any) => ({
                name: t.name,
                description: t.description,
                input_schema: t.input_schema || t.parameters
            }))
        });

        if (!(stream instanceof ReadableStream)) return;

        for await (const event of parseSseEvents(stream)) {
            if (event.type === 'message' || event.data) {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'ping' || data.type === 'error') continue;
                    yield data;
                } catch (e) { }
            }
        }
    }

    private static async *executeTools(toolUses: any[], options: any): AsyncGenerator<any> {
        const { tools, mcpClients, verbose, canUseTool: providedCanUseTool, onPermissionRequest } = options;

        const executeToolFn = async function* (block: any, message: any, canUse: any, context: any) {
            const toolDef = tools.find((t: any) => t.name === block.name);
            if (!toolDef) {
                yield { message: { is_error: true, content: `Tool ${block.name} not found` } };
                return;
            }

            try {
                let perm;
                if (block.name !== 'ToolManager') {
                    perm = await canUse(block.name, block.input, context);
                } else {
                    perm = { behavior: "allow" };
                }

                if (perm.behavior === 'deny') {
                    yield { message: { type: "tool_result", tool_use_id: block.id, content: `Access denied: ${perm.message || "Permission check failed"}`, is_error: true } };
                    return;
                }

                if (perm.behavior === 'ask') {
                    if (onPermissionRequest) {
                        const decision = await onPermissionRequest({ tool: toolDef, input: block.input, message: perm.message });
                        if (decision !== 'allowed') {
                            yield { message: { type: "tool_result", tool_use_id: block.id, content: `User rejected tool execution`, is_error: true } };
                            return;
                        }
                    } else {
                        yield { message: { type: "tool_result", tool_use_id: block.id, content: `Permission required but no UI handler available.`, is_error: true } };
                        return;
                    }
                }

                const result = await toolDef.call(block.input, context);

                if (result.metadata?.cwd && result.metadata.cwd !== context.cwd) {
                    context.cwd = result.metadata.cwd;
                    yield { message: { type: "cwd_update", cwd: result.metadata.cwd } };
                }

                // Hook: PostToolUse
                let finalResult = result;
                try {
                    const hookResults = await hookService.dispatch("PostToolUse", {
                        hook_event_name: "PostToolUse",
                        tool_name: block.name,
                        tool_input: block.input,
                        tool_use_id: block.id,
                        tool_response: result
                    });

                    const update = hookResults
                        .map(r => r.hookSpecificOutput)
                        .find((r): r is NonNullable<typeof r> => !!r && r.hookEventName === "PostToolUse" && 'updatedMCPToolOutput' in r);

                    if (update && update.hookEventName === "PostToolUse" && update.updatedMCPToolOutput !== undefined) {
                        finalResult = update.updatedMCPToolOutput;
                    }
                } catch (hErr) {
                    console.error("Error in PostToolUse hook:", hErr);
                }

                yield { message: { type: "tool_result", tool_use_id: block.id, content: finalResult, is_error: false } };
            } catch (err: any) {
                // Hook: PostToolUseFailure
                try {
                    await hookService.dispatch("PostToolUseFailure", {
                        hook_event_name: "PostToolUseFailure",
                        tool_name: block.name,
                        tool_input: block.input,
                        tool_use_id: block.id,
                        error: err.message
                    });
                } catch (hErr) {
                    console.error("Error in PostToolUseFailure hook:", hErr);
                }

                yield { message: { type: "tool_result", tool_use_id: block.id, content: err.message || String(err), is_error: true } };
            }
        };

        const canUseTool = async (name: string, input: any, context: any) => {
            if (providedCanUseTool) return providedCanUseTool(name, input, context);
            return checkToolPermissions(name, input, { toolPermissionContext: { mode: "prompt" } });
        };

        const manager = new ToolExecutionManager(tools, canUseTool, { ...options }, executeToolFn);
        const assistantMessageStub = { uuid: "stub-uuid" };
        for (const usage of toolUses) {
            manager.addTool({ id: usage.id, name: usage.name, input: usage.input }, assistantMessageStub);
        }

        for await (const res of manager.getRemainingResults()) {
            const msg = res.message;
            if (msg && msg.type === 'cwd_update') {
                yield msg;
            } else {
                yield { type: 'tool_result', message: msg };
            }
        }
    }
}
