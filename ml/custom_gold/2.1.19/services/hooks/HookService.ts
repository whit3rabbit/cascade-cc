import { spawn } from 'node:child_process';
import { getSettings } from '../config/SettingsService.js';
import { HookEvent, HookInput, HookOutput, HookOutputSchema, HookTrigger, HookDefinition, HookInputSchema } from './HookTypes.js';
import { PromptManager } from '../conversation/PromptManager.js';
import { Anthropic } from '../anthropic/AnthropicClient.js';
import { EnvService } from '../config/EnvService.js';
import { findAgent } from '../agents/AgentPersistence.js';
import { terminalLog } from '../../utils/shared/runtime.js';

export class HookService {
    private dynamicHooks = new Map<string, HookTrigger[]>();

    /**
     * Registers a dynamic hook (e.g. from a plugin).
     */
    registerHook(event: HookEvent, hook: HookTrigger) {
        const existing = this.dynamicHooks.get(event) || [];
        this.dynamicHooks.set(event, [...existing, hook]);
    }
    /**
     * Dispatches a hook event, executing all configured hooks for that event.
     */
    async dispatch(event: HookEvent, input: HookInput): Promise<HookOutput[]> {
        const settings = getSettings();
        if (!settings.hooks) {
            return [];
        }

        const hooks = settings.hooks[event] || [];
        const dynamic = this.dynamicHooks.get(event) || [];
        const allHooks = [...hooks, ...dynamic];

        if (allHooks.length === 0) {
            return [];
        }

        if (settings.allowManagedHooksOnly) {
            // In a real implementation, we would filter hooks by source (policy vs user).
            // For now, we are just acknowledging the setting exists and is checked here.
            // console.warn("[HookService] allowManagedHooksOnly is enabled. Verifying hooks...");
        }

        // Validate input (optional, but good practice)
        const parseResult = HookInputSchema.safeParse(input);
        if (!parseResult.success) {
            console.warn(`[HookService] Invalid input for event ${event}:`, parseResult.error);
            // Proceeding with provided input anyway, or could return empty
        }

        const results: HookOutput[] = [];

        // Loop over all triggers (from settings and dynamic)
        for (const trigger of allHooks) {
            // Check Matcher
            if (!this.matches(trigger.matcher, event, input)) {
                continue;
            }

            // Loop over defined hooks in this trigger
            for (const hookDef of trigger.hooks) {
                try {
                    const result = await this.executeHook(hookDef, input);
                    results.push(result);

                    if (result.continue === false) {
                        // If one hook says stop, we should probably stop?
                        // Reference implementation logic for stopping chain?
                    }
                } catch (error) {
                    console.error(`[HookService] Error executing hook for ${event}:`, error);
                    results.push({
                        continue: true,
                        systemMessage: `Hook execution failed: ${error instanceof Error ? error.message : String(error)}`,
                        decision: 'approve'
                    });
                }
            }
        }
        return results;
    }

    private matches(matcher: string | undefined, event: HookEvent, input: HookInput): boolean {
        if (!matcher) return true;
        if (matcher === '*') return true;

        let valueToMatch = '';

        switch (event) {
            case 'PreToolUse':
            case 'PostToolUse':
            case 'PostToolUseFailure':
            case 'PermissionRequest':
                valueToMatch = input.tool_name || '';
                break;
            case 'SessionStart':
                // "how the session started": startup, resume, clear, compact
                // input.trigger? input.source?
                // HookInputSchema has 'trigger' for PreCompact. 
                // SessionStart doesn't have a specific field in schema for source yet, assuming generic input context?
                // Let's assume input.session_start_type or similar if defined, or just fallback?
                // The docs say "SessionStart matches on how the session started".
                // Just return true if we can't find it for now, or match against "startup" default.
                valueToMatch = (input as any).source || 'startup';
                break;
            case 'SessionEnd':
                valueToMatch = (input as any).reason || 'other';
                break;
            case 'Notification':
                valueToMatch = input.notification_type || '';
                break;
            case 'SubagentStart':
            case 'SubagentStop':
                valueToMatch = input.tool_name || (input as any).agent_name || '';
                break;
            case 'PreCompact':
                valueToMatch = input.trigger || '';
                break;
            default:
                // UserPromptSubmit, Stop have no matcher support
                return true;
        }

        try {
            const regex = new RegExp(matcher);
            return regex.test(valueToMatch);
        } catch (e) {
            return matcher === valueToMatch;
        }
    }

    /**
     * Executes a single hook configuration (Command, Prompt, or Agent).
     */
    private async executeHook(hookDef: HookDefinition, input: HookInput): Promise<HookOutput> {
        switch (hookDef.type) {
            case 'command':
                return this.executeCommandHook(hookDef, input);
            case 'prompt':
                return this.executePromptHook(hookDef, input);
            case 'agent':
                return this.executeAgentHook(hookDef, input);
            default:
                throw new Error(`Unknown hook type: ${(hookDef as any).type}`);
        }
    }

    private async executePromptHook(hookDef: HookDefinition, input: HookInput): Promise<HookOutput> {
        const prompt = hookDef.prompt;
        if (!prompt) throw new Error("No prompt specified for prompt hook");

        const client = new Anthropic({
            baseUrl: EnvService.get("ANTHROPIC_BASE_URL")
        });

        const system = `You are a hook evaluator for Claude Code. 
Your job is to evaluate the provided context against a condition and return a JSON decision.
Return ONLY valid JSON. Format: { "ok": boolean, "reason": string }`;

        const userMessage = `Context: ${JSON.stringify(input, null, 2)}\n\nCondition: ${prompt}`;

        try {
            // We use a safe default model
            const response = await client.messages.create({
                model: EnvService.get("ANTHROPIC_MODEL") || 'claude-3-haiku-20240307',
                max_tokens: 1024,
                system,
                messages: [{ role: 'user', content: userMessage }],
                stream: false
            });

            const content = response.content[0].text;
            const jsonMatch = content.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
                const decision = JSON.parse(jsonMatch[0]);
                if (decision.ok) {
                    return { continue: true, decision: 'approve' };
                } else {
                    return {
                        continue: true, // Should we stop? Hooks usually provide feedback.
                        decision: 'block',
                        reason: decision.reason
                    };
                }
            }
            return { continue: true, decision: 'approve', systemMessage: "Could not parse hook decision" };
        } catch (e) {
            return { continue: true, decision: 'approve', systemMessage: `Prompt hook failed: ${e}` };
        }
    }

    private async executeAgentHook(hookDef: HookDefinition, input: HookInput): Promise<HookOutput> {
        // Dynamically import to avoid circular dependency
        const { ConversationService } = await import('../conversation/ConversationService.js');

        const prompt = hookDef.prompt;
        if (!prompt) throw new Error("No prompt specified for agent hook");

        try {
            terminalLog("Executing Agent Hook...", "info");
            // Run a short conversation
            // We need to capture the output or decision. 
            // The docs say the agent returns { "ok": boolean, "reason": string } as final text or via tool?
            // "Agent-based hooks use the same 'ok' / 'reason' response format"
            // This implies the agent's LAST message should be JSON.

            // We'll run startConversation with maxTurns=50 (default)
            const generator = ConversationService.startConversation(prompt, {
                commands: [], // Should probably pass registry
                tools: [], // Should pass tools
                mcpClients: [],
                cwd: hookDef.cwd || process.cwd(),
                verbose: false,
                model: EnvService.get("ANTHROPIC_MODEL") || 'claude-3-haiku-20240307',
                agent: 'Plan', // or just default? Docs say "agent-based hooks", maybe imply using "Agent" capability
                // We need to inject the context as system prompt context or user message
            });

            let lastMessage = "";
            // Consume generator
            for await (const chunk of generator) {
                if (chunk.type === 'result' && chunk.result) {
                    lastMessage = chunk.result;
                }
            }

            // Parse last message for JSON
            const jsonMatch = lastMessage.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
                const decision = JSON.parse(jsonMatch[0]);
                return {
                    continue: true,
                    decision: decision.ok ? 'approve' : 'block',
                    reason: decision.reason
                };
            }

            return { continue: true, decision: 'approve', systemMessage: "Agent hook did not return valid JSON decision" };
        } catch (e) {
            return { continue: true, decision: 'approve', systemMessage: `Agent hook failed: ${e}` };
        }
    }

    private async executeCommandHook(hookConfig: HookDefinition, input: HookInput): Promise<HookOutput> {
        // Support legacy 'commands' array if present in raw config, but HookDefinition schema focuses on 'command'
        let commandStr = hookConfig.command;
        // Fallback logic if needed (though definition schema is strict now)
        if (!commandStr && (hookConfig as any).commands) {
            const cmds = (hookConfig as any).commands;
            commandStr = Array.isArray(cmds) ? cmds[0] : cmds;
        }

        if (!commandStr) {
            throw new Error("No command specified in hook configuration");
        }

        // Prepare input JSON
        const inputJson = JSON.stringify(input);

        return new Promise((resolve, reject) => {
            const timeoutMs = hookConfig.timeout ? hookConfig.timeout * 1000 : 10000; // Default 10s
            const abortController = new AbortController();
            const timeoutId = setTimeout(() => {
                abortController.abort();
            }, timeoutMs);

            const cwd = hookConfig.cwd || process.cwd();

            // Use shell to execute command
            const child = spawn(commandStr, [], {
                cwd,
                shell: true,
                signal: abortController.signal,
                stdio: ['pipe', 'pipe', 'pipe']
            });

            let stdout = '';
            let stderr = '';

            if (child.stdin) {
                child.stdin.write(inputJson);
                child.stdin.end();
            }

            child.stdout?.on('data', (data) => {
                stdout += data.toString();
            });

            child.stderr?.on('data', (data) => {
                stderr += data.toString();
            });

            child.on('close', (code) => {
                clearTimeout(timeoutId);
                if (code !== 0) {
                    // Hook failed?
                    console.warn(`[HookService] Hook exited with code ${code}. Stderr: ${stderr}`);
                    // We interpret non-zero as failure but maybe still try to parse output?
                    // If output is empty but code is non-zero, it's an error.
                    if (!stdout.trim()) {
                        resolve({
                            continue: true,
                            decision: 'approve',
                            systemMessage: `Hook exited with code ${code}: ${stderr}`
                        });
                        return;
                    }
                }

                // Parse Output
                try {
                    // Attempt to parse stdout as JSON
                    // Sometimes hooks might output raw text logs before JSON. 
                    // Reference implementation seems to look for JSON.
                    const jsonStart = stdout.indexOf('{');
                    const jsonEnd = stdout.lastIndexOf('}');
                    if (jsonStart !== -1 && jsonEnd !== -1 && jsonEnd > jsonStart) {
                        const jsonStr = stdout.substring(jsonStart, jsonEnd + 1);
                        const parsed = JSON.parse(jsonStr);
                        const result = HookOutputSchema.safeParse(parsed);
                        if (result.success) {
                            resolve(result.data);
                        } else {
                            // Could not validate schema
                            console.warn(`[HookService] Invalid hook output schema:`, result.error);
                            resolve({
                                continue: true,
                                systemMessage: `Invalid hook output schema`,
                                decision: 'approve'
                            });
                        }
                    } else {
                        // No JSON found, treat as success/continue but log output?
                        resolve({
                            continue: true,
                            decision: 'approve',
                            // plainText: stdout // If we want to capture it
                        });
                    }
                } catch (e) {
                    console.error(`[HookService] Error parsing hook output:`, e);
                    resolve({
                        continue: true,
                        decision: 'approve',
                        systemMessage: `Error parsing hook output`
                    });
                }
            });

            child.on('error', (err) => {
                clearTimeout(timeoutId);
                if (err.name === 'AbortError') {
                    resolve({
                        continue: true,
                        decision: 'approve',
                        systemMessage: `Hook execution timed out after ${timeoutMs}ms`
                    });
                } else {
                    reject(err);
                }
            });
        });
    }
}

export const hookService = new HookService();
