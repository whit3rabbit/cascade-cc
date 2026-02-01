/**
 * File: src/commands/SlashCommandDispatcher.ts
 * Role: Central dispatcher for handling slash commands in the REPL.
 */

import { PromptManager } from "../services/conversation/PromptManager.js";
import { DoctorService, HealthCheckResult } from "../services/terminal/DoctorService.js";
import { EnvService } from "../services/config/EnvService.js";
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';

// ... (imports remain the same)

export interface CommandContext {
    setMessages: (updater: (prev: any[]) => any[]) => void;
    setPlanMode: (updater: (prev: boolean) => boolean) => void;
    setVimModeEnabled: (updater: (prev: boolean) => boolean) => void;
    setShowTasks: (updater: (prev: boolean) => boolean) => void;
    setIsTyping: (isTyping: boolean) => void;
    exit: () => void;
    cwd: string;
    setCurrentMenu: (menu: 'config' | 'mcp' | 'tasks' | 'search' | 'model' | 'status' | 'agents' | null) => void;
    messages: any[];
}

export class SlashCommandDispatcher {
    static async handleCommand(input: string, context: CommandContext): Promise<boolean> {
        const cmd = input.trim().split(' ')[0];
        const args = input.trim().split(' ').slice(1).join(' ');

        switch (cmd) {
            case '/clear':
                context.setMessages(() => []);
                return true;

            case '/exit':
                context.exit();
                return true;

            case '/tasks':
            case '/todos':
                context.setCurrentMenu('tasks');
                return true;


            case '/plan':
                context.setPlanMode(prev => !prev);
                context.setMessages(prev => [...prev, { role: 'assistant', content: 'Plan mode toggled.' }]);
                return true;

            case '/vim':
                context.setVimModeEnabled(prev => !prev);
                context.setMessages(prev => [...prev, { role: 'assistant', content: 'Vim mode toggled.' }]);
                return true;

            case '/help':
                context.setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: `**Available commands:**
- **/agents**: Create and manage autonomous agents
- **/bug**: Report a bug or feedback
- **/clear**: Clear conversation history
- **/compact**: Summarize older conversation history to save tokens
- **/config**: Open configuration menu
- **/cost**: Show detailed token usage and estimated cost
- **/doctor**: Run system health diagnostics
- **/exit**: Exit the session
- **/help**: Show this help message
- **/init**: Initialize project with CLAUDE.md
- **/login**: Sign in to Anthropic
- **/logout**: Sign out from Anthropic
- **/mcp**: Manage MCP servers and marketplace plugins
- **/memory**: View current MEMORY.md project context
- **/model**: Switch between Claude models
- **/plan**: Toggle Planning Mode (read-only + PLAN.md)
- **/status**: Show status, config, and usage
- **/tasks**: Toggle background task tracking list
- **/undo**: Revert the last turn of the conversation
- **/vim**: Toggle Vim modal editing mode
`
                }]);
                return true;

            case '/config':
                context.setCurrentMenu('config');
                return true;

            case '/mcp':
                context.setCurrentMenu('mcp');
                return true;

            case '/init':
                {
                    const claudeMdPath = join(context.cwd, 'CLAUDE.md');
                    if (existsSync(claudeMdPath)) {
                        context.setMessages(prev => [...prev, { role: 'assistant', content: '✅ `CLAUDE.md` already exists in this directory.' }]);
                    } else {
                        context.setMessages(prev => [...prev, { role: 'assistant', content: 'I will help you initialize this project by creating a `CLAUDE.md` file. I need to analyze your project structure first...' }]);
                        // We return false to let the LLM handle the prompt defined in init.ts
                        return false;
                    }
                }
                return true;

            case '/doctor':
                context.setMessages(prev => [...prev, { role: 'assistant', content: 'Running diagnostics...' }]);
                DoctorService.runChecks().then((results: HealthCheckResult[]) => {
                    const content = `**Claude Code Doctor**\n\n` +
                        results.map((r: HealthCheckResult) => {
                            const icon = r.status === 'ok' ? '✅' : r.status === 'warn' ? '⚠️' : '❌';
                            let item = `${icon} **${r.name}**: ${r.message}`;
                            if (r.details) {
                                item += `\n\`\`\`\n${r.details}\n\`\`\``;
                            }
                            return item;
                        }).join('\n\n');

                    context.setMessages(prev => [...prev, {
                        role: 'assistant',
                        content
                    }]);
                });
                return true;

            case '/compact':
                context.setIsTyping(true);
                PromptManager.compactMessages(context.messages || [], { model: "claude-3-5-sonnet-20241022" })
                    .then(compacted => {
                        context.setMessages(() => compacted);
                        context.setMessages(prev => [...prev, { role: 'assistant', content: '✅ Conversation history compacted.' }]);
                    })
                    .finally(() => context.setIsTyping(false));
                return true;

            case '/context':
            case '/cost':
                {
                    const { costService } = await import('../services/terminal/CostService.js');
                    const usage = costService.getUsage();
                    const totalCost = costService.calculateCost();

                    context.setMessages(prev => [...prev, {
                        role: 'assistant',
                        content: `**Session Usage & Cost**
- **Tokens**: ${usage.inputTokens} in / ${usage.outputTokens} out
- **Cache**: ${usage.cacheReadTokens || 0} read / ${usage.cacheWriteTokens || 0} written
- **Estimated Cost**: **$${totalCost.toFixed(4)}**`
                    }]);
                }
                return true;

            case '/model':
                if (!args) {
                    context.setCurrentMenu('model');
                } else {
                    // In a real app, this would persist the selection to session state or settings
                    context.setMessages(prev => [...prev, { role: 'assistant', content: `Model preference updated to: **${args}**` }]);
                    // Ideally we would trigger a callback or service update here
                }
                return true;

            case '/status':
                context.setCurrentMenu('status');
                return true;

            case '/agents':
                context.setCurrentMenu('agents');
                return true;

            case '/undo':
                context.setMessages(prev => prev.length >= 2 ? prev.slice(0, -2) : []);
                return true;

            case '/memory':
                {
                    const memoryPath = join(context.cwd, 'MEMORY.md');
                    if (existsSync(memoryPath)) {
                        const content = readFileSync(memoryPath, 'utf8');
                        context.setMessages(prev => [...prev, { role: 'assistant', content: `**Current MEMORY.md Content:**\n\n${content}` }]);
                    } else {
                        context.setMessages(prev => [...prev, { role: 'assistant', content: '_MEMORY.md not found in current directory._' }]);
                    }
                }
                return true;

            case '/bug':
                context.setMessages(prev => [...prev, {
                    role: 'assistant',
                    content: 'Found a bug or have feedback? Please report it on our GitHub repository: [https://github.com/anthropics/claude-code/issues](https://github.com/anthropics/claude-code/issues)'
                }]);
                return true;

            case '/login':
                (async () => {
                    try {
                        const { OAuthService, LoopbackServerHandler } = await import('../services/auth/OAuthService.js');
                        const { generateRandomString, pkceChallenge } = await import('../utils/shared/crypto.js');
                        const open = (await import('open')).default;

                        const state = generateRandomString(32);
                        const codeVerifier = generateRandomString(64);
                        const codeChallenge = pkceChallenge(codeVerifier);

                        const handler = new LoopbackServerHandler();
                        const port = await handler.start();
                        const promise = handler.listenForAuthCode(
                            "Login successful! You can close this window.",
                            "Login failed. Please try again."
                        );

                        const authUrl = OAuthService.buildAuthUrl({
                            codeChallenge,
                            state,
                            port,
                            isManual: false
                        });

                        context.setMessages(prev => [...prev, {
                            role: 'assistant',
                            content: `Opening browser for login...\n\nIf it doesn't open automatically, please visit this URL:\n[${authUrl}](${authUrl})`
                        }]);

                        await open(authUrl);

                        const params = await promise;
                        if (params.state !== state) {
                            throw new Error("Invalid state returned from OAuth flow.");
                        }

                        context.setIsTyping(true);
                        const tokenResponse = await OAuthService.exchangeToken(
                            params.code!,
                            state,
                            codeVerifier,
                            port
                        );

                        await OAuthService.saveToken(tokenResponse);
                        context.setMessages(prev => [...prev, { role: 'assistant', content: '✅ Successfully logged in! Your credentials have been saved to the system keychain.' }]);
                    } catch (e) {
                        context.setMessages(prev => [...prev, { role: 'assistant', content: `❌ Login failed: ${e instanceof Error ? e.message : String(e)}` }]);
                    } finally {
                        context.setIsTyping(false);
                    }
                })();
                return true;

            case '/logout':
                (async () => {
                    try {
                        const { KeychainService } = await import('../services/auth/KeychainService.js');
                        const { getProductName } = await import('../utils/shared/product.js');
                        const { setStatsigStorage } = await import('../services/auth/MsalAuthService.js');

                        const serviceName = getProductName("-credentials");
                        if (KeychainService.isAvailable()) {
                            KeychainService.saveToken(serviceName, ""); // Clear keychain
                        }
                        setStatsigStorage({ accessToken: null, oauthAccount: null }); // Clear in-memory

                        context.setMessages(prev => [...prev, { role: 'assistant', content: '✅ Successfully logged out.' }]);
                    } catch (e) {
                        context.setMessages(prev => [...prev, { role: 'assistant', content: `❌ Logout failed: ${e instanceof Error ? e.message : String(e)}` }]);
                    }
                })();
                return true;

            default:
                if (cmd.startsWith('/')) {
                    context.setMessages(prev => [...prev, { role: 'assistant', content: `Unknown command: ${cmd}` }]);
                    return true;
                }
                return false;
        }
    }
}
