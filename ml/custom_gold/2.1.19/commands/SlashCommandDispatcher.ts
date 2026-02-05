/**
 * File: src/commands/SlashCommandDispatcher.ts
 * Role: Central dispatcher for handling slash commands in the REPL.
 */

import { PromptManager } from "../services/conversation/PromptManager.js";
import { DoctorService } from "../services/terminal/DoctorService.js";
import { EnvService } from "../services/config/EnvService.js";
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';
import { updateSettings } from "../services/config/SettingsService.js";
import { commandRegistry } from "../services/terminal/CommandRegistry.js";

// ... (imports remain the same)

export interface CommandContext {
    setMessages: (updater: (prev: any[]) => any[]) => void;
    setPlanMode: (updater: (prev: boolean) => boolean) => void;
    setVimModeEnabled: (updater: (prev: boolean) => boolean) => void;
    setShowTasks: (updater: (prev: boolean) => boolean) => void;
    setIsTyping: (isTyping: boolean) => void;
    exit: () => void;
    cwd: string;
    setCurrentMenu: (menu: 'config' | 'mcp' | 'tasks' | 'search' | 'model' | 'status' | 'agents' | 'bug' | 'doctor' | 'compact' | 'memory' | 'cost' | 'marketplace' | 'resources' | 'prompts' | null) => void;
    setBugReportInitialDescription: (description: string) => void;
    messages: any[];
}

export class SlashCommandDispatcher {
    static async handleCommand(input: string, context: CommandContext): Promise<boolean | string> {
        const cmd = input.trim().split(' ')[0];
        const args = input.trim().split(' ').slice(1).join(' ');

        // Check registry for prompt commands first
        const commandDef = commandRegistry.findCommand(cmd);
        if (commandDef && commandDef.type === 'prompt') {
            const promptResult = await commandDef.getPromptForCommand(args, { ...context, version: '2.0.0' /* Stub version */ });
            if (Array.isArray(promptResult) && promptResult.length > 0 && promptResult[0].text) {
                return promptResult[0].text;
            }
        }

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

            case '/marketplace':
                context.setCurrentMenu('marketplace');
                return true;

            case '/resources':
                context.setCurrentMenu('resources');
                return true;

            case '/prompts':
                context.setCurrentMenu('prompts');
                return true;

            case '/init':
                {
                    const claudeMdPath = join(context.cwd, 'CLAUDE.md');
                    if (existsSync(claudeMdPath)) {
                        context.setMessages(prev => [...prev, { role: 'assistant', content: '✅ `CLAUDE.md` already exists in this directory.' }]);
                        return true;
                    } else {
                        // Return false to let the REPL handle this as a prompt command
                        return false; // This triggers fallthrough to REPL which likely fails if not handled. 
                        // But since we added registry check at top, /init from registry SHOULD have been caught?
                        // "init" is registered. "findCommand('/init')" finds it.
                        // So the check at top returns the string prompt.
                        // Wait, if it matches switch case '/init', we enter here.
                        // So we should remove '/init' from switch OR return the prompt here?
                        // Registry has logic. Switch has logic.
                        // Use registry logic if available?
                        // For init, we want the prompt.

                        // We handled prompt commands at TOP. So /init shouldn't reach here if it's in registry.
                        // BUT init logic here checks CLAUDE.md existence.
                        // Registry logic (in init.ts) gets prompt.
                        // We should fallback to registry logic by returning false? 
                        // No, if I return false, REPL executes '/init' as prompt.

                        // If I remove '/init' from switch, the top check works.
                        // But checking CLAUDE.md is useful.

                        // I will update this block to return the prompt manually or just let top block handle it?
                        // If I let top block handle, I need to REMOVE this case or check inside top block?
                        // The top block runs BEFORE switch.
                        // So /init is handled by top block. This switch case is UNREACHABLE for /init if registered.
                        // UNLESS getPromptForCommand returns null?

                        // initCommandDefinition is registered.
                        // So top block catches it.
                        // So I should remove /init case?
                        // But wait, the check `if (existsSync(claudeMdPath))` is nice.
                        // init.ts getPromptForCommand doesn't check existence?
                        // Let's check init.ts.
                    }
                }
                return true;

            case '/doctor':
                context.setCurrentMenu('doctor');
                return true;

            case '/compact':
                context.setCurrentMenu('compact');
                return true;

            case '/context':
            case '/cost':
                context.setCurrentMenu('cost');
                return true;

            case '/model':
                if (!args) {
                    context.setCurrentMenu('model');
                } else {
                    updateSettings({ primaryModel: args });
                    context.setMessages(prev => [...prev, { role: 'assistant', content: `Model preference updated and persisted to: **${args}**` }]);
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
                context.setCurrentMenu('memory');
                return true;

            case '/bug':
                context.setBugReportInitialDescription(args);
                context.setCurrentMenu('bug');
                return true;

            case '/login':
                (async () => {
                    const { OAuthService } = await import('../services/auth/OAuthService.js');
                    const open = (await import('open')).default;

                    try {
                        context.setIsTyping(true);
                        await OAuthService.login({
                            onUrl: async (url) => {
                                context.setMessages(prev => [...prev, {
                                    role: 'assistant',
                                    content: `Opening browser for login...\n\nIf it doesn't open automatically, please visit this URL:\n[${url}](${url})`
                                }]);
                                await open(url);
                            }
                        });

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
                        const { OAuthService } = await import('../services/auth/OAuthService.js');
                        await OAuthService.logout();
                        context.setMessages(prev => [...prev, { role: 'assistant', content: '✅ Logged out successfully. Your credentials have been removed.' }]);
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
