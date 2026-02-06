/**
 * File: src/commands/SlashCommandDispatcher.ts
 * Role: Central dispatcher for handling slash commands in the REPL.
 */

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
    setCurrentMenu: (menu: 'config' | 'mcp' | 'tasks' | 'search' | 'model' | 'status' | 'agents' | 'bug' | 'doctor' | 'compact' | 'memory' | 'cost' | 'marketplace' | 'resources' | 'prompts' | 'remote-env' | null, options?: any) => void;
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

            case '/usage':
                context.setCurrentMenu('config', { tab: 'Usage' });
                return true;

            case '/keybindings':
                (async () => {
                    const { join } = await import('path');
                    const { homedir } = await import('os');
                    const { existsSync, mkdirSync, writeFileSync } = await import('fs');
                    const open = (await import('open')).default;

                    const keybindingsPath = join(homedir(), '.claude', 'keybindings.json');
                    const dir = join(homedir(), '.claude');

                    if (!existsSync(dir)) mkdirSync(dir, { recursive: true });
                    if (!existsSync(keybindingsPath)) {
                        writeFileSync(keybindingsPath, JSON.stringify([
                            {
                                "key": "alt+enter",
                                "command": "chat:insertNewline",
                                "when": "inputHasFocus"
                            }
                        ], null, 2));
                    }

                    try {
                        await open(keybindingsPath);
                        context.setMessages(prev => [...prev, { role: 'assistant', content: `Opened keybindings file: **${keybindingsPath}**` }]);
                    } catch (e) {
                        context.setMessages(prev => [...prev, { role: 'assistant', content: `❌ Failed to open keybindings: ${e instanceof Error ? e.message : String(e)}` }]);
                    }
                })();
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


            case '/doctor':
                context.setCurrentMenu('doctor');
                return true;

            case '/compact':
                context.setCurrentMenu('config', { tab: 'Status' });
                return true;

            case '/usage':
            case '/stats':
            case '/status': {
                const tab = cmd === '/usage' ? 'Usage' : 'Status';
                context.setCurrentMenu('config', { tab });
                return true;
            }

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

            case '/teleport':
                context.setMessages(prev => [...prev, { role: 'assistant', content: args ? `Teleporting to session: **${args}** (Note: Teleportation logic is currently a stub)` : `Usage: /teleport <session_id>` }]);
                return true;

            case '/remote-env':
                context.setCurrentMenu('remote-env');
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
                            onUrl: async (_manualUrl, autoUrl) => {
                                context.setMessages(prev => [...prev, {
                                    role: 'assistant',
                                    content: `Opening browser for login...\n\nIf it doesn't open automatically, please visit this URL:\n[${autoUrl}](${autoUrl})`
                                }]);
                                await open(autoUrl);
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
