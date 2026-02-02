/**
 * File: src/commands/SlashCommandDispatcher.ts
 * Role: Central dispatcher for handling slash commands in the REPL.
 */

import { PromptManager } from "../services/conversation/PromptManager.js";
import { DoctorService, HealthCheckResult } from "../services/terminal/DoctorService.js";
import { EnvService } from "../services/config/EnvService.js";
import { readFileSync, existsSync } from 'fs';
import { join } from 'path';
import { updateSettings } from "../services/config/SettingsService.js";

// ... (imports remain the same)

export interface CommandContext {
    setMessages: (updater: (prev: any[]) => any[]) => void;
    setPlanMode: (updater: (prev: boolean) => boolean) => void;
    setVimModeEnabled: (updater: (prev: boolean) => boolean) => void;
    setShowTasks: (updater: (prev: boolean) => boolean) => void;
    setIsTyping: (isTyping: boolean) => void;
    exit: () => void;
    cwd: string;
    setCurrentMenu: (menu: 'config' | 'mcp' | 'tasks' | 'search' | 'model' | 'status' | 'agents' | 'bug' | 'doctor' | 'compact' | 'memory' | 'cost' | null) => void;
    setBugReportInitialDescription: (description: string) => void;
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
                        return true;
                    } else {
                        // Return false to let the REPL handle this as a prompt command
                        // The actual prompt is handled by the LLM seeing the system instructions or 
                        // by being injected into the conversation.
                        return false;
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
                    const { OAuthService, LoopbackServerHandler } = await import('../services/auth/OAuthService.js');
                    const handler = new LoopbackServerHandler();
                    try {
                        const { generateRandomString, pkceChallenge } = await import('../utils/shared/crypto.js');
                        const open = (await import('open')).default;

                        const state = generateRandomString(32);
                        const codeVerifier = generateRandomString(64);
                        const codeChallenge = pkceChallenge(codeVerifier);

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

                        // Fetch profile and roles
                        const profile = await OAuthService.fetchProfile(tokenResponse.access_token);
                        const roles = await OAuthService.fetchRoles(tokenResponse.access_token);

                        const authAccount = {
                            accountUuid: profile.account.uuid,
                            emailAddress: profile.account.email,
                            organizationUuid: profile.organization.uuid,
                            displayName: profile.account.display_name,
                            hasExtraUsageEnabled: profile.organization.has_extra_usage_enabled,
                            billingType: profile.organization.billing_type,
                            organizationRole: roles.organization_role,
                            workspaceRole: roles.workspace_role,
                            organizationName: roles.organization_name
                        };

                        await OAuthService.saveToken({
                            ...tokenResponse,
                            account: authAccount
                        });

                        context.setMessages(prev => [...prev, { role: 'assistant', content: '✅ Successfully logged in! Your credentials have been saved to the system keychain.' }]);
                    } catch (e) {
                        context.setMessages(prev => [...prev, { role: 'assistant', content: `❌ Login failed: ${e instanceof Error ? e.message : String(e)}` }]);
                    } finally {
                        handler.closeServer();
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
