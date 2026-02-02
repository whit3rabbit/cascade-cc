/**
 * File: src/components/terminal/REPL.tsx
 * Role: Main interactive UI component using Ink.
 */

import React, { useState, useEffect } from 'react';
import { Box, Text, useApp } from 'ink';
import { ConversationService } from '../../services/conversation/ConversationService.js';
import { checkToolPermissions, handlePermissionResponse } from '../../services/terminal/PermissionService.js';
import { ToolExecutionManager } from '../../services/tools/ToolExecutionManager.js';
import { UserPromptMessage } from '../messages/UserPromptMessage.js';
import { MessageHistory } from '../MessageHistory.js';
import { UnifiedSearchMenu } from '../menus/UnifiedSearchMenu.js';
import { TaskMenu } from '../menus/TaskMenu.js';




export interface REPLProps {
    initialPrompt?: string;
    verbose?: boolean;
    model?: string;
    agent?: string;
    isFirstRun?: boolean;
}

/**
 * REPL component for interactive Claude sessions.
 */
import { PermissionDialog } from '../permissions/PermissionDialog.js';
import { WorkerPermissionDialog } from '../permissions/WorkerPermissionDialog.js';
import { SandboxPermissionDialog } from '../permissions/SandboxPermissionDialog.js';
import { CostThresholdDialog } from '../permissions/CostThresholdDialog.js';
import { IdeOnboardingDialog } from '../onboarding/IdeOnboardingDialog.js';
import { OnboardingWorkflow } from '../onboarding/OnboardingWorkflow.js';
import { LspRecommendationDialog } from '../onboarding/LspRecommendationDialog.js';
import { SettingsMenu } from '../menus/SettingsMenu.js';
import { McpMenu } from '../menus/McpMenu.js';
import { AgentsMenu } from '../menus/AgentsMenu.js';
import { BugReportCommand } from '../../commands/BugReportCommand.js';
import { TaskList } from '../TaskList.js';
import { StatusLine } from './StatusLine.js';
import { ModelPicker } from '../ModelPicker/ModelPicker.js';
import { CompactCommand } from '../../commands/CompactCommand.js';
import { MemoryCommand } from '../../commands/MemoryCommand.js';
import { CostCommand } from '../../commands/CostCommand.js';
import { DoctorCommand } from '../../commands/DoctorCommand.js';
import Spinner from 'ink-spinner';
import { DocumentationService } from '../../services/documentation/DocumentationService.js';
import { useInput } from 'ink';
import { SlashCommandDispatcher } from '../../commands/SlashCommandDispatcher.js';
import { Logo } from './Logo.js';
import { commandRegistry } from '../../services/terminal/CommandRegistry.js';
import { CORE_TOOLS } from '../../tools/index.js';
import { EnvService } from '../../services/config/EnvService.js';
import { getAuthDetails } from '../../services/auth/AuthService.js';
import { getSettings, updateSettings } from '../../services/config/SettingsService.js';
import { taskManager, Task } from '../../services/terminal/TaskManager.js';
import { costService } from '../../services/terminal/CostService.js';

import { hookService } from '../../services/hooks/HookService.js';

export const REPL: React.FC<REPLProps> = ({ initialPrompt, verbose, model, agent, isFirstRun }) => {
    const { exit } = useApp();
    const [messages, setMessages] = useState<any[]>([]);
    const [isTyping, setIsTyping] = useState(false);
    const [history, setHistory] = useState<string[]>([]);
    const [tasks, setTasks] = useState<Task[]>([]);
    const [showTasks, setShowTasks] = useState(false);
    const [vimModeEnabled, setVimModeEnabled] = useState(false);
    const [vimMode, setVimMode] = useState<'NORMAL' | 'INSERT'>('INSERT');
    const [planMode, setPlanMode] = useState(false);
    const [verboseMode, setVerboseMode] = useState(verbose || false);
    const [lastEscPress, setLastEscPress] = useState(0);

    // Permission & Modal States
    const [toolPermissions, setToolPermissions] = useState<any[]>([]);
    const [workerPermissions, setWorkerPermissions] = useState<any[]>([]);
    const [sandboxPermissions, setSandboxPermissions] = useState<any[]>([]);
    const [showCostWarning, setShowCostWarning] = useState(false);
    const [showIdeOnboarding, setShowIdeOnboarding] = useState(!getSettings().onboardingComplete);
    const [lspRecommendation, setLspRecommendation] = useState<any>(null);
    const [cost, setCost] = useState(0);
    const [costThreshold] = useState(0.50); // Default threshold of $0.50
    const [usage, setUsage] = useState<{ inputTokens: number; outputTokens: number }>({ inputTokens: 0, outputTokens: 0 });
    const [mcpTools, setMcpTools] = useState<any[]>([]);
    const [currentMenu, setCurrentMenu] = useState<'config' | 'mcp' | 'search' | 'tasks' | 'model' | 'status' | 'agents' | 'bug' | 'doctor' | 'compact' | 'memory' | 'cost' | null>(null);
    const [bugReportInitialDescription, setBugReportInitialDescription] = useState<string>('');
    const [suggestions, setSuggestions] = useState<string[]>([]);
    const [exitConfirmation, setExitConfirmation] = useState(false);
    const [shellSnapshotPath, setShellSnapshotPath] = useState<string | undefined>(undefined);
    const [subscription, setSubscription] = useState<string>('');
    const [scrollOffset, setScrollOffset] = useState(0);
    const [cwd, setCwd] = useState(process.cwd());
    const [lastChar, setLastChar] = useState('');
    const [isHistorySearching, setIsHistorySearching] = useState(false);
    const [historySearchQuery, setHistorySearchQuery] = useState('');

    useEffect(() => {
        // Dispatch SessionStart hook
        const runSessionHooks = async () => {
            const results = await hookService.dispatch('SessionStart', {
                hook_event_name: 'SessionStart',
                session_id: 'current-session',
                cwd: process.cwd(),
                transcript_path: undefined // placeholder
            });

            // Check for blocking errors or context
            // Check for blocking errors or context
            const context = results
                .map(r => r.hookSpecificOutput)
                .filter((r): r is NonNullable<typeof r> => !!r && 'additionalContext' in r)
                .map(r => (r as any).additionalContext)
                .filter(Boolean);
            if (context.length > 0) {
                // Add to messages so it's part of context
                setMessages(prev => [...prev, {
                    // Using 'system' role for context
                    role: 'system',
                    content: `[SessionStart Context]: ${context.join('\n')}`
                }]);
            }
        };
        runSessionHooks();
    }, []);

    useEffect(() => {
        const settings = getSettings();
        if (settings.vimModeEnabled !== undefined) {
            setVimModeEnabled(settings.vimModeEnabled);
        }
    }, [currentMenu]);

    // Global Shortcuts
    useInput((input, key) => {
        if (activeScreen === 'doctor-report' && input !== '') {
            setCurrentMenu(null);
            return;
        }

        if (isHistorySearching) {
            if (key.return) {
                setIsHistorySearching(false);
                const query = historySearchQuery.toLowerCase();
                // Find first message from Bottom (current) to Top that matches
                const index = messages.slice().reverse().findIndex(m =>
                    m.content && typeof m.content === 'string' && m.content.toLowerCase().includes(query)
                );
                if (index !== -1) {
                    setScrollOffset(index);
                }
                return;
            }
            if (key.escape) {
                setIsHistorySearching(false);
                return;
            }
            if (key.backspace) {
                setHistorySearchQuery(prev => prev.slice(0, -1));
                return;
            }
            if (!key.ctrl && !key.meta && input) {
                setHistorySearchQuery(prev => prev + input);
                return;
            }
            return;
        }

        if (input === 't' && key.ctrl) {
            setCurrentMenu('tasks');
        }

        if (input === 'l' && key.ctrl) {
            setMessages([]);
            setScrollOffset(0);
        }
        if (input === 'd' && key.ctrl) {
            if (vimModeEnabled && vimMode === 'NORMAL') {
                setScrollOffset(prev => Math.max(0, prev - 10));
                return;
            }
            exit();
        }
        if (input === 'c' && key.ctrl) {
            if (exitConfirmation) {
                exit();
            } else {
                setExitConfirmation(true);
                setTimeout(() => setExitConfirmation(false), 2000);
            }
        }
        if (input === 'b' && key.ctrl) {
            // Toggle background tasks view
            setShowTasks(prev => !prev);
        }
        if (key.tab && key.shift) {
            setPlanMode(prev => !prev);
        }
        if (input === 'm' && key.meta) {
            setPlanMode(prev => !prev);
        }

        if (key.pageUp) {
            setScrollOffset(prev => prev + 5);
        }
        if (key.pageDown) {
            setScrollOffset(prev => Math.max(0, prev - 5));
        }

        // Vim Scrolling
        if (vimModeEnabled && vimMode === 'NORMAL') {
            if (input === 'g') {
                if (lastChar === 'g') {
                    setScrollOffset(messages.length);
                    setLastChar('');
                } else {
                    setLastChar('g');
                }
                return;
            }
            setLastChar('');

            if (input === 'G') {
                setScrollOffset(0);
                return;
            }

            if (key.ctrl && input === 'u') {
                setScrollOffset(prev => prev + 10);
                return;
            }

            if (input === '/') {
                setIsHistorySearching(true);
                setHistorySearchQuery('');
                return;
            }
        }

        if (input === 'o' && key.ctrl) {
            setVerboseMode(prev => !prev);
        }

        if (input === 'p' && key.meta) {
            setCurrentMenu('model');
        }

        if (key.escape) {
            const now = Date.now();
            if (now - lastEscPress < 500) {
                if (messages.length > 0) {
                    setMessages(prev => prev.length >= 2 ? prev.slice(0, -2) : []);
                }
            }
            setLastEscPress(now);
        }
    });

    useEffect(() => {
        costService.reset();
        return taskManager.subscribe((newTasks) => {
            setTasks(newTasks);
        });
    }, []);

    useEffect(() => {
        const slashCommands = ['/config', '/mcp', '/compact', '/clear', '/help', '/bug', '/doctor'];
        setSuggestions([...slashCommands, ...history.slice(-10)]);
    }, [history]);

    useEffect(() => {
        const initShell = async () => {
            const { createShellSnapshot } = await import('../../services/terminal/ShellSnapshotService.js');
            const path = await createShellSnapshot(EnvService.get("SHELL"));
            setShellSnapshotPath(path);
        };
        initShell();
    }, []);

    const refreshMcpTools = async () => {
        const { mcpClientManager } = await import('../../services/mcp/McpClientManager.js');
        const tools = await mcpClientManager.getTools();
        setMcpTools(tools);
    };

    useEffect(() => {
        refreshMcpTools();
    }, []);

    useEffect(() => {
        const fetchAuth = async () => {
            const details = await getAuthDetails();
            setSubscription(details.plan);
        };
        fetchAuth();
    }, []);

    const getActiveScreen = () => {
        if (showCostWarning) return 'cost';
        if (toolPermissions.length > 0) return 'tool-permission';
        if (workerPermissions.length > 0) return 'worker-permission';
        if (sandboxPermissions.length > 0) return 'sandbox-permission';
        if (showIdeOnboarding) return 'ide-onboarding';
        if (lspRecommendation) return 'lsp-recommendation';
        if (currentMenu === 'config' || currentMenu === 'status') return 'settings-menu';
        if (currentMenu === 'mcp') return 'mcp-menu';
        if (currentMenu === 'search') return 'search-menu';
        if (currentMenu === 'tasks') return 'tasks-menu';
        if (currentMenu === 'agents') return 'agents-menu';
        if (currentMenu === 'model') return 'model-menu';
        if (currentMenu === 'bug') return 'bug-report';
        if (currentMenu === 'doctor') return 'doctor-report';
        if (currentMenu === 'compact') return 'compact-command';
        if (currentMenu === 'memory') return 'memory-command';
        if (currentMenu === 'cost') return 'cost-command';
        return 'transcript';
    };

    const activeScreen = getActiveScreen();

    useEffect(() => {
        if (initialPrompt) {
            handleInitialPrompt(initialPrompt);
        }
    }, [initialPrompt]);

    const handleInitialPrompt = async (prompt: string) => {
        setIsTyping(true);
        setMessages(prev => [...prev, { role: 'user', content: prompt }]);
        setIsTyping(false);
    };

    const handleCommand = async (input: string) => {
        return await SlashCommandDispatcher.handleCommand(input, {
            setMessages,
            setPlanMode,
            setVimModeEnabled,
            setShowTasks,
            setIsTyping,
            exit,
            cwd: process.cwd(),
            setCurrentMenu,
            setBugReportInitialDescription,
            messages
        });
    };

    const handlePermissionRequest = async (request: any) => {
        return new Promise<string>((resolve) => {
            setToolPermissions(prev => [...prev, {
                tool: request.tool,
                input: request.input,
                message: request.message,
                resolve
            }]);
        });
    };

    const handleSubmit = async (input: string) => {
        if (isTyping) return;
        setHistory(prev => [...prev, input]);

        if (input.startsWith('/')) {
            const handled = await handleCommand(input);
            if (handled) return;
        }

        // Hook: UserPromptSubmit
        let finalInput = input;
        try {
            const hookResults = await hookService.dispatch('UserPromptSubmit', {
                hook_event_name: 'UserPromptSubmit',
                prompt: input,
                cwd: process.cwd()
            });

            const blocked = hookResults.find(r => r.decision === 'block' || r.continue === false);
            if (blocked) {
                setMessages(prev => [...prev, { role: 'user', content: input }, { role: 'assistant', content: `[Hook blocked]: ${blocked.stopReason || blocked.reason || 'Blocked by UserPromptSubmit hook'}` }]);
                return;
            }

            const context = hookResults
                .map(r => r.hookSpecificOutput)
                .filter((r): r is NonNullable<typeof r> => !!r && 'additionalContext' in r)
                .map(r => (r as any).additionalContext) // Cast to any or specific type because not all variants have it
                .filter(Boolean)
                .join('\n');

            if (context) {
                finalInput = `${input}\n\n<hook_context>\n${context}\n</hook_context>`;
            }
        } catch (err) {
            console.error("Error running UserPromptSubmit hooks:", err);
            // Non-blocking error?
        }

        setMessages(prev => [...prev, { role: 'user', content: input }]); // Show original input to user
        setIsTyping(true);

        try {
            // Ensure tools are up to date (Dynamic Registration)
            await refreshMcpTools();

            // Create a generator for the new turn
            const generator = ConversationService.startConversation(finalInput, {
                commands: commandRegistry.getAllCommands(),
                // ...
                tools: [...CORE_TOOLS, ...mcpTools],
                mcpClients: [],
                cwd: cwd,
                verbose: verboseMode,
                model,
                agent,
                planMode,
                setPlanMode,
                shellSnapshotPath,
                onPermissionRequest: handlePermissionRequest
            });

            for await (const chunk of generator) {
                if (chunk.type === 'assistant' || chunk.type === 'partial_assistant') {
                    // check if we already have an assistant message pending, if so update it, else add it
                    setMessages(prev => {
                        const last = prev[prev.length - 1];
                        if (last?.role === 'assistant') {
                            // This is a naive update, in reality we might stream chunks
                            // But ConversationService yields full message objects currently
                            return [...prev.slice(0, -1), chunk.message];
                        }
                        return [...prev, chunk.message];
                    });
                } else if (chunk.type === 'tool_result') {

                    // Add tool results to messages
                    setMessages(prev => [
                        ...prev,
                        {
                            role: 'user',
                            content: `Tool Output: ${JSON.stringify(chunk.results)}`
                        }
                    ]);
                } else if (chunk.type === 'stream_event') {
                    // Track tokens from stream events
                    const event = chunk.event;
                    if (event.type === 'message_start' && event.message?.usage) {
                        const newUsage = {
                            inputTokens: event.message.usage.input_tokens || 0,
                            outputTokens: event.message.usage.output_tokens || 0,
                            cacheReadTokens: event.message.usage.cache_read_input_tokens || 0,
                            cacheWriteTokens: event.message.usage.cache_creation_input_tokens || 0
                        };
                        costService.addUsage(newUsage);
                        setCost(costService.calculateCost(model));
                        setUsage(prev => ({
                            inputTokens: prev.inputTokens + newUsage.inputTokens,
                            outputTokens: prev.outputTokens + newUsage.outputTokens
                        }));
                    } else if (event.type === 'message_delta' && event.usage) {
                        const newOutput = event.usage.output_tokens || 0;
                        costService.addUsage({ inputTokens: 0, outputTokens: newOutput });
                        setCost(costService.calculateCost(model));
                        setUsage(prev => ({
                            ...prev,
                            outputTokens: prev.outputTokens + newOutput
                        }));
                    }
                } else if (chunk.type === 'cwd_update') {
                    setCwd(chunk.cwd);
                } else if (chunk.type === 'result') {
                    // Final result
                    if (chunk.subtype === 'success') {
                        // maybe add a system message or just stop typing
                    }
                }
            }
            // After turn, check cost threshold
            if (cost >= costThreshold && !showCostWarning) {
                setShowCostWarning(true);
            }
        } catch (error) {
            setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${error}` }]);
            // Also check on error
            if (cost >= costThreshold && !showCostWarning) {
                setShowCostWarning(true);
            }
        } finally {
            setIsTyping(false);
        }
    };

    return (
        <Box flexDirection="column" padding={0} width="100%" height="100%">
            {showTasks && <TaskList tasks={tasks} />}

            <Box flexGrow={1} flexDirection="column" overflowY="hidden" paddingX={2}>
                {activeScreen === 'transcript' ? (
                    <>
                        {messages.length === 0 ? (
                            <Box flexDirection="column" paddingTop={1}>
                                <Logo version="2.1.27" model={model || "Sonnet 4.5"} cwd={process.cwd()} subscription={subscription} />
                            </Box>
                        ) : (
                            <MessageHistory messages={messages} scrollOffset={scrollOffset} />
                        )}
                    </>
                ) : (
                    <Box flexGrow={1} flexDirection="column">
                        {activeScreen === 'cost' && (
                            <CostThresholdDialog
                                onApprove={() => setShowCostWarning(false)}
                                onExit={() => exit()}
                                cost={cost}
                                threshold={costThreshold}
                            />
                        )}
                        {activeScreen === 'settings-menu' && (
                            <SettingsMenu
                                onExit={() => setCurrentMenu(null)}
                                initialTab={currentMenu === 'config' ? 'Config' : 'Status'}
                            />
                        )}
                        {activeScreen === 'mcp-menu' && (
                            <McpMenu onExit={async () => {
                                setCurrentMenu(null);
                                await refreshMcpTools();
                            }} />
                        )}
                        {activeScreen === 'agents-menu' && (
                            <AgentsMenu onExit={() => setCurrentMenu(null)} />
                        )}
                        {activeScreen === 'bug-report' && (
                            <BugReportCommand
                                messages={messages}
                                initialDescription={bugReportInitialDescription}
                                onDone={(msg) => {
                                    setCurrentMenu(null);
                                    setMessages(prev => [...prev, { role: 'assistant', content: msg }]);
                                }}
                            />
                        )}
                        {activeScreen === 'search-menu' && (
                            <UnifiedSearchMenu
                                history={history}
                                commands={commandRegistry.getAllCommands().map(c => ({
                                    label: c.name,
                                    value: `/${c.name}`,
                                    description: c.description
                                }))}
                                onSelect={(val) => {
                                    setCurrentMenu(null);
                                    handleSubmit(val);
                                }}
                                onExit={() => setCurrentMenu(null)}
                            />
                        )}
                        {activeScreen === 'tasks-menu' && <TaskMenu onExit={() => setCurrentMenu(null)} />}
                        {activeScreen === 'model-menu' && (
                            <ModelPicker
                                initialModel={model || null}
                                onSelect={(val) => {
                                    setCurrentMenu(null);
                                    if (val) handleSubmit(`/model ${val}`);
                                }}
                                onCancel={() => setCurrentMenu(null)}
                                isStandalone={true}
                            />
                        )}
                        {activeScreen === 'doctor-report' && (
                            <DoctorCommand
                                onDone={(msg) => {
                                    setCurrentMenu(null);
                                    setMessages(prev => [...prev, { role: 'assistant', content: msg }]);
                                }}
                            />
                        )}
                        {activeScreen === 'compact-command' && (
                            <CompactCommand
                                messages={messages}
                                setMessages={setMessages}
                                setIsTyping={setIsTyping}
                                onDone={(msg) => {
                                    setCurrentMenu(null);
                                    setMessages(prev => [...prev, { role: 'assistant', content: msg }]);
                                }}
                            />
                        )}
                        {activeScreen === 'memory-command' && (
                            <MemoryCommand
                                cwd={cwd}
                                onDone={(msg) => {
                                    setCurrentMenu(null);
                                    setMessages(prev => [...prev, { role: 'assistant', content: msg }]);
                                }}
                            />
                        )}
                        {activeScreen === 'cost-command' && (
                            <CostCommand
                                onDone={(msg) => {
                                    setCurrentMenu(null);
                                    setMessages(prev => [...prev, { role: 'assistant', content: msg }]);
                                }}
                            />
                        )}
                    </Box>
                )}

                {/* Modals/Dialogs that overlay or interrupt */}
                {activeScreen === 'tool-permission' && (
                    <PermissionDialog
                        toolUseConfirm={toolPermissions[0]}
                        onDone={(response: any) => {
                            const req = toolPermissions[0];
                            if (response && response.optionType === 'accept-always') {
                                handlePermissionResponse({ behavior: 'allow', scope: 'always', message: 'User accepted' }, req.tool, req.input, {});
                            }
                            if (req.resolve) req.resolve('allowed');
                            setToolPermissions(prev => prev.slice(1));
                        }}
                        onReject={() => {
                            const req = toolPermissions[0];
                            if (req.resolve) req.resolve('denied');
                            setToolPermissions(prev => prev.slice(1));
                        }}
                        parseInput={(i) => i}
                    />
                )}
                {activeScreen === 'worker-permission' && (
                    <WorkerPermissionDialog
                        request={workerPermissions[0]}
                        onApprove={() => setWorkerPermissions(prev => prev.slice(1))}
                        onReject={() => setWorkerPermissions(prev => prev.slice(1))}
                    />
                )}
                {activeScreen === 'sandbox-permission' && (
                    <SandboxPermissionDialog
                        hostPattern={sandboxPermissions[0]}
                        onDone={() => setSandboxPermissions(prev => prev.slice(1))}
                    />
                )}
                {activeScreen === 'ide-onboarding' && (
                    <OnboardingWorkflow onDone={() => setShowIdeOnboarding(false)} />
                )}
                {activeScreen === 'lsp-recommendation' && (
                    <LspRecommendationDialog onDone={() => setLspRecommendation(null)} />
                )}
            </Box>

            <Box flexDirection="column" width="100%">
                {isTyping && !planMode && (
                    <Box paddingX={1} marginBottom={0}>
                        <Text color="green">
                            <Spinner type="dots" /> Claude is thinking...
                        </Text>
                    </Box>
                )}

                {activeScreen === 'transcript' && (
                    <Box paddingX={2} paddingY={0} flexDirection="column">
                        <Box width="100%">
                            <Text dimColor>{"─".repeat(Math.max(0, Math.min(100, (process.stdout.columns || 80) - 4)))}</Text>
                        </Box>
                        <UserPromptMessage
                            onSubmit={handleSubmit}
                            onClear={() => setMessages([])}
                            onClearScreen={() => {
                                process.stdout.write('\u001b[2J\u001b[3J\u001b[H');
                                setMessages([]);
                            }}
                            history={history}
                            vimModeEnabled={vimModeEnabled}
                            onVimModeChange={setVimMode}
                            planMode={planMode}
                            suggestions={suggestions}
                            onCancel={() => {
                                if (exitConfirmation) exit();
                                else {
                                    setExitConfirmation(true);
                                    setTimeout(() => setExitConfirmation(false), 2000);
                                }
                            }}
                        />
                    </Box>
                )}

                <StatusLine
                    vimMode={vimMode}
                    vimModeEnabled={vimModeEnabled}
                    model={model || 'Sonnet 4.5'}
                    isTyping={isTyping}
                    cwd={cwd}
                    showTasks={showTasks}
                    usage={usage}
                    planMode={planMode}
                    exitConfirmation={exitConfirmation}
                />

                {isHistorySearching && (
                    <Box paddingX={2} marginBottom={0}>
                        <Text color="yellow">/ {historySearchQuery}</Text>
                    </Box>
                )}
            </Box>
        </Box>
    );
};
