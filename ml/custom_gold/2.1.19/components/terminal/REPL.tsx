/**
 * File: src/components/terminal/REPL.tsx
 * Role: Main interactive UI component using Ink.
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Box, useApp, useInput } from 'ink';
import { ConversationService } from '../../services/conversation/ConversationService.js';
import { mcpClientManager } from '../../services/mcp/McpClientManager.js';
import { commandRegistry } from '../../services/terminal/CommandRegistry.js';
import { SlashCommandDispatcher } from '../../commands/SlashCommandDispatcher.js';
import { REPLHeader } from './REPLHeader.js';
import { REPLInput } from './REPLInput.js';
import { REPLKeybindings } from './REPLKeybindings.js';
import { TerminalInput } from './TerminalInput.js';
import { Transcript } from './Transcript.js';
import { UnifiedSearchMenu } from '../menus/UnifiedSearchMenu.js';
import { TaskMenu } from '../menus/TaskMenu.js';
import { LoopMenu } from '../menus/LoopMenu.js';
import { CostThresholdDialog } from '../permissions/CostThresholdDialog.js';
import { SettingsMenu } from '../menus/SettingsMenu.js';
import { McpMenu } from '../menus/McpMenu.js';
import { MarketplaceMenu } from '../menus/MarketplaceMenu.js';
import { ResourceMenu } from '../menus/ResourceMenu.js';
import { PromptMenu } from '../menus/PromptMenu.js';
import { AgentsMenu } from '../menus/AgentsMenu.js';
import { RemoteEnvMenu } from '../menus/RemoteEnvMenu.js';
import { BugReportCommand } from '../../commands/BugReportCommand.js';
import { StatusLine } from './StatusLine.js';
import { ModelPicker } from '../ModelPicker/ModelPicker.js';
import { CompactCommand } from '../../commands/CompactCommand.js';
import { MemoryCommand } from '../../commands/MemoryCommand.js';
import { CostCommand } from '../../commands/CostCommand.js';
import { Doctor } from './Doctor.js';
import { CORE_TOOLS } from '../../tools/index.js';
import { getAuthDetails } from '../../services/auth/AuthService.js';
import { getSettings } from '../../services/config/SettingsService.js';
import { costService } from '../../services/terminal/CostService.js';
import { hookService } from '../../services/hooks/HookService.js';
import { useAppState } from '../../hooks/useAppState.js';
import { useTermSize } from '../../hooks/useTermSize.js';
import { addToPromptHistory } from '../../services/terminal/HistoryService.js';

export interface REPLProps {
    initialPrompt?: string;
    verbose?: boolean;
    model?: string;
    agent?: string;
    isFirstRun?: boolean;
}

export const REPL: React.FC<REPLProps> = ({ initialPrompt: _initialPrompt, verbose: _verbose, model, agent, isFirstRun: _isFirstRun }) => {
    const { exit } = useApp();
    const [appState, updateAppState] = useAppState();

    // Core Local UI states
    const [messages, setMessages] = useState<any[]>([]);
    const [isTyping, setIsTyping] = useState(false);
    const [inputValue, setInputValue] = useState('');
    const [screen, setScreen] = useState<'prompt' | 'transcript'>('prompt');
    const [currentMenu, setCurrentMenu] = useState<any>(null);
    const [history, setHistory] = useState<string[]>([]);
    const [vimModeEnabled, setVimModeEnabled] = useState(false);
    const [currentVimMode, _setCurrentVimMode] = useState('INSERT');
    const [exitConfirmation, _setExitConfirmation] = useState(false);
    const [bugReportInitialDescription, setBugReportInitialDescription] = useState('');
    const [showAllInTranscript, setShowAllInTranscript] = useState(false);
    const [selectedAgent, setSelectedAgent] = useState<string | undefined>(agent);
    const [settingsTab, setSettingsTab] = useState<'Status' | 'Config' | 'Usage'>('Status');
    const [_screenToggleId, setScreenToggleId] = useState(0);

    // Original Screen States
    const [tasksSelected, setTasksSelected] = useState(false);
    const [diffSelected, _setDiffSelected] = useState(false);
    const [loopSelected, setLoopSelected] = useState(false);
    const [teamsSelected, setTeamsSelected] = useState(false);

    // Permission & Modal States
    const [_toolPermissions, setToolPermissions] = useState<any[]>([]);
    const [_workerPermissions, _setWorkerPermissions] = useState<any[]>([]);
    const [_sandboxPermissions, _setSandboxPermissions] = useState<any[]>([]);
    const [_showCostWarning, setShowCostWarning] = useState(false);
    const [_showIdeOnboarding, _setShowIdeOnboarding] = useState(!getSettings().onboardingComplete);
    const [lspRecommendation] = useState<any>(null);
    const [cost, setCost] = useState(0);
    const [usage, _setUsage] = useState({ inputTokens: 0, outputTokens: 0 });
    const [mcpTools, setMcpTools] = useState<any[]>([]);
    const [_subscription, setSubscription] = useState<string>('');
    const [scrollOffset, _setScrollOffset] = useState(0);

    const costThreshold = 0.50;

    useEffect(() => {
        const init = async () => {
            await hookService.dispatch('SessionStart', {
                hook_event_name: 'SessionStart',
                session_id: 'current-session',
                cwd: process.cwd()
            });
            const details = await getAuthDetails();
            setSubscription(details.plan);
            const tools = await mcpClientManager.getTools();
            setMcpTools(tools || []);
        };
        init();
    }, []);

    const handlePermissionRequest = useCallback(async (request: any) => {
        return new Promise<string>((resolve) => {
            setToolPermissions(prev => [...prev, { ...request, resolve }]);
        });
    }, []);

    const cycleMode = useCallback(() => {
        const modes = ['default', 'acceptEdits', 'plan'];
        const currentMode = appState.toolPermissionContext.mode;
        const currentIndex = modes.indexOf(currentMode);
        const nextMode = modes[(currentIndex + 1) % modes.length];
        updateAppState(prev => ({
            ...prev,
            toolPermissionContext: { ...prev.toolPermissionContext, mode: nextMode as any }
        }));
    }, [appState.toolPermissionContext.mode, updateAppState]);

    const cycleScreen = useCallback((direction: 'up' | 'down') => {
        // Original carousel logic
        const availableScreens = ['prompt'];
        if (diffSelected) availableScreens.push('diff');
        if (tasksSelected) availableScreens.push('tasks');
        if (loopSelected) availableScreens.push('loop');
        if (teamsSelected) availableScreens.push('teams');

        const currentActive = tasksSelected ? 'tasks' : loopSelected ? 'loop' : teamsSelected ? 'teams' : 'prompt'; // Simplified check
        const currentIndex = availableScreens.indexOf(currentActive);
        let nextIndex;
        if (direction === 'up') {
            nextIndex = (currentIndex + 1) % availableScreens.length;
        } else {
            nextIndex = (currentIndex - 1 + availableScreens.length) % availableScreens.length;
        }

        const nextScreen = availableScreens[nextIndex];
        setTasksSelected(nextScreen === 'tasks');
        setLoopSelected(nextScreen === 'loop');
        setTeamsSelected(nextScreen === 'teams');
        // diffSelected remains true if it was true, just toggle visibility
    }, [tasksSelected, loopSelected, teamsSelected, diffSelected]);

    useInput((input, key) => {
        if (key.tab && key.shift) {
            if (appState.toolPermissionContext.mode === 'plan') {
                updateAppState(prev => ({
                    ...prev,
                    toolPermissionContext: { ...prev.toolPermissionContext, mode: 'acceptEdits' }
                }));
            } else {
                cycleMode();
            }
        }
        if (key.meta && (key.upArrow || key.downArrow)) {
            cycleScreen(key.upArrow ? 'up' : 'down');
        }
    });

    const handleSubmit = useCallback(async (input: string) => {
        if (isTyping || !input.trim()) return;

        if (input.startsWith('/')) {
            const handled = await SlashCommandDispatcher.handleCommand(input, {
                setMessages,
                setVimModeEnabled,
                setShowTasks: () => setTasksSelected(true),
                setPlanMode: (updater: any) => updateAppState(prev => {
                    const currentPlan = prev.toolPermissionContext.mode === 'plan';
                    const nextPlan = typeof updater === 'function' ? updater(currentPlan) : updater;
                    return {
                        ...prev,
                        toolPermissionContext: { ...prev.toolPermissionContext, mode: nextPlan ? 'plan' : 'default' }
                    };
                }),
                setIsTyping,
                exit,
                cwd: process.cwd(),
                setCurrentMenu: (menu: any, options?: any) => {
                    if (menu === 'config' && options?.tab) {
                        setSettingsTab(options.tab);
                    }
                    setCurrentMenu(menu);
                },
                setBugReportInitialDescription,
                messages
            });
            if (handled === true) return;
            if (typeof handled === 'string') {
                // Execute the returned prompt instead of the original command
                input = handled;
                // Fallthrough to execution logic below
            } else if (handled === false) {
                // Explicit false means not handled, proceed with input (e.g. unknown command or no-op)
            }
        }

        setIsTyping(true);
        setHistory(prev => [...prev, input]);
        addToPromptHistory({ display: input, project: process.cwd() }).catch(() => { });
        setMessages(prev => [...prev, { role: 'user', content: input }]);

        try {
            const generator = ConversationService.startConversation(input, {
                commands: commandRegistry.getAllCommands(),
                tools: [...CORE_TOOLS, ...mcpTools],
                mcpClients: [],
                cwd: process.cwd(),
                verbose: appState.verbose,
                model,
                agent: selectedAgent || agent,
                onPermissionRequest: handlePermissionRequest
            });

            for await (const chunk of generator) {
                if (chunk.type === 'assistant' || chunk.type === 'partial_assistant') {
                    setMessages(prev => {
                        const last = prev[prev.length - 1];
                        if (last?.role === 'assistant') {
                            return [...prev.slice(0, -1), chunk.message];
                        }
                        return [...prev, chunk.message];
                    });
                } else if (chunk.type === 'stream_event') {
                    const event = chunk.event;
                    if (event.type === 'message_start' && event.message?.usage) {
                        costService.addUsage(event.message.usage);
                        setCost(costService.calculateCost(model));
                    }
                }
            }
        } catch (error) {
            setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${error}` }]);
        } finally {
            setIsTyping(false);
        }
    }, [isTyping, mcpTools, appState.verbose, model, agent, handlePermissionRequest, exit, messages]);

    const activeScreen = useMemo(() => {
        if (lspRecommendation) return 'lsp-recommendation';
        if (tasksSelected) return 'tasks';
        if (diffSelected) return 'diff';
        if (loopSelected) return 'loop';
        if (teamsSelected) return 'teams';
        if (currentMenu) return currentMenu;
        return screen;
    }, [lspRecommendation, tasksSelected, diffSelected, loopSelected, teamsSelected, currentMenu, screen]);

    const termSize = useTermSize();

    return (
        <Box flexDirection="column" width="100%" height="100%">
            <REPLKeybindings
                onCancel={() => exit()}
                screen={screen}
                vimMode="INSERT"
                isSearchingHistory={false}
                isHelpOpen={currentMenu === 'help'}
                inputMode="prompt"
                inputValue={inputValue}
            />

            <REPLHeader
                screen={screen}
                setScreen={setScreen}
                setScreenToggleId={setScreenToggleId}
                setShowAllInTranscript={setShowAllInTranscript}
                onEnterTranscript={() => {
                    setScreen('transcript');
                    setScreenToggleId(prev => prev + 1);
                }}
                onExitTranscript={() => {
                    setScreen('prompt');
                    setScreenToggleId(prev => prev + 1);
                }}
                todos={appState.tasks[appState.viewingAgentTaskId || ""]?.steps || []}
                agentName={selectedAgent || agent}
            />

            <Box flexGrow={1} flexDirection="column" paddingX={2}>
                {activeScreen === 'transcript' ? (
                    <Transcript
                        messages={messages}
                        scrollOffset={scrollOffset}
                        showAll={showAllInTranscript}
                        onToggleShowAll={() => setShowAllInTranscript(prev => !prev)}
                        rows={termSize.rows}
                        columns={termSize.columns}
                    />
                ) : (
                    <Box flexGrow={1} flexDirection="column">
                        {activeScreen === 'cost' && <CostThresholdDialog onApprove={() => setShowCostWarning(false)} onExit={() => exit()} cost={cost} threshold={costThreshold} />}
                        {activeScreen === 'config' && <SettingsMenu onExit={() => setCurrentMenu(null)} initialTab={settingsTab} />}
                        {activeScreen === 'mcp' && <McpMenu onExit={async () => setCurrentMenu(null)} />}
                        {activeScreen === 'marketplace' && <MarketplaceMenu onExit={() => setCurrentMenu(null)} />}
                        {activeScreen === 'resources' && <ResourceMenu onExit={() => setCurrentMenu(null)} />}
                        {activeScreen === 'prompts' && <PromptMenu onExit={() => setCurrentMenu(null)} />}
                        {activeScreen === 'agents' && <AgentsMenu onSelect={setSelectedAgent} onExit={() => setCurrentMenu(null)} />}
                        {activeScreen === 'bug' && <BugReportCommand messages={messages} initialDescription={bugReportInitialDescription} onDone={() => setCurrentMenu(null)} />}
                        {activeScreen === 'doctor' && <Doctor onExit={() => setCurrentMenu(null)} />}
                        {activeScreen === 'compact' && <CompactCommand messages={messages} setMessages={setMessages} setIsTyping={setIsTyping} onDone={() => setCurrentMenu(null)} />}
                        {activeScreen === 'memory' && <MemoryCommand cwd={process.cwd()} onDone={() => setCurrentMenu(null)} />}
                        {activeScreen === 'cost-menu' && <CostCommand onDone={() => setCurrentMenu(null)} />}
                        {activeScreen === 'remote-env' && <RemoteEnvMenu onExit={() => setCurrentMenu(null)} />}
                        {activeScreen === 'model' && <ModelPicker initialModel={model || null} onSelect={() => setCurrentMenu(null)} onCancel={() => setCurrentMenu(null)} isStandalone={true} />}
                        {activeScreen === 'tasks' && <TaskMenu onExit={() => setTasksSelected(false)} />}
                        {activeScreen === 'loop' && <LoopMenu onExit={() => setLoopSelected(false)} />}
                        {activeScreen === 'search' && <UnifiedSearchMenu
                            history={history}
                            commands={commandRegistry.getAllCommands().map(c => ({
                                label: c.name,
                                value: c.name,
                                description: c.description
                            }))}
                            onSelect={(val) => { setCurrentMenu(null); handleSubmit(val); }}
                            onExit={() => setCurrentMenu(null)}
                        />}
                    </Box>
                )}
            </Box>

            <REPLInput onSubmit={handleSubmit} isActive={activeScreen === 'prompt' || activeScreen === 'transcript'}>
                <TerminalInput
                    value={inputValue}
                    onChange={setInputValue}
                    onSubmit={handleSubmit}
                    onExit={() => exit()}
                    history={history}
                    planMode={false}
                    agentName={selectedAgent || agent}
                />
            </REPLInput>

            <StatusLine
                vimMode={currentVimMode as any}
                vimModeEnabled={vimModeEnabled}
                model={model || ''}
                isTyping={isTyping}
                cwd={process.cwd()}
                showTasks={tasksSelected}
                showDiff={diffSelected}
                showLoop={loopSelected}
                showTeams={teamsSelected}
                usage={usage}
                planMode={appState.toolPermissionContext.mode === 'plan'}
                acceptEdits={appState.toolPermissionContext.mode === 'acceptEdits'}
                exitConfirmation={!!exitConfirmation}
            />
        </Box>
    );
};
