/**
 * File: src/components/terminal/REPL.tsx
 * Role: Main interactive UI component using Ink.
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { Box, useApp } from 'ink';
import { ConversationService } from '../../services/conversation/ConversationService.js';
import { handlePermissionResponse } from '../../services/terminal/PermissionService.js';
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
import { PermissionDialog } from '../permissions/PermissionDialog.js';
import { WorkerPermissionDialog } from '../permissions/WorkerPermissionDialog.js';
import { SandboxPermissionDialog } from '../permissions/SandboxPermissionDialog.js';
import { CostThresholdDialog } from '../permissions/CostThresholdDialog.js';
import { OnboardingWorkflow } from '../onboarding/OnboardingWorkflow.js';
import { LspRecommendationDialog } from '../onboarding/LspRecommendationDialog.js';
import { SettingsMenu } from '../menus/SettingsMenu.js';
import { McpMenu } from '../menus/McpMenu.js';
import { MarketplaceMenu } from '../menus/MarketplaceMenu.js';
import { ResourceMenu } from '../menus/ResourceMenu.js';
import { PromptMenu } from '../menus/PromptMenu.js';
import { AgentsMenu } from '../menus/AgentsMenu.js';
import { BugReportCommand } from '../../commands/BugReportCommand.js';
import { StatusLine } from './StatusLine.js';
import { ModelPicker } from '../ModelPicker/ModelPicker.js';
import { CompactCommand } from '../../commands/CompactCommand.js';
import { MemoryCommand } from '../../commands/MemoryCommand.js';
import { CostCommand } from '../../commands/CostCommand.js';
import { DoctorCommand } from '../../commands/DoctorCommand.js';
import { CORE_TOOLS } from '../../tools/index.js';
import { getAuthDetails } from '../../services/auth/AuthService.js';
import { getSettings } from '../../services/config/SettingsService.js';
import { taskManager } from '../../services/terminal/TaskManager.js';
import { costService } from '../../services/terminal/CostService.js';
import { hookService } from '../../services/hooks/HookService.js';
import { useAppState } from '../../hooks/useAppState.js';
import { useTermSize } from '../../hooks/useTermSize.js';

export interface REPLProps {
    initialPrompt?: string;
    verbose?: boolean;
    model?: string;
    agent?: string;
    isFirstRun?: boolean;
}

export const REPL: React.FC<REPLProps> = ({ initialPrompt, verbose, model, agent, isFirstRun }) => {
    const { exit } = useApp();
    const [appState, updateAppState] = useAppState();

    // Core Local UI states
    const [messages, setMessages] = useState<any[]>([]);
    const [isTyping, setIsTyping] = useState(false);
    const [inputValue, setInputValue] = useState('');
    const [screen, setScreen] = useState<'prompt' | 'transcript'>('prompt');
    const [currentMenu, setCurrentMenu] = useState<any>(null);
    const [history, setHistory] = useState<string[]>([]);
    const [planMode, setPlanMode] = useState(false);
    const [vimModeEnabled, setVimModeEnabled] = useState(false);
    const [bugReportInitialDescription, setBugReportInitialDescription] = useState('');
    const [showAllInTranscript, setShowAllInTranscript] = useState(false);

    // Permission & Modal States
    const [toolPermissions, setToolPermissions] = useState<any[]>([]);
    const [workerPermissions, setWorkerPermissions] = useState<any[]>([]);
    const [sandboxPermissions, setSandboxPermissions] = useState<any[]>([]);
    const [showCostWarning, setShowCostWarning] = useState(false);
    const [showIdeOnboarding, setShowIdeOnboarding] = useState(!getSettings().onboardingComplete);
    const [lspRecommendation, setLspRecommendation] = useState<any>(null);
    const [cost, setCost] = useState(0);
    const [usage, setUsage] = useState({ inputTokens: 0, outputTokens: 0 });
    const [mcpTools, setMcpTools] = useState<any[]>([]);
    const [subscription, setSubscription] = useState<string>('');
    const [scrollOffset, setScrollOffset] = useState(0);

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

    const handleSubmit = useCallback(async (input: string) => {
        if (isTyping || !input.trim()) return;

        if (input.startsWith('/')) {
            const handled = await SlashCommandDispatcher.handleCommand(input, {
                setMessages,
                setPlanMode,
                setVimModeEnabled,
                setShowTasks: () => { }, // placeholder
                setIsTyping,
                exit,
                cwd: process.cwd(),
                setCurrentMenu,
                setBugReportInitialDescription,
                messages
            });
            if (handled) return;
        }

        setIsTyping(true);
        setHistory(prev => [...prev, input]);
        setMessages(prev => [...prev, { role: 'user', content: input }]);

        try {
            const generator = ConversationService.startConversation(input, {
                commands: commandRegistry.getAllCommands(),
                tools: [...CORE_TOOLS, ...mcpTools],
                mcpClients: [],
                cwd: process.cwd(),
                verbose: appState.verbose,
                model,
                agent,
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
        if (showCostWarning) return 'cost';
        if (toolPermissions.length > 0) return 'tool-permission';
        if (workerPermissions.length > 0) return 'worker-permission';
        if (sandboxPermissions.length > 0) return 'sandbox-permission';
        if (showIdeOnboarding) return 'ide-onboarding';
        if (lspRecommendation) return 'lsp-recommendation';
        if (currentMenu) return currentMenu;
        return 'transcript';
    }, [showCostWarning, toolPermissions, workerPermissions, sandboxPermissions, showIdeOnboarding, lspRecommendation, currentMenu]);

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
                setScreen={setScreen as any}
                setScreenToggleId={() => { }}
                setShowAllInTranscript={() => setShowAllInTranscript(prev => !prev)}
                onEnterTranscript={() => setScreen('transcript')}
                onExitTranscript={() => setScreen('prompt')}
                todos={[]}
                agentName={agent}
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
                        {activeScreen === 'config' && <SettingsMenu onExit={() => setCurrentMenu(null)} initialTab="Config" />}
                        {activeScreen === 'mcp' && <McpMenu onExit={async () => setCurrentMenu(null)} />}
                        {activeScreen === 'marketplace' && <MarketplaceMenu onExit={() => setCurrentMenu(null)} />}
                        {activeScreen === 'resources' && <ResourceMenu onExit={() => setCurrentMenu(null)} />}
                        {activeScreen === 'prompts' && <PromptMenu onExit={() => setCurrentMenu(null)} />}
                        {activeScreen === 'agents' && <AgentsMenu onExit={() => setCurrentMenu(null)} />}
                        {activeScreen === 'bug' && <BugReportCommand messages={messages} initialDescription={bugReportInitialDescription} onDone={() => setCurrentMenu(null)} />}
                        {activeScreen === 'doctor' && <DoctorCommand onDone={() => setCurrentMenu(null)} />}
                        {activeScreen === 'compact' && <CompactCommand messages={messages} setMessages={setMessages} setIsTyping={setIsTyping} onDone={() => setCurrentMenu(null)} />}
                        {activeScreen === 'memory' && <MemoryCommand cwd={process.cwd()} onDone={() => setCurrentMenu(null)} />}
                        {activeScreen === 'cost-menu' && <CostCommand onDone={() => setCurrentMenu(null)} />}
                        {activeScreen === 'model' && <ModelPicker initialModel={model || null} onSelect={() => setCurrentMenu(null)} onCancel={() => setCurrentMenu(null)} isStandalone={true} />}
                        {activeScreen === 'search' && <UnifiedSearchMenu history={history} commands={[]} onSelect={(val) => { setCurrentMenu(null); handleSubmit(val); }} onExit={() => setCurrentMenu(null)} />}
                    </Box>
                )}
            </Box>

            <REPLInput onSubmit={handleSubmit} isActive={activeScreen === 'transcript'}>
                <TerminalInput
                    value={inputValue}
                    onChange={setInputValue}
                    onSubmit={handleSubmit}
                    onExit={() => exit()}
                    history={history}
                    planMode={false}
                    agentName={agent}
                />
            </REPLInput>

            <StatusLine
                vimMode="INSERT"
                vimModeEnabled={false}
                model={model || 'Sonnet 4.5'}
                isTyping={isTyping}
                cwd={process.cwd()}
                showTasks={false}
                usage={usage}
                planMode={false}
                exitConfirmation={false}
            />
        </Box>
    );
};
