/**
 * File: src/components/terminal/REPL.tsx
 * Role: Main interactive UI component using Ink.
 */

import React, { useState, useEffect } from 'react';
import { Box, Text, useApp } from 'ink';
import { ConversationService } from '../../services/conversation/ConversationService.js';
import { UserPromptMessage } from '../messages/UserPromptMessage.js';
import { MessageHistory } from '../MessageHistory.js';

export interface REPLProps {
    initialPrompt?: string;
    verbose?: boolean;
    model?: string;
    agent?: string;
}

/**
 * REPL component for interactive Claude sessions.
 */
import { PermissionDialog } from '../permissions/PermissionDialog.js';
import { WorkerPermissionDialog } from '../permissions/WorkerPermissionDialog.js';
import { SandboxPermissionDialog } from '../permissions/SandboxPermissionDialog.js';
import { CostThresholdDialog } from '../permissions/CostThresholdDialog.js';
import { IdeOnboardingDialog } from '../onboarding/IdeOnboardingDialog.js';
import { LspRecommendationDialog } from '../onboarding/LspRecommendationDialog.js';
import { TaskList } from '../TaskList.js';
import { StatusLine } from './StatusLine.js';
import Spinner from 'ink-spinner';
import { DocumentationService } from '../../services/documentation/DocumentationService.js';
import { useInput } from 'ink';
import { SlashCommandDispatcher } from '../../commands/SlashCommandDispatcher.js';
import { Logo } from './Logo.js';

export const REPL: React.FC<REPLProps> = ({ initialPrompt, verbose, model, agent }) => {
    const { exit } = useApp();
    const [messages, setMessages] = useState<any[]>([]);
    const [isTyping, setIsTyping] = useState(false);
    const [history, setHistory] = useState<string[]>([]);
    const [tasks, setTasks] = useState<string[]>([]);
    const [showTasks, setShowTasks] = useState(false);
    const [vimModeEnabled, setVimModeEnabled] = useState(false);
    const [vimMode, setVimMode] = useState<'NORMAL' | 'INSERT'>('INSERT');
    const [planMode, setPlanMode] = useState(false);

    // Permission & Modal States
    const [toolPermissions, setToolPermissions] = useState<any[]>([]);
    const [workerPermissions, setWorkerPermissions] = useState<any[]>([]);
    const [sandboxPermissions, setSandboxPermissions] = useState<any[]>([]);
    const [showCostWarning, setShowCostWarning] = useState(false);
    const [showIdeOnboarding, setShowIdeOnboarding] = useState(false);
    const [lspRecommendation, setLspRecommendation] = useState<any>(null);

    // Global Shortcuts
    useInput((input, key) => {
        if (input === 't' && key.ctrl) {
            setShowTasks(prev => !prev);
        }
        if (input === 'l' && key.ctrl) {
            setMessages([]);
        }
        if (input === 'd' && key.ctrl) {
            exit();
        }
        if (input === 'b' && key.ctrl) {
            // Toggle background tasks view or move task to background
            // For now just toggle tasks
            setShowTasks(prev => !prev);
        }
    });

    // Logo on mount - we now render this conditionally in the JSX
    useEffect(() => {
        if (messages.length === 0 && !initialPrompt) {
            // No initial message needed if we render the Logo component explicitly
        }
    }, [initialPrompt]);

    // Determine active screen (Priority Order)
    const getActiveScreen = () => {
        if (showCostWarning) return 'cost';
        if (toolPermissions.length > 0) return 'tool-permission';
        if (workerPermissions.length > 0) return 'worker-permission';
        if (sandboxPermissions.length > 0) return 'sandbox-permission';
        if (showIdeOnboarding) return 'ide-onboarding';
        if (lspRecommendation) return 'lsp-recommendation';
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
        // Use external dispatcher for scalability
        return await SlashCommandDispatcher.handleCommand(input, {
            setMessages,
            setPlanMode,
            setVimModeEnabled,
            setShowTasks,
            setIsTyping,
            exit,
            cwd: process.cwd()
        });
    };

    const handleSubmit = async (input: string) => {
        setHistory(prev => [...prev, input]);

        if (input.startsWith('/')) {
            const handled = await handleCommand(input);
            if (handled) return;
        }

        setMessages(prev => [...prev, { role: 'user', content: input }]);
        setIsTyping(true);

        try {
            // Create a generator for the new turn
            const generator = ConversationService.startConversation(input, {
                commands: [],
                tools: [],
                mcpClients: [],
                cwd: process.cwd(),
                verbose,
                model,
                agent,
                // Pass plan mode context if supported
            });

            for await (const chunk of generator) {
                if (chunk.type === 'assistant') {
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
                } else if (chunk.type === 'result') {
                    // Final result
                    if (chunk.subtype === 'success') {
                        // maybe add a system message or just stop typing
                    }
                }
            }
        } catch (error) {
            setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${error}` }]);
        } finally {
            setIsTyping(false);
        }
    };

    return (
        <Box flexDirection="column" padding={0} width="100%" height="100%">
            {/* Header/Title Area */}
            <Box paddingX={1} marginBottom={0} justifyContent="space-between" borderStyle="single" borderBottom={true} borderTop={false} borderLeft={false} borderRight={false} borderColor="gray">
                <Text color="cyan" bold>Claude Code</Text>
                <Text color="gray">{planMode ? 'PLANNING MODE' : 'INTERACTIVE'}</Text>
            </Box>

            {showTasks && <TaskList tasks={tasks} />}

            {messages.length === 0 && (
                <Box flexDirection="column" paddingX={1}>
                    <Logo version="2.1.23" model={model || "3.5"} cwd={process.cwd()} />
                    <Box marginTop={1} paddingBottom={1} borderStyle="single" borderTop={true} borderBottom={true} borderLeft={false} borderRight={false} borderColor="gray">
                        <Text dimColor>❯ Try "fix lint errors"</Text>
                    </Box>
                    <Text dimColor>  ? for shortcuts</Text>
                </Box>
            )}

            <Box flexGrow={1} flexDirection="column" overflowY="hidden">
                <MessageHistory messages={messages} />
            </Box>

            {isTyping && (
                <Box paddingX={1}>
                    <Text color="green">
                        <Spinner type="dots" /> Claude is thinking...
                    </Text>
                </Box>
            )}

            {/* Render conditional screens */}
            {activeScreen === 'cost' && (
                <CostThresholdDialog
                    onApprove={() => setShowCostWarning(false)}
                />
            )}

            {activeScreen === 'tool-permission' && (
                <PermissionDialog
                    toolUseConfirm={toolPermissions[0]}
                    onDone={() => setToolPermissions(prev => prev.slice(1))}
                    onReject={() => setToolPermissions(prev => prev.slice(1))}
                    parseInput={(input) => input}
                />
            )}

            {activeScreen === 'worker-permission' && (
                <WorkerPermissionDialog
                    request={workerPermissions[0]}
                />
            )}

            {activeScreen === 'sandbox-permission' && (
                <SandboxPermissionDialog
                    hostPattern={sandboxPermissions[0]}
                />
            )}

            {activeScreen === 'ide-onboarding' && (
                <IdeOnboardingDialog
                    onDone={() => setShowIdeOnboarding(false)}
                />
            )}

            {activeScreen === 'lsp-recommendation' && (
                <LspRecommendationDialog
                    recommendation={lspRecommendation}
                    onAccept={() => setLspRecommendation(null)}
                />
            )}

            {/* Status Line */}
            <StatusLine
                vimMode={vimMode}
                vimModeEnabled={vimModeEnabled}
                model={model || 'default'}
                isTyping={isTyping}
                cwd={process.cwd()}
                showTasks={showTasks}
            />

            {activeScreen === 'transcript' && (
                <Box marginTop={0} paddingX={1}>
                    <UserPromptMessage
                        onSubmit={handleSubmit}
                        onClear={() => setMessages([])}
                        history={history}
                        vimModeEnabled={vimModeEnabled}
                        onVimModeChange={setVimMode}
                    />
                </Box>
            )}
        </Box>
    );
};
