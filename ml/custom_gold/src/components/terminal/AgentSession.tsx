
import React, { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { Box, Text } from 'ink';
import { useApp } from '../../hooks/useApp.js';
import { ChatFeed } from './ChatFeed.js';
import { TerminalInput } from './TerminalInput.js';
import { TerminalShell } from './TerminalShell.js';
import { AgentOutputAndToolUse } from './AgentOutputAndToolUse.js';
import { AgentMode } from '../../types/AgentTypes.js';
import { runChatSession } from '../../services/terminal/ChatSessionOrchestrator.js';
import { getGlobalState, setGlobalState, getSessionId } from '../../services/session/globalState.js';
import { logTelemetryEvent } from '../../services/telemetry/telemetryInit.js';
import { useHistory } from '../../hooks/useHistory.js';
import { saveToHistory } from '../../services/terminal/HistoryService.js';

// --- Logic from chunk_525.ts (Models & Notifications) ---

const DEPRECATED_MODELS: Record<string, { modelName: string; retirementDates: Record<string, string | null> }> = {
    "claude-3-opus": {
        modelName: "Claude 3 Opus",
        retirementDates: {
            firstParty: "January 5, 2026",
            bedrock: "January 15, 2026",
            vertex: "January 5, 2026",
            foundry: "January 5, 2026"
        }
    },
    "claude-3-haiku": { // Haiku 3.0
        modelName: "Claude 3 Haiku",
        retirementDates: {
            firstParty: "February 19, 2026",
            bedrock: null,
            vertex: null,
            foundry: null
        }
    }
};

function getModelDeprecationMessage(model: string | undefined): string | null {
    if (!model) return null;
    const lowerModel = model.toLowerCase();
    const provider = "firstParty"; // Assume first party for CLI

    for (const [key, data] of Object.entries(DEPRECATED_MODELS)) {
        const retirementDate = data.retirementDates[provider];
        if (!lowerModel.includes(key) || !retirementDate) continue;
        return `⚠ ${data.modelName} will be retired on ${retirementDate}. Consider switching to a newer model.`;
    }
    return null;
}

function useModelDeprecationCheck(model?: string) {
    const { addNotification } = useApp();
    const lastMessageRef = useRef<string | null>(null);

    useEffect(() => {
        const message = getModelDeprecationMessage(model);
        if (message && message !== lastMessageRef.current) {
            lastMessageRef.current = message;
            addNotification({
                key: "model-deprecation-warning",
                text: message,
                color: "warning",
                priority: "high"
            });
        }
        if (!message) lastMessageRef.current = null;
    }, [model, addNotification]);
}

/**
 * Check if user has a Pro/Max subscription. (DB7)
 */
async function checkSubscriptionPlan(): Promise<string | null> {
    // Logic from chunk_525:211
    // This normally calls an API. Simplified for now.
    return null;
}

function useSubscriptionNotice() {
    const { addNotification } = useApp();
    const SUBSCRIPTION_NOTICE_LIMIT = 3;

    useEffect(() => {
        const state = getGlobalState();
        if ((state as any).subscriptionNoticeCount >= SUBSCRIPTION_NOTICE_LIMIT) return;

        checkSubscriptionPlan().then((plan) => {
            if (plan) {
                setGlobalState({
                    ...state,
                    subscriptionNoticeCount: ((state as any).subscriptionNoticeCount ?? 0) + 1
                } as any);

                logTelemetryEvent("tengu_switch_to_subscription_notice_shown");

                addNotification({
                    key: "switch-to-subscription",
                    text: `Use your existing Claude ${plan} plan with Claude Code · /login to activate`,
                    color: "suggestion",
                    priority: "low"
                });
            }
        });
    }, [addNotification]);
}

function useSonnetUpdateNotice() {
    const { addNotification } = useApp();
    useEffect(() => {
        const state = getGlobalState();
        const timestamp = (state as any).sonnet45MigrationTimestamp;
        if (timestamp && Date.now() - timestamp < 3000) {
            addNotification({
                key: "sonnet-4.5-update",
                text: "Model updated to Sonnet 4.5",
                color: "suggestion",
                priority: "high",
                timeoutMs: 3000
            });
        }
    }, [addNotification]);
}

// --- Main AgentSession Component (mDA) ---

export function AgentSession(props: any) {
    const {
        commands = [],
        initialPrompt,
        initialMessages = [],
        initialTools = [],
        mcpClients = [],
        verbose,
        mainLoopModel,
        onTurnComplete,
        agents = []
    } = props;

    const { addNotification } = useApp();

    const [messages, setMessages] = useState<any[]>(initialMessages);
    const [inputValue, setInputValue] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [mode, setMode] = useState<AgentMode>("prompt");
    const [showSpinner, setShowSpinner] = useState(false);
    const [spinnerMessage, setSpinnerMessage] = useState<string | null>(null);
    const [abortController, setAbortController] = useState<AbortController | null>(null);

    // Hooks from chunk_525
    useModelDeprecationCheck(mainLoopModel);
    useSubscriptionNotice();
    useSonnetUpdateNotice();

    const history = useHistory(process.cwd());

    const getAppState = useCallback(async () => getGlobalState(), []);
    const setAppState = useCallback((update: any) => {
        if (typeof update === 'function') {
            setGlobalState(update(getGlobalState()));
        } else {
            setGlobalState({ ...getGlobalState(), ...update });
        }
    }, []);

    // Main Query Loop Logic
    const handleQuery = useCallback(async (newUserMessage: string) => {
        if (isLoading) return;

        setIsLoading(true);
        setShowSpinner(true);
        setSpinnerMessage("Thinking...");

        const abort = new AbortController();
        setAbortController(abort);

        const session = runChatSession({
            prompt: newUserMessage,
            cwd: process.cwd(),
            tools: initialTools,
            mcpClients,
            commands,
            getAppState,
            setAppState,
            abortController: abort,
            mutableMessages: messages,
            userSpecifiedModel: mainLoopModel,
            agents
        });

        try {
            for await (const event of session) {
                // The mutableMessages array is updated in place, so we trigger a re-render
                setMessages([...messages]);

                if (event.type === 'result') {
                    setIsLoading(false);
                    setShowSpinner(false);
                    setAbortController(null);
                    onTurnComplete?.();
                }

                if (event.type === 'progress' && event.message) {
                    setSpinnerMessage(event.message);
                }
            }
        } catch (err: any) {
            if (err.name !== 'AbortError') {
                addNotification({
                    key: "session-error",
                    text: `Error: ${err.message}`,
                    color: "error",
                    priority: "high"
                });
            }
            setIsLoading(false);
            setShowSpinner(false);
            setAbortController(null);
        }
    }, [isLoading, messages, initialTools, mcpClients, commands, getAppState, setAppState, mainLoopModel, agents, onTurnComplete, addNotification]);

    // Handle initial prompt
    useEffect(() => {
        if (initialPrompt) {
            handleQuery(initialPrompt);
        }
    }, []); // Only runs once on mount

    const onInputChange = useCallback((value: string) => {
        setInputValue(value);
    }, []);

    const onHistoryUp = useCallback(() => {
        const val = history.up(inputValue);
        if (val !== null) setInputValue(val);
    }, [history, inputValue]);

    const onHistoryDown = useCallback(() => {
        const val = history.down();
        if (val !== null) setInputValue(val);
    }, [history]);

    const onHistoryReset = useCallback(() => {
        history.reset();
    }, [history]);

    const onSubmit = useCallback(() => {
        if (!inputValue.trim() || isLoading) return;
        const text = inputValue;
        saveToHistory(text);
        setInputValue("");
        history.reset();
        handleQuery(text);
    }, [inputValue, isLoading, handleQuery, history]);

    const onCancel = useCallback(() => {
        if (abortController) {
            abortController.abort();
            setIsLoading(false);
            setShowSpinner(false);
            setAbortController(null);
        }
    }, [abortController]);

    return (
        <TerminalShell>
            <ChatFeed
                messages={messages}
                normalizedMessageHistory={[]}
                tools={initialTools}
                commands={commands}
                verbose={verbose}
                toolUseConfirmQueue={[]}
                inProgressToolUseIDs={new Set()}
                isMessageSelectorVisible={false}
                conversationId={getSessionId()}
                screen="chat"
                screenToggleId="1"
                streamingToolUses={[]}
                agentDefinitions={{ activeAgents: agents, allAgents: [] }}
                isLoading={isLoading}
                renderCustomMessage={(msg: any) => {
                    if (msg.type === 'progress' || msg.type === 'tool_progress') {
                        return <AgentOutputAndToolUse messages={[msg]} verbose={verbose} />;
                    }
                    return null;
                }}
            />

            <Box flexDirection="column" marginTop={1}>
                {isLoading && showSpinner && (
                    <Box marginBottom={1}>
                        <Text dimColor>{spinnerMessage || "Thinking..."}</Text>
                        <Text dimColor> (Press Esc to cancel)</Text>
                    </Box>
                )}
                <TerminalInput
                    value={inputValue}
                    onChange={onInputChange}
                    onSubmit={onSubmit}
                    isLoading={isLoading}
                    mode={mode}
                    onModeChange={setMode}
                    onHistoryUp={onHistoryUp}
                    onHistoryDown={onHistoryDown}
                    onHistoryReset={onHistoryReset}
                />
            </Box>
        </TerminalShell>
    );
}
