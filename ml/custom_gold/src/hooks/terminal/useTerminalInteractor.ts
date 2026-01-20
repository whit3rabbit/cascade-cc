// Logic from chunk_525.ts (Terminal Hook & Usage Notices)

import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { useNotifications } from '../../services/terminal/NotificationService.js';
import { useAppState } from '../../contexts/AppStateContext.js';
import { handleSubmission } from '../../services/terminal/SubmissionPipeline.js';
// import { runChatSession } from '../../services/terminal/ChatSessionOrchestrator.js'; // Used within SubmissionPipeline or here?
import { useIdeSelection } from '../../services/terminal/ChatSessionOrchestrator.js';

// --- Constants ---

const RETIREMENT_DATES: Record<string, { modelName: string, retirementDates: any }> = {
    "claude-3-opus": {
        modelName: "Claude 3 Opus",
        retirementDates: {
            firstParty: "January 5, 2026",
            bedrock: "January 15, 2026",
            vertex: "January 5, 2026",
            foundry: "January 5, 2026"
        }
    },
    "claude-3-7-sonnet": {
        modelName: "Claude 3.7 Sonnet",
        retirementDates: {
            firstParty: "February 10, 2026",
            bedrock: "April 28, 2026",
            vertex: "May 11, 2026",
            foundry: "February 10, 2026"
        }
    },
    "claude-3-5-haiku": {
        modelName: "Claude 3.5 Haiku",
        retirementDates: {
            firstParty: "February 19, 2026",
            bedrock: null,
            vertex: null,
            foundry: null
        }
    }
};

const SUBSCRIPTION_NOTICE_LIMIT = 3;

// --- Helper Hooks ---

function useModelDeprecationCheck(model: string) {
    const { addNotification } = useNotifications();
    const lastWarningRef = useRef<string | null>(null);

    useEffect(() => {
        if (!model) return;
        const entry = RETIREMENT_DATES[model];
        if (!entry) {
            lastWarningRef.current = null;
            return;
        }

        // Simplified provider check - assuming firstParty for generic CLI usage
        const date = entry.retirementDates.firstParty;
        if (!date) return;

        const message = `⚠ ${entry.modelName} will be retired on ${date}. Consider switching to a newer model.`;
        if (message !== lastWarningRef.current) {
            lastWarningRef.current = message;
            addNotification({
                key: "model-deprecation-warning",
                text: message,
                priority: "high",
                color: "warning"
            });
        }
    }, [model, addNotification]);
}

function useRateLimitCheck(context: any) { // context from rate limit provider
    const { addNotification } = useNotifications();
    // Simplified stub for rate limit context usage
    // logic would go here
}

function useIdeStatusCheck(ideSelection: any, mcpClients: any, ideInstallationStatus: any) {
    const { addNotification } = useNotifications();
    // Simplified status check
    // Real implementation would parse 'mcpClients' to see if 'ide' is connected
}

function useSonnetUpdateNotice() {
    const { addNotification } = useNotifications();
    // Logic to check settings for migration timestamp
}

function useSubscriptionNotice() {
    const { addNotification } = useNotifications();
    // Logic to check account tier and show upsell
}

// --- Main Hook ---

export function useTerminalInteractor({
    initialMessages = [],
    initialPrompt,
    commands,
    mcpClients,
    // ... other props
}: any) {
    const [messages, setMessages] = useState<any[]>(initialMessages);
    const [isLoading, setIsLoading] = useState(false);
    const [mode, setMode] = useState("prompt");
    const [inputValue, setInputValue] = useState(initialPrompt || "");
    const [isThinking, setIsThinking] = useState(false);

    // State integration
    const [appState, setAppState] = useAppState();
    const { addNotification } = useNotifications();

    // Helper hooks
    useModelDeprecationCheck(appState.model || "claude-3-5-sonnet"); // Example model usage
    useSonnetUpdateNotice();
    useSubscriptionNotice();

    // IDE Integration
    const [ideSelection, setIdeSelection] = useState<any>(undefined);
    useIdeSelection(mcpClients, setIdeSelection);

    const submitQuery = useCallback(async (input: string, modeOverride?: string) => {
        setIsLoading(true);
        const currentMode = modeOverride || mode;

        try {
            await handleSubmission({
                input,
                mode: currentMode,
                messages,
                onQuery: async (processedMessages: any) => {
                    // This is where we would call ChatSessionOrchestrator to run the loop
                    // For now, updating messages to simulate local echo + processing
                    setMessages(prev => [...processedMessages]);

                    // The actual runChatSession integration would happen here
                    // yielding results back to messages
                }
            });
        } catch (err) {
            console.error(err);
            addNotification({
                key: "submission-error",
                text: "Error submitting query",
                priority: "high",
                color: "error"
            });
        } finally {
            setIsLoading(false);
        }
    }, [mode, messages, addNotification]);

    const handleInputChange = useCallback((val: string) => {
        setInputValue(val);
    }, []);

    return {
        messages,
        setMessages,
        isLoading,
        setIsLoading,
        mode,
        setMode,
        inputValue,
        setInputValue: handleInputChange,
        submitQuery,
        ideSelection
    };
}
