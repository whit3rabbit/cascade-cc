/**
 * File: src/hooks/usePermissionContext.ts
 * Role: Hook for managing the state and interactions of a tool permission confirmation dialog.
 */

import { useState, useCallback, Dispatch, SetStateAction } from 'react';
import { track } from '../services/telemetry/Telemetry.js';

export interface ToolUseConfirm {
    tool: { name: string };
    assistantMessage?: { id: string };
    input: any;
    onAllow: (input: any, options: any[], feedback?: string) => void;
    onReject: (feedback?: string) => void;
}

export interface UsePermissionContextProps {
    toolUseConfirm: ToolUseConfirm;
    onDone: () => void;
    onReject?: () => void;
    filePath?: string | null;
    operationType?: string;
}

export interface UsePermissionContextResult {
    acceptFeedback: string;
    setAcceptFeedback: Dispatch<SetStateAction<string>>;
    rejectFeedback: string;
    setRejectFeedback: Dispatch<SetStateAction<string>>;
    focusedOption: string;
    setFocusedOption: Dispatch<SetStateAction<string>>;
    isAcceptInputMode: boolean;
    isRejectInputMode: boolean;
    toggleInputMode: (mode: "accept" | "reject") => void;
    handleOptionSelect: (option: "yes" | "no", feedback?: string) => void;
}

/**
 * Hook to manage tool permission confirmation state.
 */
export function usePermissionContext({
    toolUseConfirm,
    onDone,
    onReject,
    filePath = null,
    operationType = "write"
}: UsePermissionContextProps): UsePermissionContextResult {
    const [acceptFeedback, setAcceptFeedback] = useState<string>("");
    const [rejectFeedback, setRejectFeedback] = useState<string>("");
    const [focusedOption, setFocusedOption] = useState<string>("yes");
    const [isAcceptInputMode, setIsAcceptInputMode] = useState<boolean>(false);
    const [isRejectInputMode, setIsRejectInputMode] = useState<boolean>(false);

    const handleOptionSelect = useCallback((option: "yes" | "no", feedback: string = "") => {
        const metadata = {
            toolName: toolUseConfirm.tool.name,
            messageId: toolUseConfirm.assistantMessage?.id,
            filePath,
            operationType,
            hasFeedback: !!feedback.trim()
        };

        if (option === "yes") {
            track("permission_accepted", metadata);
            toolUseConfirm.onAllow(toolUseConfirm.input, [], feedback.trim() || undefined);
            onDone();
        } else if (option === "no") {
            track("permission_rejected", metadata);
            toolUseConfirm.onReject(feedback.trim() || undefined);
            onReject?.();
            onDone();
        }
    }, [toolUseConfirm, filePath, operationType, onDone, onReject]);

    const toggleInputMode = useCallback((mode: "accept" | "reject") => {
        if (mode === "accept") {
            setIsAcceptInputMode(prev => !prev);
            setIsRejectInputMode(false);
        } else {
            setIsRejectInputMode(prev => !prev);
            setIsAcceptInputMode(false);
        }
    }, []);

    return {
        acceptFeedback,
        setAcceptFeedback,
        rejectFeedback,
        setRejectFeedback,
        focusedOption,
        setFocusedOption,
        isAcceptInputMode,
        isRejectInputMode,
        toggleInputMode,
        handleOptionSelect
    };
}
