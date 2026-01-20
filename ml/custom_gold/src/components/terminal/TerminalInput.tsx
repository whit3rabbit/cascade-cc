
import React from "react";
import { Box, Text, useInput } from "ink";
import { useStandardTerminalInput } from "../../services/terminal/StandardInputService.js";
import { useVimTerminalInput } from "../../services/terminal/VimInputService.js";
import { usePasteHandler } from "../../services/terminal/PasteService.js";

export interface TerminalInputProps {
    value: string;
    onChange: (value: string) => void;
    onSubmit?: (value: string) => void;
    onExit?: () => void;
    onExitMessage?: (message: string, source?: string) => void;
    placeholder?: string;
    showCursor?: boolean;
    mask?: string;
    vimMode?: boolean;
    columns?: number;
    onImagePaste?: (base64: string, mediaType: string) => void;
    cursorOffset?: number;
    onChangeCursorOffset?: (offset: number) => void;
    isLoading?: boolean;
    mode?: any; // Relaxed type to match AgentMode
    onModeChange?: (mode: any) => void;
    onHistoryUp?: () => void;
    onHistoryDown?: () => void;
    onHistoryReset?: () => void;
    argumentHint?: string;
}

/**
 * Real Terminal Input Component (C6)
 * Supports both Standard (Emacs) and Vim modes.
 */
export function TerminalInput(props: TerminalInputProps) {
    const {
        value,
        onChange,
        placeholder = "",
        vimMode = false,
        onImagePaste,
        onChangeCursorOffset,
        argumentHint
    } = props;

    // Standard input hook
    const standardInput = useStandardTerminalInput({
        ...props,
        columns: props.columns ?? 80,
        onOffsetChange: onChangeCursorOffset
    });

    // Vim input hook (wraps standard)
    const vimInput = useVimTerminalInput({
        ...props,
        columns: props.columns ?? 80,
        onOffsetChange: onChangeCursorOffset
    });

    // Paste handler
    const pasteHandler = usePasteHandler({
        onPaste: (text: string) => onChange(value + text),
        onImagePaste
    });

    const activeInput = vimMode ? vimInput : standardInput;

    useInput((input, key) => {
        // First, check if it's a paste event
        if (pasteHandler.handleInput(input, key)) return;

        // Otherwise, pass to active input handler
        activeInput.onInput(input, key);
    });

    // Render logic
    const rendered = activeInput.renderedValue;

    return (
        <Box flexDirection="column">
            <Box>
                {value.length === 0 && placeholder ? (
                    <Text dimColor>{placeholder}</Text>
                ) : (
                    <Text>
                        {rendered}
                        {argumentHint && <Text dimColor>{argumentHint}</Text>}
                    </Text>
                )}
            </Box>
        </Box>
    );
}

/**
 * Interactive Option View (aKA)
 * Used in wizards and settings.
 */
export function InputOptionView({
    label,
    value,
    onChange,
    isFocused,
    description,
    placeholder,
    index
}: any) {
    return (
        <Box flexDirection="column" marginBottom={1}>
            <Box flexDirection="row">
                {index !== undefined && <Text dimColor>{index}. </Text>}
                <Text bold color={isFocused ? "cyan" : undefined}>{label}: </Text>
                {isFocused ? (
                    <TerminalInput
                        value={value}
                        onChange={onChange}
                        placeholder={placeholder}
                        showCursor={true}
                    />
                ) : (
                    <Text color={value ? undefined : "gray"}>{value || placeholder || "None"}</Text>
                )}
            </Box>
            {description && (
                <Box marginLeft={3}>
                    <Text dimColor italic>{description}</Text>
                </Box>
            )}
        </Box>
    );
}
