/**
 * File: src/components/common/BashOutputComponents.tsx
 * Role: Ink components for rendering bash command output and user memory hints.
 */

import React, { useMemo } from 'react';
import { Box, Text } from 'ink';
import { sample } from 'lodash';
import { cleanCwdResetWarning } from '../../utils/shared/bashUtils.js';

interface UserMemoryInputProps {
    text: string;
    addMargin?: boolean;
}

/**
 * Displays user memory input with an icon and random feedback.
 */
export const UserMemoryInput: React.FC<UserMemoryInputProps> = ({ text, addMargin }) => {
    const displayValue = useMemo(() => {
        return text.replace(/<user-memory-input>|<\/user-memory-input>/g, "").trim();
    }, [text]);

    const feedbackMessage = useMemo(() =>
        sample(["Got it.", "Good to know.", "Noted.", "Remembered.", "Updated."]),
        []);

    if (!displayValue) return null;

    return (
        <Box flexDirection="column" marginTop={addMargin ? 1 : 0} width="100%">
            <Box>
                <Text color="blue" backgroundColor="gray"> # </Text>
                <Text color="white" backgroundColor="gray"> {displayValue} </Text>
            </Box>
            <Box height={1}>
                <Text dimColor>{feedbackMessage}</Text>
            </Box>
        </Box>
    );
};

interface OutputLineProps {
    content: string;
    verbose?: boolean;
    isError?: boolean;
}

/**
 * Renders a single line of output.
 */
export const OutputLine: React.FC<OutputLineProps> = ({ content, isError }) => {
    const trimmed = content.trim();
    if (!trimmed) return null;

    return (
        <Box>
            <Text color={isError ? "red" : undefined}>{trimmed}</Text>
        </Box>
    );
};

interface BashOutputProps {
    content: string;
    verbose?: boolean;
}

/**
 * Renders output from a bash command with stdout and stderr.
 */
export const BashOutput: React.FC<BashOutputProps> = ({ content }) => {
    const stdoutMatch = content.match(/<bash-stdout>([\s\S]*?)<\/bash-stdout>/);
    const stderrMatch = content.match(/<bash-stderr>([\s\S]*?)<\/bash-stderr>/);

    const stdout = stdoutMatch ? stdoutMatch[1] : "";
    const stderr = stderrMatch ? stderrMatch[1] : "";

    const { cleanedStderr, cwdResetWarning } = cleanCwdResetWarning(stderr);

    return (
        <Box flexDirection="column">
            {stdout && (
                <Box flexDirection="column">
                    <Text>{stdout}</Text>
                </Box>
            )}
            {cleanedStderr && (
                <Box flexDirection="column">
                    <Text color="red">{cleanedStderr}</Text>
                </Box>
            )}
            {cwdResetWarning && (
                <Box marginTop={1}>
                    <Text dimColor>{cwdResetWarning}</Text>
                </Box>
            )}
        </Box>
    );
};

interface CommandOutputProps {
    content: {
        stdout: string;
        stderr: string;
        isImage?: boolean;
        backgroundTaskId?: string;
        returnCodeInterpretation?: string;
    };
    verbose?: boolean;
}

/**
 * Unified CommandOutput component that decides how to render based on content type.
 */
export const CommandOutput: React.FC<CommandOutputProps> = ({ content }) => {
    const { stdout, stderr, isImage, backgroundTaskId, returnCodeInterpretation } = content;
    const { cleanedStderr, cwdResetWarning } = cleanCwdResetWarning(stderr);

    if (isImage) {
        return (
            <Box height={1}>
                <Text dimColor>[Image data detected and sent to Claude]</Text>
            </Box>
        );
    }

    const hasOutput = stdout.trim() !== "" || cleanedStderr.trim() !== "";

    return (
        <Box flexDirection="column">
            {stdout.trim() !== "" && (
                <Box marginBottom={cleanedStderr.trim() !== "" ? 1 : 0}>
                    <Text>{stdout}</Text>
                </Box>
            )}
            {cleanedStderr.trim() !== "" && (
                <Box>
                    <Text color="red">{cleanedStderr}</Text>
                </Box>
            )}
            {cwdResetWarning && (
                <Box marginTop={hasOutput ? 1 : 0}>
                    <Text dimColor>{cwdResetWarning}</Text>
                </Box>
            )}
            {!hasOutput && !cwdResetWarning && backgroundTaskId && (
                <Box height={1}>
                    <Text dimColor>
                        Running in the background (ID: {backgroundTaskId})
                    </Text>
                </Box>
            )}
            {!hasOutput && !cwdResetWarning && !backgroundTaskId && returnCodeInterpretation && (
                <Box height={1}>
                    <Text dimColor>{returnCodeInterpretation}</Text>
                </Box>
            )}
        </Box>
    );
};
