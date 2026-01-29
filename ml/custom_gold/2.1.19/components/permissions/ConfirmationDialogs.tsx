// @ts-nocheck
/**
 * File: src/components/permissions/ConfirmationDialogs.tsx
 * Role: React components for rendering tool permission confirmation dialogs (Bash, MCP, etc.).
 */

import React from 'react';
import { Box, Text } from 'ink';
import { Card } from '../common/Card.js';
import { usePermissionContext } from '../../hooks/usePermissionContext.js';

interface ToolUseConfirm {
    tool: {
        name: string;
    };
    assistantMessage?: { id: string };
    description?: string;
    input: any;
    onAllow: (input: any, options: any[], feedback?: string) => void;
    onReject: (feedback?: string) => void;
}

interface ToolConfirmationDialogProps {
    toolUseConfirm: ToolUseConfirm;
    onDone: (decision: any) => void;
    onReject: (reason?: string) => void;
}

/**
 * Renders a confirmation dialog for a generic tool or MCP call.
 */
export function ToolConfirmationDialog({
    toolUseConfirm,
    onDone,
    onReject
}: ToolConfirmationDialogProps) {
    const {
        focusedOption,
        // setFocusedOption,
        // handleOptionSelect,
        isRejectInputMode
    } = usePermissionContext({ toolUseConfirm, onDone, onReject });

    return (
        <Card
            title="Tool Permission Request"
            subtitle={toolUseConfirm.description}
            borderColor="yellow"
        >
            <Box flexDirection="column">
                <Text color="cyan">{toolUseConfirm.tool.name}({JSON.stringify(toolUseConfirm.input)})</Text>

                <Box marginTop={1}>
                    <Text color={focusedOption === "yes" ? "green" : "white"}>[ Allow ]</Text>
                    <Text>   </Text>
                    <Text color={focusedOption === "no" ? "red" : "white"}>[ Deny ]</Text>
                </Box>

                {isRejectInputMode && (
                    <Box marginTop={1}>
                        <Text dimColor>Reason for rejection: </Text>
                        <Text italic>(Type your reason and press Enter)</Text>
                    </Box>
                )}
            </Box>
        </Card>
    );
}

interface BashConfirmationDialogProps {
    command: string;
    toolUseConfirm: ToolUseConfirm;
    onDone: (decision: any) => void;
    onReject: (reason?: string) => void;
}

/**
 * Specialized confirmation for Bash commands with a focused code display.
 */
export function BashConfirmationDialog({
    command,
    // toolUseConfirm,
    // onDone,
    // onReject
}: BashConfirmationDialogProps) {
    return (
        <Card title="Security Warning: Bash Command" borderColor="red">
            <Box flexDirection="column">
                <Box padding={1} backgroundColor="gray">
                    <Text color="white">$ {command}</Text>
                </Box>
                <Box marginTop={1}>
                    <Text dimColor>Running arbitrary shell commands can be dangerous. Proceed?</Text>
                </Box>

                <Box marginTop={1}>
                    <Text backgroundColor="green" color="black"> [Y]es </Text>
                    <Box marginLeft={2}>
                        <Text backgroundColor="red" color="white"> [N]o </Text>
                    </Box>
                </Box>
            </Box>
        </Card>
    );
}
