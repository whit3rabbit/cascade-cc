
import React from "react";
import { Box, Text } from "ink";

export interface TerminalFooterProps {
    apiKeyStatus?: string;
    debug?: boolean;
    exitMessage?: { show: boolean, message: string };
    vimMode?: boolean;
    mode?: "prompt" | "shell" | "thinking";
    autoUpdaterResult?: any;
    isAutoUpdating?: boolean;
    verbose?: boolean;
    suggestions?: any[];
    selectedSuggestion?: any;
    toolPermissionContext?: any;
    helpOpen?: boolean;
    suppressHint?: boolean;
    tasksSelected?: boolean;
    mcpClients?: any[];
}

/**
 * Terminal Footer Component (ls5)
 * Shows status, help hints, and suggestions.
 */
export function TerminalFooter(props: TerminalFooterProps) {
    const {
        suggestions = [],
        selectedSuggestion,
        helpOpen,
        suppressHint,
        mode,
        exitMessage,
        isPasting
    } = props as any;

    if (suggestions.length > 0) {
        return (
            <Box paddingX={2}>
                <Text color="cyan">Suggestions: {suggestions.length}</Text>
                {/* Real implementation would render a list of suggestions */}
            </Box>
        );
    }

    if (helpOpen) {
        return (
            <Box paddingX={2} flexDirection="column">
                <Text dimColor>! for bash mode</Text>
                <Text dimColor>/ for commands</Text>
                <Text dimColor>@ for file paths</Text>
                <Text dimColor>& for background</Text>
                <Text dimColor>double tap esc to clear input</Text>
                <Text dimColor>ctrl + o for verbose output</Text>
            </Box>
        );
    }

    return (
        <Box flexDirection="row" justifyContent="space-between" paddingX={2}>
            <Box flexDirection="column">
                {mode === "prompt" && !exitMessage?.show && !suppressHint && (
                    <Text dimColor italic>Try asking: "What can you do?" or "@file find bugs"</Text>
                )}
            </Box>
            <Box>
                {props.apiKeyStatus === "missing" && <Text color="yellow">API Key Missing</Text>}
                {props.isAutoUpdating && <Text color="cyan">Updating...</Text>}
            </Box>
        </Box>
    );
}

/**
 * Shell Details View (rl2)
 */
export function ShellDetailsView({
    shell,
    onDone,
    onKillShell
}: any) {
    return (
        <Box borderStyle="round" borderColor="cyan" flexDirection="column" paddingX={1}>
            <Text bold>Shell Details</Text>
            <Text>Status: <Text color={shell.status === "running" ? "yellow" : "green"}>{shell.status}</Text></Text>
            <Text>Command: <Text dimColor>{shell.command}</Text></Text>
            <Box marginTop={1} flexDirection="column">
                <Text bold>Output:</Text>
                <Box height={10} borderStyle="single" borderDimColor paddingX={1}>
                    <Text>{shell.output || "No output yet."}</Text>
                </Box>
            </Box>
            <Box marginTop={1}>
                <Text dimColor>Press Esc to close • k to kill</Text>
            </Box>
        </Box>
    );
}
