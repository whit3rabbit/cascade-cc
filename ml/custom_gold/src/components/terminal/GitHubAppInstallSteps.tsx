import React, { useState } from "react";
import { Box, Text, useInput } from "ink";
import ExternalLink from "./ExternalLink.js";
import { MANUAL_SETUP_URL } from "./GitHubAppInstallView.js";

export type InstallWarning = {
    title: string;
    message: string;
    instructions: string[];
};

const WORKFLOW_OPTIONS = [
    {
        value: "claude",
        label: "@Claude Code",
        description: "Tag @claude in issues and PR comments"
    },
    {
        value: "claude-review",
        label: "Claude Code Review",
        description: "Automated code review on new PRs"
    }
];

export function InstallWarningsView({
    warnings,
    onContinue
}: {
    warnings: InstallWarning[];
    onContinue: () => void;
}) {
    useInput((_input, key) => {
        if (key.return) onContinue();
    });

    return (
        <Box flexDirection="column">
            <Box flexDirection="column" borderStyle="round" borderDimColor paddingX={1}>
                <Box flexDirection="column" marginBottom={1}>
                    <Text bold>! Setup Warnings</Text>
                    <Text dimColor>We found some potential issues, but you can continue anyway</Text>
                </Box>

                {warnings.map((warning, index) => (
                    <Box key={`${warning.title}-${index}`} flexDirection="column" marginBottom={1}>
                        <Text color="warning" bold>{warning.title}</Text>
                        <Text>{warning.message}</Text>
                        {warning.instructions.length > 0 && (
                            <Box flexDirection="column" marginLeft={2} marginTop={1}>
                                {warning.instructions.map((line, lineIndex) => (
                                    <Text key={`${warning.title}-line-${lineIndex}`} dimColor>
                                        • {line}
                                    </Text>
                                ))}
                            </Box>
                        )}
                    </Box>
                ))}

                <Box marginTop={1}>
                    <Text bold color="permission">Press Enter to continue anyway, or Ctrl+C to exit and fix issues</Text>
                </Box>
                <Box marginTop={1}>
                    <Text dimColor>
                        You can also try the manual setup steps if needed:{" "}
                        <Text color="claude">{MANUAL_SETUP_URL}</Text>
                    </Text>
                </Box>
            </Box>
        </Box>
    );
}

export function WorkflowSelectionView({
    onSubmit,
    defaultSelections
}: {
    onSubmit: (selections: string[]) => void;
    defaultSelections: string[];
}) {
    const [selected, setSelected] = useState(new Set(defaultSelections));
    const [cursorIndex, setCursorIndex] = useState(0);
    const [showError, setShowError] = useState(false);

    useInput((input, key) => {
        if (key.upArrow) {
            setCursorIndex((value) => (value > 0 ? value - 1 : WORKFLOW_OPTIONS.length - 1));
            setShowError(false);
        } else if (key.downArrow) {
            setCursorIndex((value) => (value < WORKFLOW_OPTIONS.length - 1 ? value + 1 : 0));
            setShowError(false);
        } else if (input === " ") {
            const option = WORKFLOW_OPTIONS[cursorIndex];
            if (option) {
                setSelected((current) => {
                    const next = new Set(current);
                    if (next.has(option.value)) next.delete(option.value);
                    else next.add(option.value);
                    return next;
                });
            }
        } else if (key.return) {
            if (selected.size === 0) {
                setShowError(true);
                return;
            }
            onSubmit(Array.from(selected));
        }
    });

    return (
        <Box flexDirection="column">
            <Box flexDirection="column" borderStyle="round" borderDimColor paddingX={1} width="100%">
                <Box flexDirection="column" marginBottom={1}>
                    <Text bold>Select GitHub workflows to install</Text>
                    <Text dimColor>We'll create a workflow file in your repository for each one you select.</Text>
                    <Box marginTop={1}>
                        <Text dimColor>
                            More workflow examples (issue triage, CI fixes, etc.) at:{" "}
                            <ExternalLink url="https://github.com/anthropics/claude-code-action/blob/main/examples/" />
                        </Text>
                    </Box>
                </Box>

                <Box flexDirection="column" paddingX={1}>
                    {WORKFLOW_OPTIONS.map((option, index) => {
                        const isSelected = selected.has(option.value);
                        const isActive = index === cursorIndex;
                        return (
                            <Box key={option.value} flexDirection="row" marginBottom={index < WORKFLOW_OPTIONS.length - 1 ? 1 : 0}>
                                <Box marginRight={1} minWidth={2}>
                                    <Text bold={isActive}>{isSelected ? "✓" : " "}</Text>
                                </Box>
                                <Box flexDirection="column">
                                    <Text bold={isActive}>{option.label}</Text>
                                    <Text dimColor>{option.description}</Text>
                                </Box>
                            </Box>
                        );
                    })}
                </Box>
            </Box>

            <Box marginLeft={2}>
                <Text dimColor>↑↓ Navigate · Space toggle · Enter confirm</Text>
            </Box>

            {showError && (
                <Box marginLeft={1}>
                    <Text color="error">You must select at least one workflow to continue</Text>
                </Box>
            )}
        </Box>
    );
}
