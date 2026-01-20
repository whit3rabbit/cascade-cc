
// Logic from chunk_545.ts (GitHub App Setup & Init Command)

import React, { useState } from "react";
import { Box, Text, useInput, useStdout } from "ink";
import InkTextInput from "ink-text-input";

const TextInput = InkTextInput as unknown as React.FC<{
    value: string;
    onChange: (value: string) => void;
    onSubmit?: (value: string) => void;
    placeholder?: string;
    focus?: boolean;
    showCursor?: boolean;
    columns?: number;
    cursorOffset?: number;
    onChangeCursorOffset?: (offset: number) => void;
}>;

// --- GitHub CLI Check View (_59) ---
export function GitHubCliCheckView() {
    return <Text>Checking GitHub CLI installation…</Text>;
}

// --- GitHub Repo Selection (T59) ---
export function GitHubRepoSelection({
    currentRepo,
    useCurrentRepo,
    repoUrl,
    onRepoUrlChange,
    onSubmit,
    onToggleUseCurrentRepo
}: any) {
    const [cursorOffset, setCursorOffset] = useState(0);
    const [error, setError] = useState(false);
    const { stdout } = useStdout();
    const columns = stdout?.columns || 80;

    const handleContinue = () => {
        const targetRepo = useCurrentRepo ? currentRepo : repoUrl;
        if (!targetRepo?.trim()) {
            setError(true);
            return;
        }
        onSubmit();
    };

    useInput((_input, key) => {
        if (key.upArrow) {
            onToggleUseCurrentRepo(true);
            setError(false);
        } else if (key.downArrow) {
            onToggleUseCurrentRepo(false);
            setError(false);
        } else if (key.return) {
            handleContinue();
        }
    });

    return (
        <Box flexDirection="column">
            <Box flexDirection="column" borderStyle="round" borderDimColor paddingX={1}>
                <Box flexDirection="column" marginBottom={1}>
                    <Text bold>Install GitHub App</Text>
                    <Text dimColor>Select GitHub repository</Text>
                </Box>

                {currentRepo && (
                    <Box marginBottom={1}>
                        <Text bold={useCurrentRepo} color={useCurrentRepo ? "permission" : undefined}>
                            {useCurrentRepo ? "> " : "  "}Use current repository: {currentRepo}
                        </Text>
                    </Box>
                )}

                <Box marginBottom={1}>
                    <Text bold={!useCurrentRepo} color={!useCurrentRepo ? "permission" : undefined}>
                        {!useCurrentRepo ? "> " : "  "}{currentRepo ? "Enter a different repository" : "Enter repository"}
                    </Text>
                </Box>

                {(!useCurrentRepo || !currentRepo) && (
                    <Box marginLeft={2} marginBottom={1}>
                        <TextInput
                            value={repoUrl}
                            onChange={(val) => { onRepoUrlChange(val); setError(false); }}
                            onSubmit={handleContinue}
                            focus={true}
                            placeholder="Enter a repo as owner/repo or https://github.com/owner/repo…"
                            showCursor={true}
                            columns={columns}
                            cursorOffset={cursorOffset}
                            onChangeCursorOffset={setCursorOffset}
                        />
                    </Box>
                )}
            </Box>

            {error && (
                <Box marginLeft={3} marginBottom={1}>
                    <Text color="error">Please enter a repository name to continue</Text>
                </Box>
            )}

            <Box marginLeft={3}>
                <Text dimColor>
                    {currentRepo ? "↑/↓ to select · " : ""}Enter to continue
                </Text>
            </Box>
        </Box>
    );
}

// --- Init Command Configuration (o47) ---
export const InitCommand = {
    name: "init",
    description: "Initialize a new CLAUDE.md file with codebase documentation",
    progressMessage: "analyzing your codebase",
    async getPrompt() {
        return `Please analyze this codebase and create a CLAUDE.md file, which will be given to future instances of Claude Code to operate in this repository.

What to add:
1. Commands that will be commonly used, such as how to build, lint, and run tests. Include the necessary commands to develop in this codebase, such as how to run a single test.
2. High-level code architecture and structure so that future instances can be productive more quickly. Focus on the "big picture" architecture that requires reading multiple files to understand.

Usage notes:
- If there's already a CLAUDE.md, suggest improvements to it.
- When you make the initial CLAUDE.md, do not repeat yourself and do not include obvious instructions like "Provide helpful error messages to users", "Write unit tests for all new utilities", "Never include sensitive information (API keys, tokens) in code or commits".
- Avoid listing every component or file structure that can be easily discovered.
- Don't include generic development practices.
- If there are Cursor rules (in .cursor/rules/ or .cursorrules) or Copilot rules (in .github/copilot-instructions.md), make sure to include the important parts.
- If there is a README.md, make sure to include the important parts.
- Do not make up information such as "Common Development Tasks", "Tips for Development", "Support and Documentation" unless this is expressly included in other files that you read.
- Be sure to prefix the file with the following text:

\`\`\`
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
\`\`\``;
    }
};
