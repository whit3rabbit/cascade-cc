
// Logic from chunk_527.ts (Git Policy & Shell UI)

import React from "react";
import { Box, Text, useInput } from "ink";
import { getBashTimeout } from "../../utils/terminal/BashConfig.js";
import { Shortcut } from "../../components/shared/Shortcut.js";
import { Indent } from "../../components/shared/Indent.js";
import { OutputBlock } from "../../components/shared/OutputBlock.js";
import { formatDuration } from "../../utils/shared/formatUtils.js";
import { stripUnderline } from "./terminalUtils.js";

// --- Git Policy Prompt (jB7) ---
export function getGitPolicyPrompt() {
    return `
# Committing changes with git

Only create commits when requested by the user. 

Git Safety Protocol:
- NEVER update the git config.
- NEVER run destructive/irreversible commands (push --force, hard reset) to main.
- Avoid git commit --amend unless explicitly asked and commit was local.

PR Instructions:
- Use 'gh' command for GitHub tasks.
- Draft summaries and test plans.
- Pass commit messages via HEREDOC for formatting.
`;
}

const MAX_SUMMARY_LINES = 2;
const MAX_SUMMARY_CHARS = 160;

// --- Command Renderer (g99) ---
export function formatCommandForDisplay(command: string, verbose: boolean = false) {
    if (!command) return "";
    let display = command;

    if (command.includes("\"$(cat <<'EOF'")) {
        const match = command.match(/^(.*?)"?\$\(cat <<'EOF'\n([\s\S]*?)\n\s*EOF\n\s*\)"(.*)$/);
        if (match && match[1] && match[2]) {
            display = `${match[1].trim()} "${match[2].trim()}"${(match[3] || "").trim()}`;
        }
    }

    if (!verbose) {
        const lines = display.split("\n");
        const hasTooManyLines = lines.length > MAX_SUMMARY_LINES;
        const hasTooManyChars = display.length > MAX_SUMMARY_CHARS;

        if (hasTooManyLines || hasTooManyChars) {
            let trimmed = display;
            if (hasTooManyLines) trimmed = lines.slice(0, MAX_SUMMARY_LINES).join("\n");
            if (trimmed.length > MAX_SUMMARY_CHARS) trimmed = trimmed.slice(0, MAX_SUMMARY_CHARS);
            return `${trimmed.trim()}…`;
        }
    }

    return display;
}

// --- Background Shortcut Hint (yV1) ---
export function BackgroundShortcutHint({ onBackground }: { onBackground: () => void }) {
    useInput((input, key) => {
        if (input === "b" && key.ctrl) onBackground();
    });

    const shortcut = process.env.TERM_PROGRAM === "tmux" ? "ctrl+b ctrl+b" : "ctrl+b";

    return (
        <Box paddingLeft={5}>
            <Text dimColor>
                <Shortcut shortcut={shortcut} action="run in background" />
            </Text>
        </Box>
    );
}

// --- Timeout Indicator (u99) ---
export function TimeoutIndicator({ timeout }: { timeout?: number }) {
    if (!timeout) return null;
    const defaultTimeout = getBashTimeout();
    if (timeout === defaultTimeout) return null;

    return (
        <Box flexWrap="nowrap" marginLeft={1}>
            <Text dimColor>timeout: {formatDuration(timeout)}</Text>
        </Box>
    );
}

// --- Tool Use Rejected View (m99) ---
export function ToolUseRejectedView() {
    return (
        <Indent>
            <Text color="error">Interrupted </Text>
            <Text dimColor>· What should Claude do instead?</Text>
        </Indent>
    );
}

// --- Execution Progress View (d99) ---
export function ExecutionProgressView({
    output,
    fullOutput,
    elapsedTimeSeconds,
    totalLines,
    verbose
}: {
    output: string;
    fullOutput: string;
    elapsedTimeSeconds?: number;
    totalLines?: number;
    verbose?: boolean;
}) {
    const trimmedFull = stripUnderline(fullOutput.trim());
    const recentLines = stripUnderline(output.trim())
        .split("\n")
        .filter(Boolean);
    const display = verbose ? trimmedFull : recentLines.slice(-5).join("\n");
    const hiddenCount = verbose ? 0 : totalLines ? Math.max(0, totalLines - 5) : 0;
    const durationLabel = elapsedTimeSeconds !== undefined ? `(${formatDuration(elapsedTimeSeconds * 1000)})` : undefined;

    if (!recentLines.length) {
        return (
            <Indent>
                <Text dimColor>Running… {durationLabel}</Text>
            </Indent>
        );
    }

    return (
        <Indent>
            <Box flexDirection="column">
                <Box height={verbose ? undefined : Math.min(5, recentLines.length)} flexDirection="column" overflow="hidden">
                    <Text dimColor>{display}</Text>
                </Box>
                <Box flexDirection="row" gap={1}>
                    {!verbose && hiddenCount > 0 && (
                        <Text dimColor>{`+${hiddenCount} more line${hiddenCount === 1 ? "" : "s"}`}</Text>
                    )}
                    {durationLabel && <Text dimColor>{durationLabel}</Text>}
                </Box>
            </Box>
        </Indent>
    );
}

// --- Queued View (p99) ---
export function WaitingView() {
    return (
        <Indent>
            <Text dimColor>Waiting…</Text>
        </Indent>
    );
}

// --- Tool Result View (c99) ---
export function ToolResultView({ content, verbose }: { content: string; verbose?: boolean }) {
    return <OutputBlock content={content} verbose={verbose} />;
}

// --- Telemetry Tracker (_q0) ---
export function trackAgentMetrics(sessionId: string, stats: any) {
    console.log(`[Telemetry] Session ${sessionId} completed: ${stats.tokens} tokens, ${stats.tools} tool calls.`);
}
