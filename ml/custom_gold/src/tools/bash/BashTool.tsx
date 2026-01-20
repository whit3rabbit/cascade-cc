import React, { useCallback, useMemo, useState } from "react";
import { Box, Text, useInput } from "ink";
import { Select } from "@inkjs/ui";
import { z } from "zod";
import fs from "node:fs/promises";
import path from "node:path";
import os from "node:os";

import { OutputBuffer } from "../../utils/shared/processUtils.js";
import { getSettings } from "../../services/terminal/settings.js";
import { logTelemetryEvent } from "../../services/telemetry/telemetryInit.js";
import { sandboxService } from "../../services/sandbox/sandboxService.js";
import { runBashCommand } from "../../services/terminal/BashExecutor.js";
import { LocalBashTask } from "../../services/tasks/TaskTypes.js";
import {
    evaluateBashCommandSafety,
    summarizeBashOutput,
    SEARCH_COMMANDS,
    READ_COMMANDS,
    BACKGROUND_EXCLUDED_COMMANDS
} from "../../services/bash/BashSafetyService.js";
import { getOriginalCwd } from "../../services/session/sessionStore.js";
import { FileReadTool } from "../definitions/fileRead.js";
import { useCtrlExit } from "../../hooks/useCtrlExit.js";
import { useTerminalSize } from "../../vendor/inkContexts.js";
import { TerminalInput } from "../../components/terminal/TerminalInput.js";
import { figures } from "../../vendor/terminalFigures.js";
import { Indent } from "../../components/shared/Indent.js";
import { ToolError } from "../../components/shared/ToolError.js";
import { getBashTimeout, getMaxBashTimeout, getMaxBashOutputLength } from "../../utils/terminal/BashConfig.js";
import { optimizeImage } from "../../utils/shared/imageOptimizer.js";
import { processMcpToolResult } from "../../services/mcp/McpContentProcessor.js";
import {
    BackgroundShortcutHint,
    ExecutionProgressView,
    formatCommandForDisplay,
    TimeoutIndicator,
    ToolResultView,
    ToolUseRejectedView,
    WaitingView
} from "../../services/terminal/GitPolicyService.js";

const PROGRESS_INTERVAL_MS = 2000;
const YIELD_INTERVAL_MS = 1000;

const COMMAND_TYPE_TRIGGERS = [
    "npm",
    "yarn",
    "pnpm",
    "node",
    "python",
    "python3",
    "go",
    "cargo",
    "make",
    "docker",
    "terraform",
    "webpack",
    "vite",
    "jest",
    "pytest",
    "curl",
    "wget",
    "build",
    "test",
    "serve",
    "watch",
    "dev"
];

const SelectInput = Select as unknown as React.FC<{
    options: { label: string; value: string }[];
    onChange: (value: string) => void;
    onCancel?: () => void;
    visibleOptionCount?: number;
}>;

type PermissionRuleValue = {
    toolName: string;
    ruleContent?: string;
};

type PermissionRuleBehavior = "allow" | "deny" | "ask";

type BashToolInput = {
    command: string;
    timeout?: number;
    description?: string;
    shellExecutable?: string;
    run_in_background?: boolean;
    dangerouslyDisableSandbox?: boolean;
};

type BashToolOutput = {
    stdout: string;
    stderr: string;
    summary?: string;
    rawOutputPath?: string;
    interrupted: boolean;
    isImage?: boolean;
    backgroundTaskId?: string;
    dangerouslyDisableSandbox?: boolean;
    returnCodeInterpretation?: string;
    structuredContent?: any[];
};

function parseBooleanEnv(value?: string): boolean {
    if (!value) return false;
    return value === "1" || value.toLowerCase() === "true";
}

function trimBlankLines(text: string): string {
    const lines = text.split("\n");
    let start = 0;
    while (start < lines.length && lines[start]?.trim() === "") start += 1;
    let end = lines.length - 1;
    while (end >= 0 && lines[end]?.trim() === "") end -= 1;
    if (start > end) return "";
    return lines.slice(start, end + 1).join("\n");
}

function truncateOutput(text: string) {
    const isImage = /^data:image\/[a-z0-9.+_-]+;base64,/i.test(text);
    if (isImage) {
        return {
            totalLines: 1,
            truncatedContent: text,
            isImage
        };
    }

    const maxLength = getMaxBashOutputLength();
    if (text.length <= maxLength) {
        return {
            totalLines: text.split("\n").length,
            truncatedContent: text,
            isImage
        };
    }

    const truncated = text.slice(0, maxLength);
    const remainingLines = text.slice(maxLength).split("\n").length;
    return {
        totalLines: text.split("\n").length,
        truncatedContent: `${truncated}\n\n... [${remainingLines} lines truncated] ...`,
        isImage
    };
}

function splitShellCommands(command: string): string[] {
    try {
        const parts = command
            .split(/\s*(?:&&|\|\||;)+\s*/)
            .map((part) => part.trim())
            .filter(Boolean);
        return parts.length > 0 ? parts : [command];
    } catch {
        return [command];
    }
}

function getCommandType(command: string): string {
    const segments = splitShellCommands(command);
    if (segments.length === 0) return "other";
    for (const segment of segments) {
        const first = segment.split(" ")[0] || "";
        if (COMMAND_TYPE_TRIGGERS.includes(first)) return first;
    }
    return "other";
}

function trackGitOperations(command: string, exitCode: number) {
    if (exitCode !== 0) return;
    if (command.match(/\bgit\s+commit\b/)) {
        logTelemetryEvent("tengu_git_operation", { operation: "commit" });
        if (command.match(/--amend\b/)) {
            logTelemetryEvent("tengu_git_operation", { operation: "commit_amend" });
        }
    }
    if (command.match(/\bgh\s+pr\s+create\b/)) {
        logTelemetryEvent("tengu_git_operation", { operation: "pr_create" });
    }
    if (command.match(/\bglab\s+mr\s+create\b/)) {
        logTelemetryEvent("tengu_git_operation", { operation: "pr_create" });
    }
    if (command.match(/\bgit\s+(checkout|branch|switch)\b/)) {
        logTelemetryEvent("tengu_git_operation", { operation: "branch_management" });
    }
}

function isBackgroundAllowed(command: string): boolean {
    const segments = splitShellCommands(command);
    if (segments.length === 0) return true;
    const first = segments[0]?.trim();
    if (!first) return true;
    return !BACKGROUND_EXCLUDED_COMMANDS.includes(first);
}

function parseExcludedCommandPattern(pattern: string) {
    if (pattern.endsWith("*")) {
        return {
            type: "prefix" as const,
            prefix: pattern.slice(0, -1)
        };
    }
    return {
        type: "exact" as const,
        command: pattern
    };
}

function isCommandExcludedFromSandbox(command: string): boolean {
    const excluded = getSettings("localSettings")?.sandbox?.excludedCommands ?? [];
    if (excluded.length === 0) return false;
    for (const rule of excluded) {
        const parsed = parseExcludedCommandPattern(rule);
        switch (parsed.type) {
            case "exact":
                if (command.trim() === parsed.command) return true;
                break;
            case "prefix": {
                const trimmed = command.trim();
                if (trimmed === parsed.prefix || trimmed.startsWith(`${parsed.prefix} `)) return true;
                break;
            }
        }
    }
    return false;
}

function shouldUseSandbox(input: BashToolInput): boolean {
    if (!sandboxService.isEnabled()) return false;
    if (input.dangerouslyDisableSandbox && sandboxService.isUnsandboxedAllowed()) return false;
    if (!input.command) return false;
    if (isCommandExcludedFromSandbox(input.command)) return false;
    return true;
}

function getSearchOrReadFlags(command: string) {
    let parts: string[];
    try {
        parts = splitShellCommands(command);
    } catch {
        return { isSearch: false, isRead: false };
    }

    if (parts.length === 0) return { isSearch: false, isRead: false };

    let isSearch = false;
    let isRead = false;

    for (const part of parts) {
        const name = part.trim().split(/\s+/)[0];
        if (!name) continue;
        const search = SEARCH_COMMANDS.has(name);
        const read = READ_COMMANDS.has(name);
        if (!search && !read) return { isSearch: false, isRead: false };
        if (search) isSearch = true;
        if (read) isRead = true;
    }

    return { isSearch, isRead };
}

function getTaskOutputPath(taskId: string): string {
    const cwd = process.cwd();
    return path.join(cwd, ".claude", "tasks", `${taskId}.output`);
}

type ExitCodeInterpretation = {
    isError: boolean;
    message?: string;
};

function interpretExitCode(command: string, exitCode: number, _stdout: string, _stderr: string): ExitCodeInterpretation {
    const commands = splitShellCommands(command);
    const lastCommand = commands[commands.length - 1];
    const commandName = lastCommand?.trim().split(/\s+/)[0] ?? "";

    const table: Record<string, (code: number) => ExitCodeInterpretation> = {
        grep: (code) => ({
            isError: code >= 2,
            message: code === 1 ? "No matches found" : undefined
        }),
        rg: (code) => ({
            isError: code >= 2,
            message: code === 1 ? "No matches found" : undefined
        })
    };

    const handler = table[commandName];
    if (handler) return handler(exitCode);

    return {
        isError: exitCode !== 0,
        message: exitCode !== 0 ? `Command failed with exit code ${exitCode}` : undefined
    };
}

function normalizeOutput(text: string): string {
    return text.replace(/^\s*\n+/, "").trimEnd();
}

function parseMcpCliCommand(command: string) {
    const match = command.match(/^mcp-cli\s+(call|read)\s+([a-zA-Z0-9_-]+)\/([a-zA-Z0-9_-]+)(?:\s+([\s\S]+))?$/);
    if (!match) return null;
    const [, action, server, tool, args = ""] = match;
    if (!action || !server || !tool) return null;
    return {
        command: action,
        server,
        tool,
        toolName: tool,
        args,
        fullCommand: command
    };
}

async function parseMcpCliResult(stdout: string, command: string, meta: { tool: string; server: string }) {
    try {
        const parsed = JSON.parse(stdout);
        const processed: any = await processMcpToolResult(parsed, meta.tool, meta.server);
        const content = processed?.content;
        if (!content) return null;
        const schemaSummary = formatMcpSchemaSummary(processed?.type, processed?.schema);

        if (Array.isArray(content)) {
            const text = JSON.stringify(content, null, 2);
            if (shouldStoreMcpContent(text)) {
                const stored = await storeMcpOutput(text, command);
                if (!stored || isStoredMcpOutputInvalid(stored)) return null;
                return {
                    stdout: formatStoredMcpOutput(stored.filepath, stored.originalSize, schemaSummary, getMcpOutputLimit()),
                    structuredContent: undefined,
                    rawOutputPath: stored.filepath
                };
            }
            return {
                stdout: text,
                structuredContent: content,
                rawOutputPath: undefined
            };
        }

        if (typeof content === "string") {
            if (shouldStoreMcpContent(content)) {
                const stored = await storeMcpOutput(content, command);
                if (!stored || isStoredMcpOutputInvalid(stored)) return null;
                return {
                    stdout: formatStoredMcpOutput(stored.filepath, stored.originalSize, schemaSummary, getMcpOutputLimit()),
                    structuredContent: undefined,
                    rawOutputPath: stored.filepath
                };
            }
            return {
                stdout: content,
                structuredContent: undefined,
                rawOutputPath: undefined
            };
        }

        const serialized = JSON.stringify(content, null, 2);
        if (shouldStoreMcpContent(serialized)) {
            const stored = await storeMcpOutput(serialized, command);
            if (!stored || isStoredMcpOutputInvalid(stored)) return null;
            return {
                stdout: formatStoredMcpOutput(stored.filepath, stored.originalSize, schemaSummary, getMcpOutputLimit()),
                structuredContent: undefined,
                rawOutputPath: stored.filepath
            };
        }
        return {
            stdout: serialized,
            structuredContent: undefined,
            rawOutputPath: undefined
        };
    } catch (error) {
        logTelemetryEvent("tengu_bash_tool_mcp_cli_parse_error", {
            command,
            error: error instanceof Error ? error.message : String(error)
        });
        return null;
    }
}

type BashSummaryResult = {
    shouldSummarize: boolean;
    summary?: string;
    rawOutputPath?: string;
    reason?: string;
    queryDurationMs?: number;
    modelReason?: string;
};

async function maybeSummarizeOutput(stdout: string, stderr: string, command: string, abortController: AbortController, messages: any[]): Promise<BashSummaryResult | null> {
    try {
        return await summarizeBashOutput(stdout, command, messages.join("\n"), abortController.signal) as BashSummaryResult;
    } catch (error) {
        logTelemetryEvent("tengu_bash_tool_summarize_error", {
            error: error instanceof Error ? error.message : String(error)
        });
        return null;
    }
}

async function extractTouchedPaths(_command: string, _stdout: string): Promise<string[]> {
    return [];
}

function shouldStoreMcpContent(content: string): boolean {
    return content.length > getMaxBashOutputLength();
}

async function storeMcpOutput(content: string, command: string) {
    try {
        const safeCommand = command.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 30);
        const timestamp = Date.now();
        const filename = `mcp-cli-${safeCommand}-${timestamp}.json`;
        const outputDir = path.join(process.cwd(), ".claude", "mcp-cli");
        await fs.mkdir(outputDir, { recursive: true });
        const filepath = path.join(outputDir, filename);
        await fs.writeFile(filepath, content, "utf8");
        return { filepath, originalSize: content.length };
    } catch (error) {
        logTelemetryEvent("tengu_bash_tool_mcp_cli_write_error", {
            command,
            error: error instanceof Error ? error.message : String(error)
        });
        return null;
    }
}

function isStoredMcpOutputInvalid(stored: { filepath: string; originalSize: number } | null): boolean {
    return !stored || !stored.filepath;
}

function formatMcpSchemaSummary(type?: string, schema?: { title?: string; name?: string }) {
    if (schema?.title) return schema.title;
    if (schema?.name) return schema.name;
    if (type) return type;
    return undefined;
}

function getMcpOutputLimit(): number {
    return getMaxBashOutputLength();
}

function formatStoredMcpOutput(filepath: string, originalSize: number, schemaSummary?: string, outputLimit?: number) {
    const schemaLabel = schemaSummary ? `, ${schemaSummary}` : "";
    const limitLabel = outputLimit ? `, preview limit ${outputLimit} chars` : "";
    return `Output saved to ${filepath} (${originalSize} bytes${schemaLabel}${limitLabel})`;
}

function stripSandboxViolations(text: string): string {
    return text.replace(/<sandbox_violations>[\s\S]*?<\/sandbox_violations>/g, "");
}

function shouldStripSandboxViolations(toolPermissionContext: any): boolean {
    if (!toolPermissionContext) return false;
    if (toolPermissionContext.shouldAvoidPermissionPrompts) return true;
    return toolPermissionContext.mode === "dontAsk";
}

function getCodeIndexingTool(command: string): string | null {
    const parts = splitShellCommands(command);
    for (const part of parts) {
        const name = part.trim().split(/\s+/)[0];
        if (name && SEARCH_COMMANDS.has(name)) return name;
    }
    return null;
}

type ShellCommandHandle = {
    result?: Promise<{
        stdout: string;
        stderr: string;
        code: number;
        interrupted: boolean;
        backgroundTaskId?: string;
    }>;
    onTimeout?: (handler: (taskId: string) => void) => void;
};

async function* runShellCommand({
    input,
    abortController,
    setAppState,
    setToolJSX,
    preventCwdChanges
}: {
    input: BashToolInput;
    abortController: AbortController;
    setAppState: (updater: (state: any) => any) => void;
    setToolJSX?: (jsx: any) => void;
    preventCwdChanges?: boolean;
}) {
    const { command, description, timeout, shellExecutable, run_in_background } = input;
    const effectiveTimeout = timeout || getBashTimeout();

    let fullOutput = "";
    let recentOutput = "";
    let totalLines = 0;
    let backgroundTaskId: string | undefined;

    const allowBackground = isBackgroundAllowed(command);

    const shellCommand = await runBashCommand(command, {
        timeout: effectiveTimeout,
        signal: abortController.signal,
        shellExecutable,
        onOutput: (recent: string, all: string, lines: number) => {
            recentOutput = recent;
            fullOutput = all;
            totalLines = lines;
        },
        preventCwdChanges,
        useSandbox: shouldUseSandbox(input),
        allowBackground
    }) as ShellCommandHandle;

    const resultPromise = shellCommand.result ?? Promise.resolve({
        stdout: "",
        stderr: "",
        code: 0,
        interrupted: false
    });

    async function spawnBackgroundTask() {
        const task = await LocalBashTask.spawn({
            command,
            description: description || command,
            shellCommand
        }, {
            setAppState
        });
        return task.taskId;
    }

    function recordBackground(eventName: string, callback?: (taskId: string) => void) {
        spawnBackgroundTask().then((taskId) => {
            backgroundTaskId = taskId;
            logTelemetryEvent(eventName, { command_type: getCommandType(command) });
            callback?.(taskId);
        });
    }

    function setBackgroundFromShortcut() {
        recordBackground("tengu_bash_command_backgrounded");
    }

    if (allowBackground && shellCommand.onTimeout) {
        shellCommand.onTimeout((taskId: string) => {
            backgroundTaskId = taskId;
            logTelemetryEvent("tengu_bash_command_timeout_backgrounded", { command_type: getCommandType(command) });
        });
    }

    if (run_in_background) {
        const taskId = await spawnBackgroundTask();
        logTelemetryEvent("tengu_bash_command_explicitly_backgrounded", { command_type: getCommandType(command) });
        return {
            stdout: "",
            stderr: "",
            code: 0,
            interrupted: false,
            backgroundTaskId: taskId
        };
    }

    const start = Date.now();
    let nextYieldAt = start + PROGRESS_INTERVAL_MS;

    while (true) {
        const now = Date.now();
        const wait = Math.max(0, nextYieldAt - now);
        const result = await Promise.race([
            resultPromise,
            new Promise((resolve) => setTimeout(() => resolve(null), wait))
        ]);

        if (result !== null) {
            return result;
        }

        if (backgroundTaskId) {
            return {
                stdout: "",
                stderr: "",
                code: 0,
                interrupted: false,
                backgroundTaskId
            };
        }

        const elapsedSeconds = Math.floor((Date.now() - start) / 1000);
        if (!backgroundTaskId && elapsedSeconds >= PROGRESS_INTERVAL_MS / 1000 && setToolJSX) {
            setToolJSX({
                jsx: <BackgroundShortcutHint onBackground={setBackgroundFromShortcut} />,
                shouldHidePromptInput: false,
                shouldContinueAnimation: true,
                showSpinner: true
            });
        }

        yield {
            type: "progress",
            fullOutput,
            output: recentOutput,
            elapsedTimeSeconds: elapsedSeconds,
            totalLines
        };

        nextYieldAt = Date.now() + YIELD_INTERVAL_MS;
    }
}

export const BashTool = {
    name: "Bash",
    strict: true,

    async description({ description }: { description?: string }) {
        return description || "Run shell command";
    },

    async prompt() {
        return "";
    },

    isConcurrencySafe(input: BashToolInput) {
        return this.isReadOnly(input);
    },

    isReadOnly(input: BashToolInput) {
        return evaluateBashCommandSafety(input.command).behavior === "allow";
    },

    isSearchOrReadCommand(input: BashToolInput) {
        const result = bashInputSchema.safeParse(input);
        if (!result.success) return { isSearch: false, isRead: false };
        return getSearchOrReadFlags(result.data.command);
    },

    inputSchema: z.object({
        command: z.string().describe("The command to execute"),
        timeout: z.number().optional().describe(`Optional timeout in milliseconds (max ${getMaxBashTimeout()})`),
        description: z
            .string()
            .optional()
            .describe(
                "Clear, concise description of what this command does in 5-10 words, in active voice. Examples:\n" +
                "Input: ls\nOutput: List files in current directory\n\n" +
                "Input: git status\nOutput: Show working tree status\n\n" +
                "Input: npm install\nOutput: Install package dependencies\n\n" +
                "Input: mkdir foo\nOutput: Create directory 'foo'"
            ),
        run_in_background: z
            .boolean()
            .optional()
            .describe("Set to true to run this command in the background. Use TaskOutput to read the output later."),
        dangerouslyDisableSandbox: z
            .boolean()
            .optional()
            .describe("Set this to true to dangerously override sandbox mode and run commands without sandboxing.")
    }).strict(),

    outputSchema: z.object({
        stdout: z.string().describe("The standard output of the command"),
        stderr: z.string().describe("The standard error output of the command"),
        summary: z.string().optional().describe("Summarized output when available"),
        rawOutputPath: z.string().optional().describe("Path to raw output file when summarized"),
        interrupted: z.boolean().describe("Whether the command was interrupted"),
        isImage: z.boolean().optional().describe("Flag to indicate if stdout contains image data"),
        backgroundTaskId: z.string().optional().describe("ID of the background task if command is running in background"),
        dangerouslyDisableSandbox: z.boolean().optional().describe("Flag to indicate if sandbox mode was overridden"),
        returnCodeInterpretation: z
            .string()
            .optional()
            .describe("Semantic interpretation for non-error exit codes with special meaning"),
        structuredContent: z.array(z.any()).optional().describe("Structured content blocks from mcp-cli commands")
    }),

    userFacingName(input: BashToolInput) {
        if (!input) return "Bash";
        if (shouldUseSandbox(input) && parseBooleanEnv(process.env.CLAUDE_CODE_BASH_SANDBOX_SHOW_INDICATOR)) {
            return "SandboxedBash";
        }
        return "Bash";
    },

    getToolUseSummary(input: BashToolInput) {
        if (!input?.command) return null;
        if (input.description) return input.description;
        return formatCommandForDisplay(input.command, false);
    },

    isEnabled() {
        return true;
    },

    async checkPermissions(input: BashToolInput, context: any) {
        const { coordinateBashPermissionCheck } = await import("../../services/validation/BashPermissionCoordinator.js");
        return coordinateBashPermissionCheck(input, context, undefined);
    },

    renderToolUseMessage(input: BashToolInput, options: any) {
        return formatCommandForDisplay(input.command, options?.verbose);
    },

    renderToolUseTag(input: BashToolInput) {
        return <TimeoutIndicator timeout={input.timeout} />;
    },

    renderToolUseRejectedMessage() {
        return <ToolUseRejectedView />;
    },

    renderToolUseProgressMessage(progressMessages: any[], options: any) {
        const latest = progressMessages[progressMessages.length - 1];
        if (!latest || !latest.data || !latest.data.output) {
            return (
                <Indent>
                    <Text dimColor>Running…</Text>
                </Indent>
            );
        }

        return (
            <ExecutionProgressView
                fullOutput={latest.data.fullOutput}
                output={latest.data.output}
                elapsedTimeSeconds={latest.data.elapsedTimeSeconds}
                totalLines={latest.data.totalLines}
                verbose={options?.verbose}
            />
        );
    },

    renderToolUseQueuedMessage() {
        return <WaitingView />;
    },

    renderToolResultMessage(result: string, options: any) {
        return <ToolResultView content={result} verbose={options?.verbose} />;
    },

    renderToolUseErrorMessage(result: any, options: any) {
        return <ToolError result={result} verbose={options?.verbose} />;
    },

    mapToolResultToToolResultBlockParam(result: BashToolOutput, toolUseId: string) {
        if (result.structuredContent && result.structuredContent.length > 0) {
            return {
                tool_use_id: toolUseId,
                type: "tool_result",
                content: result.structuredContent
            };
        }

        if (result.isImage) {
            const match = result.stdout.trim().match(/^data:([^;]+);base64,(.+)$/);
            if (match) {
                const mediaType = match[1];
                const data = match[2];
                return {
                    tool_use_id: toolUseId,
                    type: "tool_result",
                    content: [
                        {
                            type: "image",
                            source: {
                                type: "base64",
                                media_type: mediaType || "image/jpeg",
                                data: data || ""
                            }
                        }
                    ]
                };
            }
        }

        if (result.summary) {
            return {
                tool_use_id: toolUseId,
                type: "tool_result",
                content: result.summary,
                is_error: result.interrupted
            };
        }

        const stdout = normalizeOutput(result.stdout || "");
        let stderr = (result.stderr || "").trim();
        if (result.interrupted) {
            if (stderr) stderr += os.EOL;
            stderr += "<error>Command was aborted before completion</error>";
        }

        const backgroundNote = result.backgroundTaskId
            ? `Command running in background with ID: ${result.backgroundTaskId}. Output is being written to: ${getTaskOutputPath(result.backgroundTaskId)}`
            : "";

        return {
            tool_use_id: toolUseId,
            type: "tool_result",
            content: [stdout, stderr, backgroundNote].filter(Boolean).join("\n"),
            is_error: result.interrupted
        };
    },

    async call(input: BashToolInput, context: any, _canUseTool?: any, _assistantMessage?: any, onProgress?: any) {
        const { abortController, readFileState, getAppState, setAppState, setToolJSX, messages } = context;

        const stdoutBuffer = new OutputBuffer();
        const stderrBuffer = new OutputBuffer();
        let returnCodeInfo: ExitCodeInterpretation | undefined;
        let interrupted = false;
        let result: any;
        const preventCwdChanges = Boolean(context?.agentId);

        try {
            const generator = runShellCommand({
                input,
                abortController,
                setAppState,
                setToolJSX,
                preventCwdChanges
            });

            let next;
            let progressIndex = 0;
            do {
                next = await generator.next();
                if (!next.done && onProgress) {
                    onProgress({
                        toolUseID: `bash-progress-${progressIndex++}`,
                        data: {
                            type: "bash_progress",
                            output: next.value.output,
                            fullOutput: next.value.fullOutput,
                            elapsedTimeSeconds: next.value.elapsedTimeSeconds,
                            totalLines: next.value.totalLines
                        }
                    });
                }
            } while (!next.done);

            result = next.value;
            trackGitOperations(input.command, result.code);

            stdoutBuffer.append((result.stdout || "").trimEnd() + os.EOL);
            returnCodeInfo = interpretExitCode(input.command, result.code, result.stdout || "", result.stderr || "");

            if (result.stderr && result.stderr.includes(".git/index.lock': File exists")) {
                logTelemetryEvent("tengu_git_index_lock_error", {});
            }

            if (returnCodeInfo.isError) {
                stderrBuffer.append(result.stderr.trimEnd() + os.EOL);
                if (result.code !== 0) stderrBuffer.append(`Exit code ${result.code}`);
            } else if (parseMcpCliCommand(input.command)) {
                stderrBuffer.append(result.stderr.trimEnd() + os.EOL);
            } else {
                stdoutBuffer.append(result.stderr.trimEnd() + os.EOL);
            }

            if (!preventCwdChanges && getAppState) {
                const latestState = await getAppState();
                if (shouldStripSandboxViolations(latestState?.toolPermissionContext)) {
                    const current = stderrBuffer.toString();
                    stderrBuffer.clear();
                    stderrBuffer.append(stripSandboxViolations(current));
                }
            }

            const annotatedStderr = sandboxService.annotateStderr?.(input.command, result.stderr || "");
            if (returnCodeInfo.isError) {
                throw new Error(annotatedStderr || result.stderr || `Command failed with exit code ${result.code}`);
            }

            interrupted = result.interrupted;
        } finally {
            if (setToolJSX) setToolJSX(null);
        }

        const stdoutText = stdoutBuffer.toString();
        const stderrText = stderrBuffer.toString();

        if (readFileState) {
            extractTouchedPaths(input.command, stdoutText).then(async (paths) => {
                for (const filePath of paths) {
                    const absolutePath = path.isAbsolute(filePath) ? filePath : path.resolve(process.cwd(), filePath);
                    try {
                        if (!FileReadTool.validateInput) {
                            readFileState.delete(absolutePath);
                            continue;
                        }
                        const validation = await FileReadTool.validateInput({ file_path: absolutePath }, context);
                        if (!validation.result) {
                            readFileState.delete(absolutePath);
                            continue;
                        }
                        await FileReadTool.call({ file_path: absolutePath }, context);
                    } catch (error) {
                        readFileState.delete(absolutePath);
                        logTelemetryEvent("tengu_bash_tool_haiku_file_paths_read_error", {
                            error: error instanceof Error ? error.message : String(error)
                        });
                    }
                }
                const readFileStateValuesCharLength = Array.from(readFileState.values()).reduce((sum, value: any) => {
                    return sum + (value?.content?.length || 0);
                }, 0);
                logTelemetryEvent("tengu_bash_tool_haiku_file_paths_read", {
                    filePathsExtracted: paths.length,
                    readFileStateSize: readFileState.size,
                    readFileStateValuesCharLength
                });
            }).catch((error) => {
                if (error instanceof Error && error.message.includes("Request was aborted")) return;
                logTelemetryEvent("tengu_bash_tool_haiku_file_paths_read_error", {
                    error: error instanceof Error ? error.message : String(error)
                });
            });
        }

        const summaryResult = await maybeSummarizeOutput(stdoutText, stderrText, input.command, abortController, messages || []);
        const shouldSummarize = summaryResult?.shouldSummarize === true;

        logTelemetryEvent("tengu_bash_tool_command_executed", {
            command_type: input.command.split(" ")[0],
            stdout_length: stdoutText.length,
            stderr_length: stderrText.length,
            exit_code: result.code,
            interrupted,
            summarization_attempted: summaryResult !== null,
            summarization_succeeded: shouldSummarize,
            summarization_duration_ms: summaryResult?.queryDurationMs,
            summarization_reason: !shouldSummarize && summaryResult ? summaryResult.reason : undefined,
            model_summarization_reason: summaryResult?.modelReason,
            summary_length: summaryResult?.summary ? summaryResult.summary.length : undefined
        });

        const indexingTool = getCodeIndexingTool(input.command);
        if (indexingTool) {
            logTelemetryEvent("tengu_code_indexing_tool_used", {
                tool: indexingTool,
                source: "cli",
                success: result.code === 0
            });
        }

        const trimmedStdout = trimBlankLines(stdoutText);
        const trimmedStderr = trimBlankLines(stderrText);

        const stdoutTruncate = truncateOutput(trimmedStdout);
        const stderrTruncate = truncateOutput(trimmedStderr);

        let structuredContent: any[] | undefined;
        let rawOutputPath: string | undefined;
        let stdoutValue = stdoutTruncate.truncatedContent;

        const mcpCommand = parseMcpCliCommand(input.command);
        if (mcpCommand) {
            const mcpResult = await parseMcpCliResult(trimmedStdout, input.command, {
                tool: mcpCommand.tool,
                server: mcpCommand.server
            });
            if (mcpResult) {
                stdoutValue = mcpResult.stdout;
                structuredContent = mcpResult.structuredContent;
                rawOutputPath = mcpResult.rawOutputPath;
            }
        }

        let finalStdout = stdoutValue;
        if (stdoutTruncate.isImage) {
            const match = stdoutValue.trim().match(/^data:([^;]+);base64,(.+)$/);
            if (match && match[1] && match[2]) {
                const buffer = Buffer.from(match[2], "base64");
                const optimized = await optimizeImage(buffer, undefined as unknown as number, match[1]);
                finalStdout = `data:${optimized.mediaType};base64,${optimized.base64}`;
            }
        }

        return {
            data: {
                stdout: finalStdout,
                stderr: stderrTruncate.truncatedContent,
                summary: shouldSummarize ? summaryResult?.summary : undefined,
                rawOutputPath: shouldSummarize ? summaryResult?.rawOutputPath : rawOutputPath,
                interrupted,
                isImage: stdoutTruncate.isImage,
                returnCodeInterpretation: returnCodeInfo?.message,
                backgroundTaskId: result.backgroundTaskId,
                structuredContent,
                dangerouslyDisableSandbox: "dangerouslyDisableSandbox" in input ? input.dangerouslyDisableSandbox : undefined
            }
        };
    }
};

const bashInputSchema = BashTool.inputSchema;

export function PermissionRuleValueDetails({ ruleValue }: { ruleValue: PermissionRuleValue }) {
    switch (ruleValue.toolName) {
        case BashTool.name:
            if (ruleValue.ruleContent) {
                if (ruleValue.ruleContent.endsWith(":*")) {
                    return (
                        <Text dimColor>
                            Any Bash command starting with <Text bold>{ruleValue.ruleContent.slice(0, -2)}</Text>
                        </Text>
                    );
                }
                return (
                    <Text dimColor>
                        The Bash command <Text bold>{ruleValue.ruleContent}</Text>
                    </Text>
                );
            }
            return <Text dimColor>Any Bash command</Text>;
        default:
            if (!ruleValue.ruleContent) {
                return (
                    <Text dimColor>
                        Any use of the <Text bold>{ruleValue.toolName}</Text> tool
                    </Text>
                );
            }
            return null;
    }
}

function LineBreak() {
    return <Text>\n</Text>;
}

export function AddPermissionRuleInputView({
    onCancel,
    onSubmit,
    ruleBehavior
}: {
    onCancel: () => void;
    onSubmit: (ruleValue: PermissionRuleValue, ruleBehavior: PermissionRuleBehavior) => void;
    ruleBehavior: PermissionRuleBehavior;
}) {
    const [value, setValue] = useState("");
    const [cursorOffset, setCursorOffset] = useState(0);
    const ctrlExit = useCtrlExit();

    useInput((_, key) => {
        if (key.escape) onCancel();
    });

    const { columns } = useTerminalSize();
    const width = columns - 6;

    const handleSubmit = (inputValue: string) => {
        const trimmed = inputValue.trim();
        if (!trimmed) return;
        onSubmit(parsePermissionRuleValue(trimmed), ruleBehavior);
    };

    return (
        <>
            <Box flexDirection="column" gap={1} borderStyle="round" paddingLeft={1} paddingRight={1} borderColor="permission">
                <Text bold color="permission">Add {ruleBehavior} permission rule</Text>
                <Box flexDirection="column">
                    <Text>
                        Permission rules are a tool name, optionally followed by a specifier in parentheses.
                        <LineBreak />
                        e.g., <Text bold>{formatPermissionRuleValue({ toolName: FileReadTool.name })}</Text>
                        <Text> or </Text>
                        <Text bold>{formatPermissionRuleValue({ toolName: BashTool.name, ruleContent: "ls:*" })}</Text>
                    </Text>
                </Box>
                <Box borderDimColor borderStyle="round" marginY={1} paddingLeft={1}>
                    <TerminalInput
                        showCursor
                        value={value}
                        onChange={setValue}
                        onSubmit={handleSubmit}
                        placeholder={`Enter permission rule${figures.ellipsis}`}
                        columns={width}
                        cursorOffset={cursorOffset}
                        onChangeCursorOffset={setCursorOffset}
                    />
                </Box>
            </Box>
            <Box marginLeft={3}>
                {ctrlExit.pending ? (
                    <Text dimColor>Press {ctrlExit.keyName} again to exit</Text>
                ) : (
                    <Text dimColor>Enter to submit · Esc to cancel</Text>
                )}
            </Box>
        </>
    );
}

export function WorkspacePermissionsView({
    onExit,
    getToolPermissionContext,
    onRequestAddDirectory,
    onRequestRemoveDirectory
}: {
    onExit: (message: string, meta?: any) => void;
    getToolPermissionContext: () => { additionalWorkingDirectories: Map<string, any> };
    onRequestAddDirectory: () => void;
    onRequestRemoveDirectory: (path: string) => void;
}) {
    const context = getToolPermissionContext();
    const additional = useMemo(() => {
        return Array.from(context.additionalWorkingDirectories.keys()).map((dir) => ({
            path: dir,
            isCurrent: false,
            isDeletable: true
        }));
    }, [context.additionalWorkingDirectories]);

    const handleSelection = useCallback(
        (value: string) => {
            if (value === "add-directory") {
                onRequestAddDirectory();
                return;
            }
            const entry = additional.find((dir) => dir.path === value);
            if (entry && entry.isDeletable) onRequestRemoveDirectory(entry.path);
        },
        [additional, onRequestAddDirectory, onRequestRemoveDirectory]
    );

    const options = useMemo(() => {
        const items = additional.map((dir) => ({ label: dir.path, value: dir.path }));
        items.push({ label: `Add directory${figures.ellipsis}`, value: "add-directory" });
        return items;
    }, [additional]);

    return (
        <Box flexDirection="column" marginBottom={1}>
            <Box flexDirection="row" marginTop={1} marginLeft={2} gap={1}>
                <Text>{`-  ${getOriginalCwd()}`}</Text>
                <Text dimColor>(Original working directory)</Text>
            </Box>
            <SelectInput
                options={options}
                onChange={handleSelection}
                onCancel={() => onExit("Workspace dialog dismissed", { display: "system" })}
                visibleOptionCount={Math.min(10, options.length)}
            />
        </Box>
    );
}

function findUnescapedForward(text: string, char: string): number {
    for (let i = 0; i < text.length; i += 1) {
        if (text[i] === char) {
            let backslashes = 0;
            for (let j = i - 1; j >= 0 && text[j] === "\\"; j -= 1) backslashes += 1;
            if (backslashes % 2 === 0) return i;
        }
    }
    return -1;
}

function findUnescapedBackward(text: string, char: string): number {
    for (let i = text.length - 1; i >= 0; i -= 1) {
        if (text[i] === char) {
            let backslashes = 0;
            for (let j = i - 1; j >= 0 && text[j] === "\\"; j -= 1) backslashes += 1;
            if (backslashes % 2 === 0) return i;
        }
    }
    return -1;
}

function escapeRuleContent(content: string): string {
    return content.replace(/\\/g, "\\\\").replace(/\(/g, "\\(").replace(/\)/g, "\\)");
}

function unescapeRuleContent(content: string): string {
    return content.replace(/\\\(/g, "(").replace(/\\\)/g, ")").replace(/\\\\/g, "\\");
}

function parsePermissionRuleValue(rule: string): PermissionRuleValue {
    const openIndex = findUnescapedForward(rule, "(");
    if (openIndex === -1) return { toolName: rule };

    const closeIndex = findUnescapedBackward(rule, ")");
    if (closeIndex === -1 || closeIndex <= openIndex || closeIndex !== rule.length - 1) {
        return { toolName: rule };
    }

    const toolName = rule.substring(0, openIndex);
    const content = rule.substring(openIndex + 1, closeIndex);
    if (!toolName || content === undefined) return { toolName: rule };

    return {
        toolName,
        ruleContent: unescapeRuleContent(content)
    };
}

function formatPermissionRuleValue(rule: PermissionRuleValue): string {
    if (!rule.ruleContent) return rule.toolName;
    return `${rule.toolName}(${escapeRuleContent(rule.ruleContent)})`;
}
