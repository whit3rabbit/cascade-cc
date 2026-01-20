// Logic from chunk_466.ts (Permission Request Components)

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Box, Text, useInput, useStdout } from "ink";
import fs from "node:fs";
import path from "node:path";
import { DiffView } from "../shared/DiffView.js";
import { createPatch } from "../../utils/shared/diffUtils.js";
import { getProjectRoot } from "../../utils/shared/pathUtils.js";
import { logTelemetryEvent } from "../../services/telemetry/telemetryInit.js";
import { PermissionDialogLayout, PermissionDebugInfo, PermissionRuleSummary, PermissionSelect, logPermissionPrompt, logUnaryEvent, usePermissionOptions } from "./PermissionComponents.js";
import { getBashPermissionOptions } from "../../services/validation/BashPermissionCoordinator.js";
import { isSandboxingEnabled } from "../../services/sandbox/sandboxService.js";
import { sendNotification } from "../../services/notifications/NotificationService.js";
import { useTheme } from "../../services/terminal/themeManager.js";

function logPermissionOptionSelected(option: string) {
    const optionIndex: Record<string, number> = {
        yes: 1,
        "yes-apply-suggestions": 2,
        no: 3
    };
    const index = optionIndex[option];
    if (index) {
        void logTelemetryEvent("tengu_permission_request_option_selected", {
            option_index: index
        });
    }
}

function parseMcpCommand(command: string) {
    if (!command?.startsWith("mcp__")) return null;
    const parts = command.split("__");
    if (parts.length < 3) return null;
    const [, server, toolName, ...args] = parts;
    return {
        server,
        toolName,
        args: args.length ? args.join("__") : ""
    };
}

const PERMISSION_PROMPT_NOTIFICATION_DELAY_MS = 6000;

export function usePermissionPromptNotification(message: string, notificationType: string) {
    const lastInputRef = useRef(Date.now());
    const notifiedRef = useRef(false);

    useEffect(() => {
        const handleInput = () => {
            lastInputRef.current = Date.now();
        };
        process.stdin.on("data", handleInput);
        return () => {
            process.stdin.off("data", handleInput);
        };
    }, []);

    useEffect(() => {
        const interval = setInterval(() => {
            if (notifiedRef.current) return;
            if (Date.now() - lastInputRef.current >= PERMISSION_PROMPT_NOTIFICATION_DELAY_MS) {
                notifiedRef.current = true;
                void sendNotification(message, notificationType);
            }
        }, PERMISSION_PROMPT_NOTIFICATION_DELAY_MS);

        return () => clearInterval(interval);
    }, [message, notificationType]);
}

// --- MCP Permission Request (ck2) ---
export function McpPermissionRequest({ toolUseConfirm, onDone, onReject, serverName, toolName, args }: any) {
    const label = `${serverName} - ${toolName}`;
    const mcpCommandName = `mcp__${serverName}__${toolName}`;

    const completionMeta = useMemo(() => ({
        completion_type: "tool_use_single",
        language_name: "none"
    }), []);

    const request = useMemo(() => ({
        ...toolUseConfirm,
        tool: {
            ...toolUseConfirm.tool,
            name: mcpCommandName,
            isMcp: true
        }
    }), [toolUseConfirm, mcpCommandName]);

    logPermissionPrompt(request, completionMeta);

    const handleAction = useCallback((action: string) => {
        switch (action) {
            case "yes":
                logUnaryEvent({
                    completionType: "tool_use_single",
                    event: "accept",
                    languageName: "none",
                    messageId: request.assistantMessage?.message?.id
                });
                request.onAllow(request.input, []);
                onDone();
                break;
            case "yes-dont-ask-again": {
                logUnaryEvent({
                    completionType: "tool_use_single",
                    event: "accept",
                    languageName: "none",
                    messageId: request.assistantMessage?.message?.id
                });
                const suggestions = request.permissionResult?.behavior === "ask" ? request.permissionResult?.suggestions ?? [] : [];
                if (suggestions.length === 0) {
                    console.error(`MCPCliPermissionRequest: No MCP suggestions found for ${serverName}/${toolName}`);
                    request.onAllow(request.input, []);
                } else {
                    request.onAllow(request.input, suggestions);
                }
                onDone();
                break;
            }
            case "no":
                logUnaryEvent({
                    completionType: "tool_use_single",
                    event: "reject",
                    languageName: "none",
                    messageId: request.assistantMessage?.message?.id
                });
                request.onReject();
                onReject();
                onDone();
                break;
        }
    }, [onDone, onReject, request, serverName, toolName]);

    const projectLabel = path.basename(getProjectRoot());
    const options = useMemo(() => [
        { label: "Yes", value: "yes" },
        {
            label: (
                <Text>
                    Yes, and don't ask again for <Text bold>{label}</Text> commands in <Text bold>{projectLabel}</Text>
                </Text>
            ),
            value: "yes-dont-ask-again"
        },
        {
            label: (
                <Text>
                    No, and tell Claude what to do differently <Text bold>(esc)</Text>
                </Text>
            ),
            value: "no"
        }
    ], [label, projectLabel]);

    return (
        <PermissionDialogLayout title="Tool use">
            <Box flexDirection="column" paddingX={2} paddingY={1}>
                <Text>
                    {label}({args || "{}"})<Text dimColor> (MCP)</Text>
                </Text>
                <Text dimColor>{request.description}</Text>
            </Box>
            <Box flexDirection="column">
                <PermissionRuleSummary permissionResult={request.permissionResult} toolType="tool" />
                <Text>Do you want to proceed?</Text>
                <PermissionSelect options={options} onChange={handleAction} onCancel={() => handleAction("no")} />
            </Box>
        </PermissionDialogLayout>
    );
}

// --- Bash Permission Request (Bd5) ---
export function BashPermissionRequest({ toolUseConfirm, toolUseContext, onDone, onReject, verbose, command, description }: any) {
    const [showDebug, setShowDebug] = useState(false);
    const [rejectFeedback, setRejectFeedback] = useState("");
    const [acceptFeedback, setAcceptFeedback] = useState("");
    const [yesInputMode, setYesInputMode] = useState(false);
    const [noInputMode, setNoInputMode] = useState(false);
    const [focusedOption, setFocusedOption] = useState("yes");
    const acceptFeedbackEnabled = process.env.CLAUDE_CODE_ACCEPT_FEEDBACK === "1";
    const [theme] = useTheme();

    const completionMeta = useMemo(() => ({
        completion_type: "tool_use_single",
        language_name: "none"
    }), []);

    logPermissionPrompt(toolUseConfirm, completionMeta);

    const options = useMemo(() => getBashPermissionOptions({
        suggestions: toolUseConfirm.permissionResult?.behavior === "ask" ? toolUseConfirm.permissionResult?.suggestions : undefined,
        onRejectFeedbackChange: setRejectFeedback,
        onAcceptFeedbackChange: setAcceptFeedback,
        yesInputMode,
        noInputMode,
        acceptFeedbackEnabled
    }), [acceptFeedbackEnabled, noInputMode, toolUseConfirm.permissionResult, yesInputMode]);

    useInput((input, key) => {
        if (key.ctrl && input === "d") {
            setShowDebug((prev) => !prev);
        }
    });

    const renderedCommand = toolUseConfirm.tool?.renderToolUseMessage
        ? toolUseConfirm.tool.renderToolUseMessage({ command, description }, { theme: theme ?? "default", verbose: true })
        : command;

    const toggleInputMode = useCallback((value: string) => {
        if (!acceptFeedbackEnabled) return;
        if (value === "yes") {
            setYesInputMode((prev) => {
                if (!prev) void logTelemetryEvent("tengu_accept_feedback_mode_entered", {});
                return !prev;
            });
        } else if (value === "no") {
            setNoInputMode((prev) => {
                if (!prev) void logTelemetryEvent("tengu_reject_feedback_mode_entered", {});
                return !prev;
            });
        }
    }, [acceptFeedbackEnabled]);

    const handleReject = useCallback((feedback?: string) => {
        const trimmed = feedback?.trim();
        const hasFeedback = Boolean(trimmed);
        if (!hasFeedback) {
            void logTelemetryEvent("tengu_permission_request_escape", {});
        }
        logUnaryEvent({
            completionType: "tool_use_single",
            event: "reject",
            languageName: "none",
            messageId: toolUseConfirm.assistantMessage?.message?.id,
            hasFeedback
        });
        toolUseConfirm.onReject(trimmed || undefined);
        onReject();
        onDone();
    }, [onDone, onReject, toolUseConfirm]);

    const handleAccept = useCallback((value: string, inputValue?: string) => {
        logPermissionOptionSelected(value);
        switch (value) {
            case "yes": {
                const trimmed = (inputValue ?? acceptFeedback).trim();
                if (trimmed) {
                    void logTelemetryEvent("tengu_accept_with_instructions_submitted", {
                        instructions_length: trimmed.length
                    });
                }
                logUnaryEvent({
                    completionType: "tool_use_single",
                    event: "accept",
                    languageName: "none",
                    messageId: toolUseConfirm.assistantMessage?.message?.id,
                    hasFeedback: Boolean(trimmed)
                });
                toolUseConfirm.onAllow(toolUseConfirm.input, [], trimmed || undefined);
                onDone();
                break;
            }
            case "yes-apply-suggestions": {
                logUnaryEvent({
                    completionType: "tool_use_single",
                    event: "accept",
                    languageName: "none",
                    messageId: toolUseConfirm.assistantMessage?.message?.id
                });
                const updates = toolUseConfirm.permissionResult?.suggestions ?? [];
                toolUseConfirm.onAllow(toolUseConfirm.input, updates);
                onDone();
                break;
            }
            case "no": {
                const trimmed = (inputValue ?? rejectFeedback).trim();
                if (!acceptFeedbackEnabled && !trimmed) return;
                handleReject(trimmed || undefined);
                break;
            }
        }
    }, [acceptFeedback, acceptFeedbackEnabled, handleReject, onDone, rejectFeedback, toolUseConfirm]);

    const sandboxEnabled = isSandboxingEnabled();
    const isUnsandboxed = sandboxEnabled && Boolean(toolUseConfirm.input?.dangerouslyDisableSandbox);

    return (
        <PermissionDialogLayout title={isUnsandboxed ? "Bash command (unsandboxed)" : "Bash command"}>
            <Box flexDirection="column" paddingX={2} paddingY={1}>
                <Text>{renderedCommand}</Text>
                <Text dimColor>{description}</Text>
            </Box>
            {showDebug ? (
                <Box flexDirection="column">
                    <PermissionDebugInfo permissionResult={toolUseConfirm.permissionResult} />
                    {toolUseContext?.options?.debug && (
                        <Box justifyContent="flex-end" marginTop={1}>
                            <Text dimColor>Ctrl-D to hide debug info</Text>
                        </Box>
                    )}
                </Box>
            ) : (
                <>
                    <Box flexDirection="column">
                        <PermissionRuleSummary permissionResult={toolUseConfirm.permissionResult} toolType="command" />
                        <Box flexDirection="row" gap={1}>
                            <Text>Do you want to proceed?</Text>
                            {acceptFeedbackEnabled && (focusedOption === "yes" || focusedOption === "no") && (
                                <Text dimColor>(Tab to add further instructions)</Text>
                            )}
                        </Box>
                        <PermissionSelect
                            options={options}
                            inlineDescriptions={acceptFeedbackEnabled}
                            onChange={(value, inputValue) => {
                                if (value === "no") {
                                    setRejectFeedback(inputValue ?? "");
                                }
                                if (value === "yes") {
                                    setAcceptFeedback(inputValue ?? "");
                                }
                                handleAccept(value, inputValue);
                            }}
                            onCancel={() => handleReject()}
                            onFocus={(value) => setFocusedOption(value)}
                            onInputModeToggle={toggleInputMode}
                        />
                    </Box>
                    <Box justifyContent="space-between" marginTop={1}>
                        <Text dimColor>Esc to cancel</Text>
                        {toolUseContext?.options?.debug && (
                            <Text dimColor>Ctrl+d to show debug info</Text>
                        )}
                    </Box>
                </>
            )}
        </PermissionDialogLayout>
    );
}

// --- Router (ik2) ---
export function PermissionRequestRouter(props: any) {
    const { toolUseConfirm } = props;
    const parsedInput = toolUseConfirm.tool?.inputSchema?.parse
        ? toolUseConfirm.tool.inputSchema.parse(toolUseConfirm.input)
        : toolUseConfirm.input;
    const parsed = parseMcpCommand(parsedInput?.command ?? "");

    if (parsed) {
        return (
            <McpPermissionRequest
                {...props}
                serverName={parsed.server}
                toolName={parsed.toolName}
                args={parsed.args}
            />
        );
    }

    return (
        <BashPermissionRequest
            {...props}
            command={parsedInput?.command}
            description={parsedInput?.description}
        />
    );
}

// --- Generic Permission Request (mX1) ---
export function GenericPermissionRequest({ toolUseConfirm, onDone, onReject, verbose }: any) {
    const [theme] = useTheme();
    const userFacingName = typeof toolUseConfirm.tool?.userFacingName === "function"
        ? toolUseConfirm.tool.userFacingName(toolUseConfirm.input)
        : toolUseConfirm.tool?.userFacingName;
    const displayName = userFacingName ?? toolUseConfirm.tool?.name ?? "";
    const isMcp = typeof displayName === "string" && displayName.endsWith(" (MCP)");
    const commandName = isMcp ? displayName.slice(0, -6) : displayName;

    const completionMeta = useMemo(() => ({
        completion_type: "tool_use_single",
        language_name: "none"
    }), []);

    logPermissionPrompt(toolUseConfirm, completionMeta);

    const projectLabel = path.basename(getProjectRoot());
    const options = useMemo(() => [
        { label: "Yes", value: "yes" },
        {
            label: (
                <Text>
                    Yes, and don't ask again for <Text bold>{commandName}</Text> commands in <Text bold>{projectLabel}</Text>
                </Text>
            ),
            value: "yes-dont-ask-again"
        },
        {
            label: (
                <Text>
                    No, and tell Claude what to do differently <Text bold>(esc)</Text>
                </Text>
            ),
            value: "no"
        }
    ], [commandName, projectLabel]);

    const handleAction = useCallback((action: string) => {
        switch (action) {
            case "yes":
                logUnaryEvent({
                    completionType: "tool_use_single",
                    event: "accept",
                    languageName: "none",
                    messageId: toolUseConfirm.assistantMessage?.message?.id
                });
                toolUseConfirm.onAllow(toolUseConfirm.input, []);
                onDone();
                break;
            case "yes-dont-ask-again":
                logUnaryEvent({
                    completionType: "tool_use_single",
                    event: "accept",
                    languageName: "none",
                    messageId: toolUseConfirm.assistantMessage?.message?.id
                });
                toolUseConfirm.onAllow(toolUseConfirm.input, [
                    {
                        type: "addRules",
                        rules: [{ toolName: toolUseConfirm.tool?.name }],
                        behavior: "allow",
                        destination: "localSettings"
                    }
                ]);
                onDone();
                break;
            case "no":
                logUnaryEvent({
                    completionType: "tool_use_single",
                    event: "reject",
                    languageName: "none",
                    messageId: toolUseConfirm.assistantMessage?.message?.id
                });
                toolUseConfirm.onReject();
                onReject();
                onDone();
                break;
        }
    }, [onDone, onReject, toolUseConfirm]);

    return (
        <PermissionDialogLayout title="Tool use">
            <Box flexDirection="column" paddingX={2} paddingY={1}>
                <Text>
                    {commandName}({toolUseConfirm.tool?.renderToolUseMessage?.(toolUseConfirm.input, { theme: theme ?? "default", verbose }) ?? ""})
                    {isMcp ? <Text dimColor> (MCP)</Text> : null}
                </Text>
                <Text dimColor>{toolUseConfirm.description}</Text>
            </Box>
            <Box flexDirection="column">
                <PermissionRuleSummary permissionResult={toolUseConfirm.permissionResult} toolType="tool" />
                <Text>Do you want to proceed?</Text>
                <PermissionSelect options={options} onChange={handleAction} onCancel={() => handleAction("no")} />
            </Box>
        </PermissionDialogLayout>
    );
}

function FileWritePreview({ filePath, content }: { filePath: string; content: string }) {
    const { stdout } = useStdout();
    const columns = stdout?.columns ?? 80;
    const exists = fs.existsSync(filePath);
    const original = exists ? fs.readFileSync(filePath, "utf8") : "";
    const patch = exists ? createPatch(filePath, original, content, false, false) : null;

    if (patch && patch.length > 0) {
        return (
            <Box flexDirection="column" borderDimColor borderColor="subtle" borderStyle="round" borderLeft={false} borderRight={false} paddingX={1}>
                {patch.map((hunk, index) => (
                    <DiffView key={`${filePath}-${index}`} patch={hunk} dim={false} width={columns - 2} />
                ))}
            </Box>
        );
    }

    return (
        <Box flexDirection="column" borderDimColor borderColor="subtle" borderStyle="round" borderLeft={false} borderRight={false} paddingX={1}>
            <Text>{content || "(No content)"}</Text>
        </Box>
    );
}

// --- Write File Permission Request (sk2) ---
export function WriteFilePermissionRequest({ toolUseConfirm, toolUseContext, onDone, onReject }: any) {
    const parseInput = toolUseConfirm.tool?.inputSchema?.parse
        ? (value: any) => toolUseConfirm.tool.inputSchema.parse(value)
        : (value: any) => value;
    const parsedInput = parseInput(toolUseConfirm.input);
    const { file_path: filePath, content } = parsedInput;
    const nextContent = content ?? "";
    const exists = fs.existsSync(filePath);
    const title = exists ? "Overwrite file" : "Create file";
    const relativePath = path.relative(getProjectRoot(), filePath);
    const questionAction = exists ? "overwrite" : "create";
    const operationType = exists ? "write" : "create";
    const languageName = path.extname(filePath).replace(".", "") || "none";
    const completionMeta = useMemo(() => ({
        completion_type: "write_file_single",
        language_name: languageName
    }), [languageName]);

    logPermissionPrompt(toolUseConfirm, completionMeta);

    const { options, onChange, focusedOption, setFocusedOption, handleInputModeToggle } = usePermissionOptions({
        filePath,
        completionType: completionMeta.completion_type,
        languageName: completionMeta.language_name,
        toolUseConfirm,
        onDone,
        onReject,
        parseInput,
        operationType,
        onRejectFeedbackChange: () => undefined
    });

    return (
        <PermissionDialogLayout title={title}>
            <Box flexDirection="column" paddingX={2} paddingY={1}>
                <Text dimColor>{relativePath}</Text>
                <Text>
                    Do you want to {questionAction} <Text bold>{path.basename(filePath)}</Text>?
                </Text>
                <FileWritePreview filePath={filePath} content={nextContent} />
            </Box>
            <Box flexDirection="column">
                <PermissionRuleSummary permissionResult={toolUseConfirm.permissionResult} toolType="tool" />
                <Text>Do you want to proceed?</Text>
                <PermissionSelect
                    options={options}
                    onChange={onChange}
                    onCancel={() => onReject()}
                    onFocus={(value) => setFocusedOption(value)}
                    onInputModeToggle={handleInputModeToggle}
                    defaultFocusValue={focusedOption}
                />
            </Box>
        </PermissionDialogLayout>
    );
}

function getToolPath(toolUseConfirm: any) {
    const tool = toolUseConfirm.tool;
    if (!tool) return null;
    if ("getPath" in tool && typeof tool.getPath === "function") {
        try {
            return tool.getPath(toolUseConfirm.input);
        } catch {
            return null;
        }
    }
    return null;
}

function FileToolPermissionRequestWithPath({
    toolUseConfirm,
    onDone,
    onReject,
    verbose,
    filePath,
    userFacingName,
    isReadOnly
}: any) {
    const [theme] = useTheme();
    const completionMeta = useMemo(() => ({
        completion_type: "tool_use_single",
        language_name: "none"
    }), []);

    logPermissionPrompt(toolUseConfirm, completionMeta);

    const { options, onChange, focusedOption, setFocusedOption, handleInputModeToggle } = usePermissionOptions({
        filePath,
        completionType: completionMeta.completion_type,
        languageName: completionMeta.language_name,
        toolUseConfirm,
        onDone,
        onReject,
        parseInput: (value: any) => value,
        operationType: isReadOnly ? "read" : "write"
    });

    return (
        <PermissionDialogLayout title={`${isReadOnly ? "Read" : "Edit"} file`}>
            <Box flexDirection="column" paddingX={2} paddingY={1}>
                <Text>
                    {userFacingName}(
                    {toolUseConfirm.tool?.renderToolUseMessage?.(toolUseConfirm.input, { theme: theme ?? "default", verbose }) ?? ""}
                    )
                </Text>
            </Box>
            <Box flexDirection="column">
                <PermissionRuleSummary permissionResult={toolUseConfirm.permissionResult} toolType="tool" />
                <Text>Do you want to proceed?</Text>
                <PermissionSelect
                    options={options}
                    onChange={onChange}
                    onCancel={() => onReject()}
                    onFocus={(value) => setFocusedOption(value)}
                    onInputModeToggle={handleInputModeToggle}
                    defaultFocusValue={focusedOption}
                />
            </Box>
        </PermissionDialogLayout>
    );
}

export function FileToolPermissionRequest({ toolUseConfirm, onDone, onReject, verbose, toolUseContext }: any) {
    const filePath = getToolPath(toolUseConfirm);
    const userFacingName = typeof toolUseConfirm.tool?.userFacingName === "function"
        ? toolUseConfirm.tool.userFacingName(toolUseConfirm.input)
        : toolUseConfirm.tool?.userFacingName ?? toolUseConfirm.tool?.name ?? "";
    const isReadOnly = typeof toolUseConfirm.tool?.isReadOnly === "function"
        ? toolUseConfirm.tool.isReadOnly(toolUseConfirm.input)
        : false;

    if (!filePath) {
        return (
            <GenericPermissionRequest
                toolUseConfirm={toolUseConfirm}
                toolUseContext={toolUseContext}
                onDone={onDone}
                onReject={onReject}
                verbose={verbose}
            />
        );
    }

    return (
        <FileToolPermissionRequestWithPath
            toolUseConfirm={toolUseConfirm}
            onDone={onDone}
            onReject={onReject}
            verbose={verbose}
            filePath={filePath}
            userFacingName={userFacingName}
            isReadOnly={isReadOnly}
        />
    );
}

export function FetchPermissionRequest(props: any) {
    const { toolUseConfirm } = props;
    const { url, prompt } = toolUseConfirm.input || {};
    return (
        <McpPermissionRequest
            {...props}
            serverName="Builtin"
            toolName="WebFetch"
            args={url + (prompt ? ` with prompt: ${prompt}` : "")}
        />
    );
}

export function NotebookPermissionRequest(props: any) {
    const { toolUseConfirm } = props;
    const { notebook_path, cell_id } = toolUseConfirm.input || {};
    return (
        <McpPermissionRequest
            {...props}
            serverName="Builtin"
            toolName="NotebookEdit"
            args={`${path.basename(notebook_path)}@${cell_id}`}
        />
    );
}
