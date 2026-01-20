import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Box, Text, useInput, useStdout } from "ink";
import InkTextInput from "ink-text-input";
import { randomUUID } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import stringWidth from "string-width";
import { figures } from "../../vendor/terminalFigures.js";
import { formatRelativeTimeAlways } from "../../utils/shared/lodashLikeRuntimeAndEnv.js";
import { KEY_COMBOS } from "../../services/terminal/keybindingConfig.js";
import { getProjectRoot } from "../../utils/shared/pathUtils.js";
import { logTelemetryEvent } from "../../services/telemetry/telemetryInit.js";
import { isSandboxingEnabled } from "../../services/sandbox/sandboxService.js";

const TextInput = InkTextInput as unknown as React.FC<{
    value: string;
    onChange: (value: string) => void;
    onSubmit?: (value: string) => void;
    placeholder?: string;
    showCursor?: boolean;
}>;

const MAX_VISIBLE_MESSAGES = 7;

export type PermissionOption = {
    value: string;
    label: React.ReactNode;
    description?: React.ReactNode;
    type?: "text" | "input";
    placeholder?: string;
    onChange?: (value: string) => void;
    allowEmptySubmit?: boolean;
    initialValue?: string;
    option?: { type: "accept-once" | "accept-session" | "reject" };
};

interface PermissionSelectProps {
    options: PermissionOption[];
    onChange: (value: string, inputValue?: string) => void;
    onCancel?: () => void;
    onFocus?: (value: string) => void;
    onInputModeToggle?: (value: string) => void;
    defaultFocusValue?: string;
    defaultValue?: string | string[];
    inlineDescriptions?: boolean;
    isDisabled?: boolean;
    layout?: "compact-vertical" | "default";
}

export function PermissionSelect({
    options,
    onChange,
    onCancel,
    onFocus,
    onInputModeToggle,
    defaultFocusValue,
    defaultValue,
    inlineDescriptions,
    isDisabled,
    layout = "default"
}: PermissionSelectProps): React.ReactElement {
    const initialValues = useMemo(() => {
        const values: Record<string, string> = {};
        for (const option of options) {
            if (option.initialValue !== undefined) values[option.value] = option.initialValue;
        }
        return values;
    }, [options]);

    const [inputValues, setInputValues] = useState<Record<string, string>>(initialValues);

    const resolvedDefault = useMemo(() => {
        if (defaultFocusValue) return defaultFocusValue;
        if (typeof defaultValue === "string") return defaultValue;
        if (Array.isArray(defaultValue) && defaultValue.length > 0) return defaultValue[0];
        return options[0]?.value;
    }, [defaultFocusValue, defaultValue, options]);

    const [focusedIndex, setFocusedIndex] = useState(() => {
        const idx = options.findIndex((option) => option.value === resolvedDefault);
        return idx >= 0 ? idx : 0;
    });

    const focusedOption = options[focusedIndex];

    useEffect(() => {
        if (!focusedOption || !onFocus) return;
        onFocus(focusedOption.value);
    }, [focusedOption, onFocus]);

    useEffect(() => {
        const idx = options.findIndex((option) => option.value === resolvedDefault);
        if (idx >= 0) setFocusedIndex(idx);
    }, [options, resolvedDefault]);

    useInput(
        useCallback(
            (input, key) => {
                if (isDisabled) return;

                if (key.upArrow || (key.ctrl && input === "p") || (!key.ctrl && input === "k")) {
                    setFocusedIndex((prev) => (prev - 1 + options.length) % options.length);
                    return;
                }
                if (key.downArrow || (key.ctrl && input === "n") || (!key.ctrl && input === "j")) {
                    setFocusedIndex((prev) => (prev + 1) % options.length);
                    return;
                }
                if (key.tab && onInputModeToggle && focusedOption) {
                    onInputModeToggle(focusedOption.value);
                    return;
                }
                if (key.return && focusedOption) {
                    if (focusedOption.type === "input") {
                        const value = inputValues[focusedOption.value] ?? "";
                        if (!focusedOption.allowEmptySubmit && value.trim() === "") return;
                        onChange(focusedOption.value, value);
                    } else {
                        onChange(focusedOption.value);
                    }
                    return;
                }
                if (key.escape && onCancel) {
                    onCancel();
                }
            },
            [focusedOption, inputValues, isDisabled, onCancel, onChange, onInputModeToggle, options.length]
        )
    );

    return (
        <Box flexDirection="column" gap={layout === "compact-vertical" ? 0 : 1}>
            {options.map((option, index) => {
                const isFocused = index === focusedIndex;
                return (
                    <Box key={option.value} flexDirection="column">
                        <Text color={isFocused ? "permission" : undefined}>
                            {isFocused ? figures.pointer : " "} {option.label}
                            {inlineDescriptions && option.description ? (
                                <Text dimColor> — {option.description}</Text>
                            ) : null}
                        </Text>
                        {!inlineDescriptions && option.description && isFocused && (
                            <Text dimColor>{option.description}</Text>
                        )}
                        {isFocused && option.type === "input" && (
                            <Box paddingLeft={2}>
                                <TextInput
                                    value={inputValues[option.value] ?? option.initialValue ?? ""}
                                    onChange={(value) => {
                                        setInputValues((prev) => ({ ...prev, [option.value]: value }));
                                        option.onChange?.(value);
                                    }}
                                    onSubmit={() => {
                                        const value = inputValues[option.value] ?? "";
                                        if (!option.allowEmptySubmit && value.trim() === "") return;
                                        onChange(option.value, value);
                                    }}
                                    placeholder={option.placeholder}
                                    showCursor={true}
                                />
                            </Box>
                        )}
                    </Box>
                );
            })}
        </Box>
    );
}

export function PermissionDialogLayout({
    title,
    color = "permission",
    innerPaddingX = 1,
    children
}: {
    title: string;
    color?: string;
    innerPaddingX?: number;
    children: React.ReactNode;
}): React.ReactElement {
    return (
        <Box flexDirection="column" borderStyle="round" borderColor={color} paddingY={1} paddingX={innerPaddingX} gap={1}>
            <Text bold color={color}>{title}</Text>
            {children}
        </Box>
    );
}

function formatDecisionReason(reason: any, toolType: "tool" | "command") {
    if (!reason) return null;
    switch (reason.type) {
        case "rule":
            return {
                reason: (
                    <Text>
                        Permission rule <Text bold>{reason.rule?.ruleValue ?? ""}</Text> requires confirmation for this {toolType}.
                    </Text>
                ),
                config: reason.rule?.source === "policySettings" ? undefined : "/permissions to update rules"
            };
        case "hook": {
            const suffix = reason.reason ? `: ${reason.reason}` : ".";
            return {
                reason: (
                    <Text>
                        Hook <Text bold>{reason.hookName}</Text> requires confirmation for this {toolType}{suffix}
                    </Text>
                ),
                config: "/hooks to update"
            };
        }
        case "classifier":
            return {
                reason: (
                    <Text>
                        Classifier <Text bold>{reason.classifier}</Text> requires confirmation for this {toolType}.
                        {reason.reason ? `\n${reason.reason}` : ""}
                    </Text>
                )
            };
        default:
            return null;
    }
}

export function PermissionRuleSummary({
    permissionResult,
    toolType
}: {
    permissionResult: any;
    toolType: "tool" | "command";
}): React.ReactElement | null {
    if (!permissionResult) return null;
    const summary = formatDecisionReason(permissionResult.decisionReason, toolType);
    if (!summary) return null;

    return (
        <Box marginBottom={1} flexDirection="column">
            <Text>{summary.reason}</Text>
            {summary.config && <Text dimColor>{summary.config}</Text>}
        </Box>
    );
}

function formatReasonSummary(reason: any) {
    switch (reason.type) {
        case "rule":
            return `${reason.rule?.ruleValue ?? ""} rule from ${reason.rule?.source ?? "unknown"}`;
        case "mode":
            return `${reason.mode} mode`;
        case "sandboxOverride":
            return "Requires permission to bypass sandbox";
        case "workingDir":
            return reason.reason;
        case "other":
            return reason.reason;
        case "permissionPromptTool":
            return `${reason.permissionPromptToolName} permission prompt tool`;
        case "hook":
            return reason.reason ? `${reason.hookName} hook: ${reason.reason}` : `${reason.hookName} hook`;
        case "asyncAgent":
            return reason.reason;
        case "classifier":
            return `${reason.classifier} classifier: ${reason.reason}`;
        default:
            return "Unknown reason";
    }
}

function formatSuggestions(suggestions: any[] | undefined) {
    if (!suggestions || suggestions.length === 0) {
        return (
            <Box flexDirection="row">
                <Box justifyContent="flex-end" minWidth={10}>
                    <Text dimColor>Suggestions </Text>
                </Box>
                <Text>None</Text>
            </Box>
        );
    }

    const ruleSuggestions = suggestions.filter((item) => item.type === "addRules").flatMap((item) => item.rules || []);
    const directorySuggestions = suggestions.filter((item) => item.type === "addDirectories").flatMap((item) => item.directories || []);
    const modeSuggestion = suggestions.find((item) => item.type === "setMode")?.mode;

    if (ruleSuggestions.length === 0 && directorySuggestions.length === 0 && !modeSuggestion) {
        return (
            <Box flexDirection="row">
                <Box justifyContent="flex-end" minWidth={10}>
                    <Text dimColor>Suggestion </Text>
                </Box>
                <Text>None</Text>
            </Box>
        );
    }

    return (
        <Box flexDirection="column">
            <Box flexDirection="row">
                <Box justifyContent="flex-end" minWidth={10}>
                    <Text dimColor>Suggestions </Text>
                </Box>
                <Text> </Text>
            </Box>
            {ruleSuggestions.length > 0 && (
                <Box flexDirection="row">
                    <Box justifyContent="flex-end" minWidth={10}>
                        <Text dimColor>Rules </Text>
                    </Box>
                    <Box flexDirection="column">
                        {ruleSuggestions.map((rule, index) => (
                            <Text key={`${rule.toolName}-${index}`}>• {rule.ruleContent ?? rule.toolName}</Text>
                        ))}
                    </Box>
                </Box>
            )}
            {directorySuggestions.length > 0 && (
                <Box flexDirection="row">
                    <Box justifyContent="flex-end" minWidth={10}>
                        <Text dimColor>Directories </Text>
                    </Box>
                    <Box flexDirection="column">
                        {directorySuggestions.map((dir: string) => (
                            <Text key={dir}>• {dir}</Text>
                        ))}
                    </Box>
                </Box>
            )}
            {modeSuggestion && (
                <Box flexDirection="row">
                    <Box justifyContent="flex-end" minWidth={10}>
                        <Text dimColor>Mode </Text>
                    </Box>
                    <Text>{modeSuggestion}</Text>
                </Box>
            )}
        </Box>
    );
}

export function PermissionDebugInfo({ permissionResult }: { permissionResult: any }) {
    const reason = permissionResult?.decisionReason;
    const suggestions = "suggestions" in (permissionResult ?? {}) ? permissionResult.suggestions : undefined;

    const renderReason = () => {
        if (!reason) return <Text>undefined</Text>;
        if (reason.type === "subcommandResults" && reason.reasons) {
            const entries = reason.reasons instanceof Map ? Array.from(reason.reasons.entries()) : Object.entries(reason.reasons);
            return (
                <Box flexDirection="column">
                    {entries.map(([command, result]: any) => {
                        const icon = result.behavior === "allow" ? figures.tick : figures.cross;
                        return (
                            <Box key={command} flexDirection="column">
                                <Text>{icon} {command}</Text>
                                {result.decisionReason && result.decisionReason.type !== "subcommandResults" && (
                                    <Text>  ⎿ {formatReasonSummary(result.decisionReason)}</Text>
                                )}
                            </Box>
                        );
                    })}
                </Box>
            );
        }
        return <Text>{formatReasonSummary(reason)}</Text>;
    };

    return (
        <Box flexDirection="column">
            <Box flexDirection="row">
                <Box justifyContent="flex-end" minWidth={10}>
                    <Text dimColor>Behavior </Text>
                </Box>
                <Text>{permissionResult.behavior}</Text>
            </Box>
            {permissionResult.behavior !== "allow" && (
                <Box flexDirection="row">
                    <Box justifyContent="flex-end" minWidth={10}>
                        <Text dimColor>Message </Text>
                    </Box>
                    <Text>{permissionResult.message}</Text>
                </Box>
            )}
            <Box flexDirection="row">
                <Box justifyContent="flex-end" minWidth={10}>
                    <Text dimColor>Reason </Text>
                </Box>
                <Box flexDirection="column">{renderReason()}</Box>
            </Box>
            {formatSuggestions(suggestions)}
        </Box>
    );
}

function isEmptyText(text: string | undefined | null) {
    return !text || text.trim().length === 0;
}

function truncateToWidth(text: string, maxWidth: number) {
    if (maxWidth <= 0) return "";
    if (stringWidth(text) <= maxWidth) return text;

    let truncated = text;
    while (truncated.length > 1 && stringWidth(truncated + "…") > maxWidth) {
        truncated = truncated.slice(0, -1);
    }
    return truncated.length > 0 ? `${truncated}…` : "";
}

function extractTagContent(text: string, tag: string): string | null {
    const match = text.match(new RegExp(`<${tag}>([\\s\\S]*?)</${tag}>`));
    return match?.[1]?.trim() ?? null;
}

function MessagePreview({
    userMessage,
    color,
    dimColor,
    isCurrent,
    paddingRight
}: {
    userMessage: any;
    color?: string;
    dimColor?: boolean;
    isCurrent?: boolean;
    paddingRight?: number;
}) {
    const { stdout } = useStdout();
    const columns = stdout?.columns ?? 80;

    if (isCurrent) {
        return (
            <Box width="100%">
                <Text italic color={color} dimColor={dimColor}>(current)</Text>
            </Box>
        );
    }

    const content = userMessage?.message?.content;
    const text: string = typeof content === "string"
        ? content.trim()
        : Array.isArray(content) && content.length > 0
            ? content[content.length - 1]?.type === "text" ? content[content.length - 1].text.trim() : "(no prompt)"
            : "(no prompt)";

    if (isEmptyText(text)) {
        return (
            <Box flexDirection="row" width="100%">
                <Text italic color={color} dimColor={dimColor}>((empty message))</Text>
            </Box>
        );
    }

    if (text.includes("<bash-input>")) {
        const bashInput = extractTagContent(text, "bash-input");
        if (bashInput) {
            return (
                <Box flexDirection="row" width="100%">
                    <Text color="bashBorder">!</Text>
                    <Text color={color} dimColor={dimColor}> {bashInput}</Text>
                </Box>
            );
        }
    }

    if (text.includes("<command-message>")) {
        const message = extractTagContent(text, "command-message");
        const args = extractTagContent(text, "command-args");
        if (message) {
            return (
                <Box flexDirection="row" width="100%">
                    <Text color={color} dimColor={dimColor}>
                        {message.startsWith("The ") ? message : `/${message} ${args ?? ""}`}
                    </Text>
                </Box>
            );
        }
    }

    const preview = paddingRight
        ? truncateToWidth(text, Math.max(0, columns - paddingRight))
        : text.slice(0, 500).split("\n").slice(0, 4).join("\n");

    return (
        <Box flexDirection="row" width="100%">
            <Text color={color} dimColor={dimColor}>{preview}</Text>
        </Box>
    );
}

function DiffStatsInline({ diffStats }: { diffStats?: DiffStats }) {
    if (!diffStats || !diffStats.filesChanged) return null;
    return (
        <>
            <Text color="diffAddedWord">+{diffStats.insertions} </Text>
            <Text color="diffRemovedWord">-{diffStats.deletions}</Text>
        </>
    );
}

function DiffStatsSummary({ diffStats }: { diffStats?: DiffStats }) {
    if (!diffStats) return null;
    if (!diffStats.filesChanged || !diffStats.filesChanged[0]) {
        return <Text dimColor>The code has not changed (nothing will be restored).</Text>;
    }

    const count = diffStats.filesChanged.length;
    let label = "";
    if (count === 1) {
        label = path.basename(diffStats.filesChanged[0] || "");
    } else if (count === 2) {
        const [first, second] = diffStats.filesChanged;
        label = `${path.basename(first || "")} and ${path.basename(second || "")}`;
    } else {
        label = `${path.basename(diffStats.filesChanged[0] || "")} and ${count - 1} other files`;
    }

    return (
        <Text dimColor>
            The code will be restored <DiffStatsInline diffStats={diffStats} /> in {label}.
        </Text>
    );
}

export type DiffStats = {
    filesChanged: string[];
    insertions: number;
    deletions: number;
};

export function calculateDiffStats(messages: any[], startUuid: string, endUuid?: string): DiffStats | undefined {
    const startIndex = messages.findIndex((message) => message.uuid === startUuid);
    if (startIndex === -1) return;

    const endIndex = endUuid ? messages.findIndex((message) => message.uuid === endUuid) : messages.length;
    const stopIndex = endIndex === -1 ? messages.length : endIndex;

    const filesChanged: string[] = [];
    let insertions = 0;
    let deletions = 0;

    for (let i = startIndex + 1; i < stopIndex; i += 1) {
        const message = messages[i];
        if (!message || !message.toolUseResult) continue;

        const result = message.toolUseResult;
        if (!result.filePath || !result.structuredPatch) continue;

        if (!filesChanged.includes(result.filePath)) filesChanged.push(result.filePath);

        try {
            if (result.type === "create" && typeof result.content === "string") {
                insertions += result.content.split(/\r?\n/).length;
            } else {
                for (const patch of result.structuredPatch || []) {
                    insertions += patch.lines.filter((line: string) => line.startsWith("+")).length;
                    deletions += patch.lines.filter((line: string) => line.startsWith("-")).length;
                }
            }
        } catch {
            continue;
        }
    }

    return {
        filesChanged,
        insertions,
        deletions
    };
}

export function isUserMessage(message: any) {
    if (message?.type !== "user") return false;
    if (Array.isArray(message.message?.content) && message.message.content[0]?.type === "tool_result") return false;
    if (message.isMeta) return false;
    const content = message.message?.content;
    const text = typeof content === "string"
        ? content.trim()
        : Array.isArray(content) && content.length > 0
            ? content[content.length - 1]?.type === "text" ? content[content.length - 1].text.trim() : ""
            : "";

    if (
        text.includes("<local-command-stdout>") ||
        text.includes("<local-command-stderr>") ||
        text.includes("<bash-stdout>") ||
        text.includes("<bash-stderr>")
    ) {
        return false;
    }

    return true;
}

export function MessageSelector({
    messages,
    onPreRestore,
    onRestoreMessage,
    onRestoreCode,
    onClose
}: any) {
    const filteredMessages = useMemo(() => messages.filter(isUserMessage), [messages]);
    const placeholderId = useMemo(() => randomUUID(), []);
    const displayMessages = useMemo(
        () => [
            ...filteredMessages,
            {
                uuid: placeholderId,
                type: "user",
                message: { content: [{ type: "text", text: "" }] },
                isPlaceholder: true
            }
        ],
        [filteredMessages, placeholderId]
    );

    const [selectedIndex, setSelectedIndex] = useState(displayMessages.length - 1);
    const [errorMessage, setErrorMessage] = useState<string | undefined>();
    const [confirmMessage, setConfirmMessage] = useState<any | undefined>();
    const [isWorking, setIsWorking] = useState(false);
    const [restoreOption, setRestoreOption] = useState<string>("both");

    const supportsCodeRestore = Boolean(onRestoreCode);
    const [diffStatsByIndex, setDiffStatsByIndex] = useState<Record<number, DiffStats | undefined>>({});

    useEffect(() => {
        void logTelemetryEvent("tengu_message_selector_opened", {});
    }, []);

    useEffect(() => {
        if (!supportsCodeRestore) return;
        const nextStats: Record<number, DiffStats | undefined> = {};
        for (let i = 0; i < displayMessages.length; i += 1) {
            const message = displayMessages[i];
            if (message.uuid === placeholderId) continue;
            const nextMessage = displayMessages[i + 1];
            const diffStats = calculateDiffStats(
                messages,
                message.uuid,
                nextMessage?.uuid !== placeholderId ? nextMessage?.uuid : undefined
            );
            nextStats[i] = diffStats;
        }
        setDiffStatsByIndex(nextStats);
    }, [displayMessages, messages, placeholderId, supportsCodeRestore]);

    const handleCancel = useCallback(() => {
        void logTelemetryEvent("tengu_message_selector_cancelled", {});
        setConfirmMessage(undefined);
        setErrorMessage(undefined);
        onClose();
    }, [onClose]);

    const handleSelect = useCallback(async (target: any) => {
        const messageIndex = messages.indexOf(target);
        const indexFromEnd = messages.length - 1 - messageIndex;

        void logTelemetryEvent("tengu_message_selector_selected", {
            index_from_end: indexFromEnd,
            message_type: target?.type,
            is_current_prompt: false
        });

        if (!messages.includes(target)) {
            onClose();
            return;
        }

        if (supportsCodeRestore) {
            setConfirmMessage(target);
            return;
        }

        onPreRestore();
        setIsWorking(true);
        try {
            await onRestoreMessage(target);
            setIsWorking(false);
            onClose();
        } catch (error) {
            setIsWorking(false);
            setErrorMessage(`Failed to restore the conversation:
${String(error)}`);
        }
    }, [messages, onClose, onPreRestore, onRestoreMessage, supportsCodeRestore]);

    useInput(
        useCallback(
            (input, key) => {
                if (confirmMessage || errorMessage) return;
                if (key.escape) {
                    handleCancel();
                    return;
                }
                if (displayMessages.length === 0 || isWorking) return;

                const goUp = () => setSelectedIndex((prev) => Math.max(0, prev - 1));
                const goDown = () => setSelectedIndex((prev) => Math.min(displayMessages.length - 1, prev + 1));
                const goTop = () => setSelectedIndex(0);
                const goBottom = () => setSelectedIndex(displayMessages.length - 1);

                if (key.return) {
                    const target = displayMessages[selectedIndex];
                    if (!filteredMessages.includes(target)) {
                        onClose();
                        return;
                    }
                    void handleSelect(target);
                    return;
                }
                if (key.upArrow) {
                    if (key.ctrl || key.shift || key.meta) goTop();
                    else goUp();
                }
                if (key.downArrow) {
                    if (key.ctrl || key.shift || key.meta) goBottom();
                    else goDown();
                }
                if (input === "k") goUp();
                if (input === "j") goDown();
                if (input === "K") goTop();
                if (input === "J") goBottom();
            },
            [confirmMessage, displayMessages, errorMessage, filteredMessages, handleCancel, handleSelect, isWorking, onClose, selectedIndex]
        )
    );

    const handleConfirm = useCallback(
        async (value: string) => {
            void logTelemetryEvent("tengu_message_selector_restore_option_selected", {
                option: value
            });

            if (!confirmMessage) {
                setErrorMessage("Message not found.");
                return;
            }
            if (value === "nevermind") {
                setConfirmMessage(undefined);
                return;
            }

            onPreRestore();
            setIsWorking(true);
            setErrorMessage(undefined);

            let codeError: any = null;
            let conversationError: any = null;

            if (value === "code" || value === "both") {
                try {
                    await onRestoreCode?.(confirmMessage);
                } catch (error) {
                    codeError = error;
                }
            }

            if (value === "conversation" || value === "both") {
                try {
                    await onRestoreMessage(confirmMessage);
                } catch (error) {
                    conversationError = error;
                }
            }

            setIsWorking(false);
            setConfirmMessage(undefined);

            if (conversationError && codeError) {
                setErrorMessage(`Failed to restore the conversation and code:
${conversationError}
${codeError}`);
            } else if (conversationError) {
                setErrorMessage(`Failed to restore the conversation:
${conversationError}`);
            } else if (codeError) {
                setErrorMessage(`Failed to restore the code:
${codeError}`);
            } else {
                onClose();
            }
        },
        [confirmMessage, onClose, onPreRestore, onRestoreCode, onRestoreMessage]
    );

    if (!displayMessages.length) {
        return (
            <Box flexDirection="column" width="100%">
                <Text>Nothing to rewind to yet.</Text>
            </Box>
        );
    }

    const diffStatsForConfirm = diffStatsByIndex[selectedIndex];
    const hasCodeChanges = Boolean(diffStatsForConfirm?.filesChanged && diffStatsForConfirm.filesChanged.length > 0);
    const restoreOptions = supportsCodeRestore && hasCodeChanges ? [
        { value: "both", label: "Restore code and conversation" },
        { value: "conversation", label: "Restore conversation" },
        { value: "code", label: "Restore code" },
        { value: "nevermind", label: "Never mind" }
    ] : [
        { value: "conversation", label: "Restore conversation" },
        { value: "nevermind", label: "Never mind" }
    ];

    const visibleFromIndex = Math.max(
        0,
        Math.min(selectedIndex - Math.floor(MAX_VISIBLE_MESSAGES / 2), displayMessages.length - MAX_VISIBLE_MESSAGES)
    );

    const visibleMessages = displayMessages.slice(visibleFromIndex, visibleFromIndex + MAX_VISIBLE_MESSAGES);

    return (
        <Box flexDirection="column" width="100%">
            <Box borderColor="suggestion" flexDirection="column">
                <Text bold color="suggestion">Rewind</Text>
            </Box>

            <Box flexDirection="column" marginX={1} gap={1}>
                {errorMessage && (
                    <Text color="error">Error: {errorMessage}</Text>
                )}

                {!filteredMessages.length && (
                    <Text>Nothing to rewind to yet.</Text>
                )}

                {confirmMessage && filteredMessages.length > 0 ? (
                    <>
                        <Text>
                            Confirm you want to restore {!supportsCodeRestore ? "the conversation " : ""}to the point before you sent this message:
                        </Text>
                        <Box flexDirection="column" paddingLeft={1} borderStyle="single" borderRight={false} borderTop={false} borderBottom={false} borderLeft borderLeftDimColor>
                            <MessagePreview userMessage={confirmMessage} color="text" isCurrent={false} />
                            {confirmMessage.timestamp && (
                                <Text dimColor>({formatRelativeTimeAlways(new Date(confirmMessage.timestamp))})</Text>
                            )}
                        </Box>
                        <Box flexDirection="column">
                            <Text dimColor>
                                {restoreOption === "both" || restoreOption === "conversation" ? "The conversation will be forked." : "The conversation will be unchanged."}
                            </Text>
                            {supportsCodeRestore && hasCodeChanges && (restoreOption === "both" || restoreOption === "code") ? (
                                <DiffStatsSummary diffStats={diffStatsForConfirm} />
                            ) : (
                                <Text dimColor>The code will be unchanged.</Text>
                            )}
                        </Box>
                        <PermissionSelect
                            isDisabled={isWorking}
                            options={restoreOptions}
                            defaultFocusValue={hasCodeChanges ? "both" : "conversation"}
                            onFocus={(value) => setRestoreOption(value)}
                            onChange={(value) => handleConfirm(value)}
                            onCancel={() => setConfirmMessage(undefined)}
                        />
                        {supportsCodeRestore && hasCodeChanges && (
                            <Box marginBottom={1}>
                                <Text dimColor>{figures.warning} Rewinding does not affect files edited manually or via bash.</Text>
                            </Box>
                        )}
                    </>
                ) : (
                    <>
                        <Text>
                            {supportsCodeRestore ? "Restore the code and/or conversation to the point before…" : "Restore and fork the conversation to the point before…"}
                        </Text>
                        <Box width="100%" flexDirection="column">
                            {visibleMessages.map((message, index) => {
                                const absoluteIndex = visibleFromIndex + index;
                                const isSelected = absoluteIndex === selectedIndex;
                                const isPlaceholder = message.uuid === placeholderId;
                                const diffStats = diffStatsByIndex[absoluteIndex];
                                const hasDiff = Boolean(diffStats?.filesChanged && diffStats.filesChanged.length > 0);
                                const diffCount = diffStats?.filesChanged?.length ?? 0;
                                return (
                                    <Box key={message.uuid ?? absoluteIndex} height={supportsCodeRestore ? 3 : 2} overflow="hidden" width="100%" flexDirection="row">
                                        <Box width={2} minWidth={2}>
                                            {isSelected ? (
                                                <Text color="permission" bold>{figures.pointer} </Text>
                                            ) : (
                                                <Text>  </Text>
                                            )}
                                        </Box>
                                        <Box flexDirection="column">
                                            <Box flexShrink={1} height={1} overflow="hidden">
                                                <MessagePreview
                                                    userMessage={message}
                                                    color={isSelected ? "suggestion" : undefined}
                                                    isCurrent={isPlaceholder}
                                                    paddingRight={10}
                                                />
                                            </Box>
                                            {supportsCodeRestore && !isPlaceholder && (
                                                <Box height={1} flexDirection="row">
                                                    {hasDiff ? (
                                                        <Text dimColor={!isSelected} color="inactive">
                                                            {diffCount === 1 && diffStats?.filesChanged?.[0]
                                                                ? `${path.basename(diffStats.filesChanged[0])} `
                                                                : `${diffCount} files changed `}
                                                            <DiffStatsInline diffStats={diffStats} />
                                                        </Text>
                                                    ) : (
                                                        <Text dimColor color="warning">{figures.warning} No code restore</Text>
                                                    )}
                                                </Box>
                                            )}
                                        </Box>
                                    </Box>
                                );
                            })}
                        </Box>
                        <Text dimColor italic>Enter to continue · Esc to exit</Text>
                    </>
                )}
            </Box>
        </Box>
    );
}

function resolvePathForPermissions(filePath: string): string {
    if (!filePath) return getProjectRoot();
    if (path.isAbsolute(filePath)) return path.normalize(filePath);
    return path.resolve(getProjectRoot(), filePath);
}

function getDirectoryForPermission(filePath: string): string {
    const resolved = resolvePathForPermissions(filePath);
    try {
        if (fs.statSync(resolved).isDirectory()) return resolved;
    } catch {
        // fall through
    }
    return path.dirname(resolved);
}

function expandPathVariants(filePath: string): string[] {
    const variants = [filePath];
    try {
        const real = fs.realpathSync(filePath);
        if (real && real !== filePath) variants.push(real);
    } catch {
        // ignore
    }
    return variants;
}

function createReadRule(directoryPath: string, destination: "session" | "localSettings" = "session") {
    const normalized = directoryPath.replace(/\\/g, "/");
    const ruleContent = normalized.endsWith("/") ? `${normalized}**` : `${normalized}/**`;
    return {
        type: "addRules",
        rules: [{ toolName: "Read", ruleContent }],
        behavior: "allow",
        destination
    };
}

function isPathWithinWorkingDir(filePath: string, toolPermissionContext: any): boolean {
    if (!filePath) return false;
    const resolved = resolvePathForPermissions(filePath);
    const projectRoot = getProjectRoot();
    if (resolved.startsWith(projectRoot)) return true;

    const additional = toolPermissionContext?.additionalWorkingDirectories;
    if (Array.isArray(additional)) {
        return additional.some((dir) => resolved.startsWith(dir));
    }
    if (additional instanceof Map) {
        for (const entry of additional.values()) {
            if (resolved.startsWith(entry?.path ?? entry)) return true;
        }
    }
    return false;
}

function getPermissionSuggestionsForPath(filePath: string, operationType: "read" | "write" | "create", toolPermissionContext: any) {
    const requiresDirectory = !isPathWithinWorkingDir(filePath, toolPermissionContext);
    const directory = getDirectoryForPermission(filePath);

    if (operationType === "read" && requiresDirectory) {
        const expanded = expandPathVariants(directory);
        return expanded
            .map((dir) => createReadRule(dir, "session"))
            .filter(Boolean);
    }

    if (operationType === "write" || operationType === "create") {
        const updates: any[] = [
            {
                type: "setMode",
                mode: "acceptEdits",
                destination: "session"
            }
        ];
        if (requiresDirectory) {
            updates.push({
                type: "addDirectories",
                directories: expandPathVariants(directory),
                destination: "session"
            });
        }
        return updates;
    }

    return [
        {
            type: "setMode",
            mode: "acceptEdits",
            destination: "session"
        }
    ];
}

export function getPermissionOptions({
    filePath,
    toolPermissionContext,
    operationType = "write",
    onRejectFeedbackChange,
    onAcceptFeedbackChange,
    yesInputMode = false,
    noInputMode = false,
    acceptFeedbackEnabled = false
}: any): PermissionOption[] {
    const options: PermissionOption[] = [];

    if (acceptFeedbackEnabled && yesInputMode && onAcceptFeedbackChange) {
        options.push({
            type: "input",
            label: "Yes,",
            value: "yes",
            placeholder: "tell Claude what to do next",
            onChange: onAcceptFeedbackChange,
            allowEmptySubmit: true,
            option: { type: "accept-once" }
        });
    } else {
        options.push({
            label: "Yes",
            value: "yes",
            option: { type: "accept-once" }
        });
    }

    const isWorkingDir = isPathWithinWorkingDir(filePath, toolPermissionContext);
    let sessionLabel: React.ReactNode;

    if (isWorkingDir) {
        if (operationType === "read") {
            sessionLabel = "Yes, during this session";
        } else {
            sessionLabel = (
                <Text>
                    Yes, allow all edits during this session <Text bold>({KEY_COMBOS.MOVE_FOCUS.displayText})</Text>
                </Text>
            );
        }
    } else {
        const directory = path.basename(getDirectoryForPermission(filePath)) || "this directory";
        if (operationType === "read") {
            sessionLabel = (
                <Text>
                    Yes, allow reading from <Text bold>{directory}/</Text> during this session
                </Text>
            );
        } else {
            sessionLabel = (
                <Text>
                    Yes, allow all edits in <Text bold>{directory}/</Text> during this session <Text bold>({KEY_COMBOS.MOVE_FOCUS.displayText})</Text>
                </Text>
            );
        }
    }

    options.push({
        label: sessionLabel,
        value: "yes-session",
        option: { type: "accept-session" }
    });

    if (acceptFeedbackEnabled && noInputMode && onRejectFeedbackChange) {
        options.push({
            type: "input",
            label: "No,",
            value: "no",
            placeholder: "tell Claude what to do differently",
            onChange: onRejectFeedbackChange,
            allowEmptySubmit: true,
            option: { type: "reject" }
        });
    } else if (acceptFeedbackEnabled) {
        options.push({
            label: "No",
            value: "no",
            option: { type: "reject" }
        });
    } else if (onRejectFeedbackChange) {
        options.push({
            type: "input",
            label: "No",
            value: "no",
            placeholder: "Type here to tell Claude what to do differently",
            onChange: onRejectFeedbackChange,
            option: { type: "reject" }
        });
    } else {
        options.push({
            label: (
                <Text>
                    No, and tell Claude what to do differently <Text bold>(esc)</Text>
                </Text>
            ),
            value: "no",
            option: { type: "reject" }
        });
    }

    return options;
}

const ACTION_HANDLERS = {
    "accept-once": ({ toolUseConfirm, onDone, input, feedback }: any) => {
        onDone();
        toolUseConfirm.onAllow(input, [], feedback);
    },
    "accept-session": ({ toolUseConfirm, onDone, input, suggestions }: any) => {
        onDone();
        toolUseConfirm.onAllow(input, suggestions);
    },
    reject: ({ toolUseConfirm, onDone, onReject, feedback }: any) => {
        onDone();
        onReject();
        toolUseConfirm.onReject(feedback);
    }
};

export function logUnaryEvent({ completionType, event, languageName, messageId, hasFeedback }: any) {
    void logTelemetryEvent("tengu_unary_event", {
        event,
        completion_type: completionType,
        language_name: languageName,
        message_id: messageId,
        platform: process.platform,
        ...(hasFeedback !== undefined ? { hasFeedback } : {})
    });
}

export function logPermissionPrompt(toolUseConfirm: any, completion: { completion_type: string; language_name: string }) {
    useEffect(() => {
        void logTelemetryEvent("tengu_tool_use_show_permission_request", {
            messageID: toolUseConfirm.assistantMessage?.message?.id,
            toolName: toolUseConfirm.tool?.name,
            isMcp: toolUseConfirm.tool?.isMcp ?? false,
            decisionReasonType: toolUseConfirm.permissionResult?.decisionReason?.type,
            sandboxEnabled: isSandboxingEnabled()
        });

        Promise.resolve(completion.language_name).then((languageName) => {
            logUnaryEvent({
                completionType: completion.completion_type,
                event: "response",
                languageName,
                messageId: toolUseConfirm.assistantMessage?.message?.id
            });
        });
    }, [completion, toolUseConfirm]);
}

function logUnaryPermissionEvent({ completionType, event, languageName, messageId, hasFeedback }: any) {
    logUnaryEvent({
        completionType,
        event,
        languageName,
        messageId,
        hasFeedback
    });
}

export function usePermissionOptions({
    filePath,
    completionType,
    languageName,
    toolUseConfirm,
    onDone,
    onReject,
    parseInput,
    operationType = "write",
    onRejectFeedbackChange
}: any) {
    const toolPermissionContext = toolUseConfirm?.toolUseContext?.toolPermissionContext ?? toolUseConfirm?.toolPermissionContext;
    const [yesInputMode, setYesInputMode] = useState(false);
    const [noInputMode, setNoInputMode] = useState(false);
    const [acceptFeedback, setAcceptFeedback] = useState("");
    const [focusedOption, setFocusedOption] = useState("yes");
    const acceptFeedbackEnabled = process.env.CLAUDE_CODE_ACCEPT_FEEDBACK === "1";

    const options = useMemo(() => getPermissionOptions({
        filePath,
        toolPermissionContext,
        operationType,
        onRejectFeedbackChange,
        onAcceptFeedbackChange: setAcceptFeedback,
        yesInputMode,
        noInputMode,
        acceptFeedbackEnabled
    }), [filePath, toolPermissionContext, operationType, onRejectFeedbackChange, yesInputMode, noInputMode, acceptFeedbackEnabled]);

    const handleChange = useCallback((value: string, inputValue?: string) => {
        const option = options.find((opt) => opt.value === value)?.option;
        if (!option) return;

        const parsedInput = parseInput ? parseInput(toolUseConfirm.input) : toolUseConfirm.input;
        const suggestions = filePath ? getPermissionSuggestionsForPath(filePath, operationType, toolPermissionContext) : [];

        if (option.type === "accept-once") {
            if (inputValue) {
                void logTelemetryEvent("tengu_accept_with_instructions_submitted", {
                    instructions_length: inputValue.length
                });
            }
            logUnaryPermissionEvent({
                completionType,
                event: "accept",
                languageName,
                messageId: toolUseConfirm.assistantMessage?.message?.id,
                hasFeedback: Boolean(inputValue)
            });
        }
        if (option.type === "accept-session") {
            logUnaryPermissionEvent({
                completionType,
                event: "accept",
                languageName,
                messageId: toolUseConfirm.assistantMessage?.message?.id
            });
        }
        if (option.type === "reject") {
            logUnaryPermissionEvent({
                completionType,
                event: "reject",
                languageName,
                messageId: toolUseConfirm.assistantMessage?.message?.id,
                hasFeedback: Boolean(inputValue)
            });
        }

        const payload = {
            messageId: toolUseConfirm.assistantMessage?.message?.id,
            completionType,
            languageName,
            toolUseConfirm,
            onDone,
            onReject,
            input: parsedInput,
            feedback: inputValue ?? acceptFeedback,
            suggestions
        };

        ACTION_HANDLERS[option.type](payload);
    }, [acceptFeedback, completionType, filePath, languageName, onDone, onReject, options, operationType, parseInput, toolPermissionContext, toolUseConfirm]);

    const handleInputModeToggle = useCallback((value: string) => {
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

    useInput((input, key) => {
        if (KEY_COMBOS.MOVE_FOCUS.check(input, key)) {
            const sessionOption = options.find((option) => option.option?.type === "accept-session");
            if (sessionOption) handleChange(sessionOption.value);
        }
    });

    return {
        options,
        onChange: handleChange,
        acceptFeedback,
        focusedOption,
        setFocusedOption,
        handleInputModeToggle,
        acceptFeedbackEnabled
    };
}
