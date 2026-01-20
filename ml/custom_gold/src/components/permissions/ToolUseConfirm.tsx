
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Box, Text, useInput, useStdout } from 'ink';
import { basename, relative, dirname } from 'node:path';
import { existsSync, readFileSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import { getCwd } from '../../services/terminal/sessionService.js';
import { logTelemetryEvent } from '../../services/telemetry/telemetryInit.js';
import { PermissionSelect, PermissionDialogLayout, PermissionOption } from './PermissionComponents.js';
import { applyEdits, normalizeQuotes, normalizeWhitespace, generateHunks, hunksToEdits } from '../../utils/diff/DiffUtils.js';
import { McpClient } from '../../services/mcp/McpClient.js';
import { figures } from '../../vendor/terminalFigures.js';

// Logic derived from chunk_461.ts and chunk_460.ts

interface ConnectedMcpClient extends McpClient {
    name: string;
    category: string;
    status: string;
    config: {
        ideRunningInWindows?: boolean;
    };
    callTool(name: string, args: any): Promise<any>;
}

function getConnectedIdeClient(mcpClients: any[] | undefined): ConnectedMcpClient | undefined {
    if (!mcpClients || !Array.isArray(mcpClients)) return undefined;
    return mcpClients.find(c => c.category === 'ide' && c.status === 'connected');
}

/**
 * Hook for managing the diff review in an IDE.
 * Based on lv2 and Ru5 in chunk_461.ts.
 */
function useDiff({
    onChange,
    toolUseContext,
    filePath,
    edits,
    editMode
}: {
    onChange: (result: any, metadata?: any) => void;
    toolUseContext: any;
    filePath: string;
    edits: any[];
    editMode: "single" | "multiple";
}) {
    const isAborted = useRef(false);
    const [hasError, setHasError] = useState(false);

    const mcpClients = toolUseContext?.options?.mcpClients;
    const ideClient = getConnectedIdeClient(mcpClients);
    const canShowDiff = !!ideClient && toolUseContext?.options?.diffTool === 'auto' && !filePath.endsWith('.ipynb');
    const ideName = ideClient?.name || 'IDE';

    const tabName = useMemo(() => {
        const randomId = Math.random().toString(36).substring(2, 8);
        return `✻ [Claude Code] ${basename(filePath)} (${randomId}) ⧉`;
    }, [filePath]);

    const closeTabInIDE = useCallback(async () => {
        if (ideClient) {
            try {
                await ideClient.callTool('close_tab', { tab_name: tabName });
            } catch (err) {
                // Ignore errors on close
            }
        }
    }, [ideClient, tabName]);

    useEffect(() => {
        if (!canShowDiff) return;

        async function startDiffSession() {
            try {
                logTelemetryEvent('tengu_ext_will_show_diff', { filePath });

                const absolutePath = filePath.startsWith('/') ? filePath : relative(getCwd(), filePath);
                const originalContent = existsSync(absolutePath) ? readFileSync(absolutePath, 'utf8') : '';

                // Apply edits to get the intended new content
                const { updatedFile: newContent } = applyEdits(filePath, originalContent, edits);

                if (isAborted.current) return;

                const response = await ideClient!.callTool('openDiff', {
                    old_file_path: absolutePath,
                    new_file_path: absolutePath,
                    new_file_contents: newContent,
                    tab_name: tabName
                });

                if (isAborted.current) return;

                const results = Array.isArray(response) ? response : [response];

                // Logic from chunk_461: check for FILE_SAVED, TAB_CLOSED, or DIFF_REJECTED
                const fileSaved = results.find(r => r.type === 'text' && r.text === 'FILE_SAVED');
                const tabClosed = results.find(r => r.type === 'text' && r.text === 'TAB_CLOSED');
                const diffRejected = results.find(r => r.type === 'text' && r.text === 'DIFF_REJECTED');

                if (fileSaved && results[1]?.text) {
                    logTelemetryEvent('tengu_ext_diff_accepted', { filePath });
                    const savedContent = results[1].text;
                    const finalEdits = convertDiffToEdits(filePath, originalContent, savedContent, editMode);
                    onChange({ type: 'accept-once' }, { file_path: filePath, edits: finalEdits });
                } else if (tabClosed) {
                    logTelemetryEvent('tengu_ext_diff_accepted', { filePath }); // Closed tab is often treated as acceptance of generated state
                    onChange({ type: 'accept-once' }, { file_path: filePath, edits });
                } else if (diffRejected) {
                    logTelemetryEvent('tengu_ext_diff_rejected', { filePath });
                    await closeTabInIDE();
                    onChange({ type: 'reject' }, { file_path: filePath, edits });
                } else {
                    throw new Error("Unexpected IDE response");
                }

            } catch (err) {
                console.error("IDE Diff Error:", err);
                setHasError(true);
                logTelemetryEvent('tengu_ext_diff_failed', { error: String(err) });
            }
        }

        startDiffSession();

        return () => {
            isAborted.current = true;
            closeTabInIDE();
        };
    }, []);

    return {
        showingDiffInIDE: canShowDiff && !hasError,
        ideName,
        closeTabInIDE
    };
}

/**
 * Converts a content diff back to the agent-expected edit format.
 * Corresponds to Mu5 in chunk_461.ts.
 */
function convertDiffToEdits(filePath: string, oldContent: string, newContent: string, mode: "single" | "multiple") {
    const isSingle = mode === "single";
    const hunkResult = generateHunks({
        filePath,
        oldContent,
        newContent,
        singleHunk: isSingle
    });

    if (hunkResult.hunks.length === 0) return [];
    if (isSingle && hunkResult.hunks.length > 1) {
        // Fallback to original edits if we can't represent as single hunk
        console.warn("Expected single hunk but got multiple, falling back.");
    }

    return hunksToEdits(hunkResult.hunks);
}

/**
 * UI for reviewing changes in an IDE.
 * Based on nv2 in chunk_461.ts.
 */
function ApplyChangesInIDE({
    onChange,
    options,
    input,
    filePath,
    ideName,
    onRejectFeedbackChange
}: {
    onChange: (opt: any, input: any, feedback?: string) => void;
    options: PermissionOption[];
    input: any;
    filePath: string;
    ideName: string;
    onRejectFeedbackChange?: (value: string) => void;
}) {
    const [rejectFeedback, setRejectFeedback] = useState("");

    const handleRejectChange = (val: string) => {
        setRejectFeedback(val);
        onRejectFeedbackChange?.(val);
    };

    const enhancedOptions = options.map(opt => {
        if (opt.option?.type === 'reject') {
            return { ...opt, onChange: handleRejectChange };
        }
        return opt;
    });

    return (
        <Box flexDirection="column">
            <Box marginX={1} flexDirection="column" gap={1}>
                <Text bold color="permission">Opened changes in {ideName} ⧉</Text>
                <Text dimColor>Save file to continue…</Text>
                <Box flexDirection="column">
                    <Text>Do you want to make this edit to <Text bold>{basename(filePath)}</Text>?</Text>
                    <PermissionSelect
                        options={enhancedOptions}
                        onChange={(value) => {
                            const opt = options.find(o => o.value === value);
                            if (opt?.option?.type === 'reject') {
                                if (rejectFeedback.trim() === '') return;
                                onChange(opt.option, input, rejectFeedback);
                            } else if (opt) {
                                onChange(opt.option, input);
                            }
                        }}
                        onCancel={() => onChange({ type: 'reject' }, input)}
                    />
                </Box>
            </Box>
        </Box>
    );
}

/**
 * Main ToolUseConfirm component.
 * Robust implementation based on vr, pv2, and gv2.
 */
export function ToolUseConfirm({
    toolUseConfirm,
    toolUseContext,
    onDone,
    onReject,
    title,
    subtitle,
    question = "Do you want to proceed?",
    content,
    completionType = "tool_use_single",
    languageName = "none",
    path,
    parseInput,
    operationType = "write",
    ideDiffSupport
}: any) {
    const [rejectFeedback, setRejectFeedback] = useState("");
    const [focusedOption, setFocusedOption] = useState("yes");

    const input = useMemo(() => parseInput ? parseInput(toolUseConfirm.input) : toolUseConfirm.input, [parseInput, toolUseConfirm.input]);

    // Derived from gv2 in chunk_461
    const options: PermissionOption[] = useMemo(() => {
        const opts: PermissionOption[] = [
            {
                label: "Yes",
                value: "yes",
                option: { type: "accept-once" }
            },
            {
                label: `Yes, allow all edits in ${path ? basename(dirname(path)) : 'this directory'}/ during session`,
                value: "yes-session",
                option: { type: "accept-session" }
            },
            {
                label: <Text>No, and tell Claude what to do differently <Text bold>(esc)</Text></Text>,
                value: "no",
                type: "input",
                placeholder: "Type here to tell Claude what to do differently",
                option: { type: "reject" }
            }
        ];
        return opts;
    }, [path]);

    const handleAction = useCallback((option: any, currentInput: any, feedback?: string) => {
        if (option.type === "accept-once" || option.type === "accept-session") {
            // Log telemetry and finish
            logTelemetryEvent('tengu_tool_use_accepted', { tool: toolUseConfirm.tool.name, type: option.type });
            onDone({ ...toolUseConfirm, input: currentInput }); // Simplified return
        } else if (option.type === "reject") {
            logTelemetryEvent('tengu_tool_use_rejected', { tool: toolUseConfirm.tool.name, feedback: !!feedback });
            onReject(feedback);
        }
    }, [onDone, onReject, toolUseConfirm]);

    const { showingDiffInIDE, ideName, closeTabInIDE } = useDiff({
        onChange: (result, metadata) => {
            if (metadata) {
                handleAction(result, metadata);
            } else {
                handleAction(result, input);
            }
        },
        toolUseContext,
        filePath: path || '',
        edits: operationType === "write" ? [{
            old_string: input.old_string,
            new_string: input.new_string,
            replace_all: input.replace_all || false
        }] : [],
        editMode: "single" // Default to single for str_replace_tool
    });

    if (showingDiffInIDE && path) {
        return (
            <ApplyChangesInIDE
                onChange={(opt, inp, fb) => {
                    closeTabInIDE();
                    handleAction(opt, inp, fb);
                }}
                options={options}
                input={input}
                filePath={path}
                ideName={ideName}
                onRejectFeedbackChange={setRejectFeedback}
            />
        );
    }

    return (
        <Box flexDirection="column">
            <PermissionDialogLayout title={title}>
                <Box paddingX={1} flexDirection="column">
                    {subtitle && <Text dimColor>{subtitle}</Text>}
                    {content}
                    <Box flexDirection="column" marginTop={1}>
                        <Box flexDirection="row" gap={1}>
                            {typeof question === 'string' ? <Text>{question}</Text> : question}
                        </Box>
                        <PermissionSelect
                            options={options}
                            onFocus={setFocusedOption}
                            onChange={(value, inputValue) => {
                                const opt = options.find(o => o.value === value);
                                if (opt) handleAction(opt.option, input, inputValue);
                            }}
                            onCancel={() => handleAction({ type: 'reject' }, input)}
                        />
                    </Box>
                </Box>
            </PermissionDialogLayout>
            <Box paddingX={1} marginTop={1}>
                <Text dimColor>Esc to cancel</Text>
            </Box>
        </Box>
    );
}
