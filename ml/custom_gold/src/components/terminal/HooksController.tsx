import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Box, Text, useInput } from "ink";
import { PermissionSelect } from "../permissions/PermissionComponents.js";
import {
    AddHookView,
    AddMatcherView,
    ConfirmDeleteHook,
    ConfirmDeleteMatcher,
    HookModal,
    ListHooksView,
    ListMatchersView,
    SaveHookView,
    SelectEventView,
    formatHookCommand
} from "./HooksManagementView.js";
import { useAppState } from "../../contexts/AppStateContext.js";
import { getAllHooks, removeHookFromSettings, diffHooks, type HookEntry } from "../../services/terminal/HookService.js";
import { getSettings, subscribeToSettings, updateSettings } from "../../services/terminal/settings.js";

type HookEvent =
    | "PreToolUse"
    | "PostToolUse"
    | "PostToolUseFailure"
    | "Notification"
    | "UserPromptSubmit"
    | "SessionStart"
    | "SessionEnd"
    | "Stop"
    | "SubagentStart"
    | "SubagentStop"
    | "PreCompact"
    | "PermissionRequest";

type HookConfig = { type: "command" | "prompt"; command?: string; prompt?: string };

type HookEventMetadata = {
    summary: string;
    description: string;
    matcherMetadata?: { fieldToMatch: string; values: string[] };
};

type HooksState =
    | { mode: "select-event" }
    | { mode: "select-matcher"; event: HookEvent; matcherMetadata: HookEventMetadata["matcherMetadata"] }
    | { mode: "add-matcher"; event: HookEvent; matcherMetadata: HookEventMetadata["matcherMetadata"] }
    | { mode: "delete-matcher"; event: HookEvent; matcher: string; matcherMetadata: HookEventMetadata["matcherMetadata"] }
    | { mode: "select-hook"; event: HookEvent; matcher: string }
    | { mode: "add-hook"; event: HookEvent; matcher: string }
    | { mode: "delete-hook"; event: HookEvent; hook: HookEntry }
    | { mode: "save-hook"; event: HookEvent; hookToSave: HookEntry };

function trackEvent(_name: string, _payload?: Record<string, any>) {}

function areHookConfigsEqual(left: HookConfig, right: HookConfig) {
    return JSON.stringify(left) === JSON.stringify(right);
}

function getHookEventMetadata(toolNames: string[]): Record<HookEvent, HookEventMetadata> {
    return {
        PreToolUse: {
            summary: "Before tool execution",
            description: `Input to command is JSON of tool call arguments.
Exit code 0 - stdout/stderr not shown
Exit code 2 - show stderr to model and block tool call
Other exit codes - show stderr to user only but continue with tool call`,
            matcherMetadata: {
                fieldToMatch: "tool_name",
                values: toolNames
            }
        },
        PostToolUse: {
            summary: "After tool execution",
            description: `Input to command is JSON with fields "inputs" (tool call arguments) and "response" (tool call response).
Exit code 0 - stdout shown in transcript mode (ctrl+o)
Exit code 2 - show stderr to model immediately
Other exit codes - show stderr to user only`,
            matcherMetadata: {
                fieldToMatch: "tool_name",
                values: toolNames
            }
        },
        PostToolUseFailure: {
            summary: "After tool execution fails",
            description: `Input to command is JSON with tool_name, tool_input, tool_use_id, error, error_type, is_interrupt, and is_timeout.
Exit code 0 - stdout shown in transcript mode (ctrl+o)
Exit code 2 - show stderr to model immediately
Other exit codes - show stderr to user only`,
            matcherMetadata: {
                fieldToMatch: "tool_name",
                values: toolNames
            }
        },
        Notification: {
            summary: "When notifications are sent",
            description: `Input to command is JSON with notification message and type.
Exit code 0 - stdout/stderr not shown
Other exit codes - show stderr to user only`,
            matcherMetadata: {
                fieldToMatch: "notification_type",
                values: ["permission_prompt", "idle_prompt", "auth_success", "elicitation_dialog"]
            }
        },
        UserPromptSubmit: {
            summary: "When the user submits a prompt",
            description: `Input to command is JSON with original user prompt text.
Exit code 0 - stdout shown to Claude
Exit code 2 - block processing, erase original prompt, and show stderr to user only
Other exit codes - show stderr to user only`
        },
        SessionStart: {
            summary: "When a new session is started",
            description: `Input to command is JSON with session start source.
Exit code 0 - stdout shown to Claude
Blocking errors are ignored
Other exit codes - show stderr to user only`,
            matcherMetadata: {
                fieldToMatch: "source",
                values: ["startup", "resume", "clear", "compact"]
            }
        },
        Stop: {
            summary: "Right before Claude concludes its response",
            description: `Exit code 0 - stdout/stderr not shown
Exit code 2 - show stderr to model and continue conversation
Other exit codes - show stderr to user only`
        },
        SubagentStart: {
            summary: "When a subagent (Task tool call) is started",
            description: `Input to command is JSON with agent_id and agent_type.
Exit code 0 - stdout shown to subagent
Blocking errors are ignored
Other exit codes - show stderr to user only`,
            matcherMetadata: {
                fieldToMatch: "agent_type",
                values: []
            }
        },
        SubagentStop: {
            summary: "Right before a subagent (Task tool call) concludes its response",
            description: `Exit code 0 - stdout/stderr not shown
Exit code 2 - show stderr to subagent and continue having it run
Other exit codes - show stderr to user only`
        },
        PreCompact: {
            summary: "Before conversation compaction",
            description: `Input to command is JSON with compaction details.
Exit code 0 - stdout appended as custom compact instructions
Exit code 2 - block compaction
Other exit codes - show stderr to user only but continue with compaction`,
            matcherMetadata: {
                fieldToMatch: "trigger",
                values: ["manual", "auto"]
            }
        },
        SessionEnd: {
            summary: "When a session is ending",
            description: `Input to command is JSON with session end reason.
Exit code 0 - command completes successfully
Other exit codes - show stderr to user only`,
            matcherMetadata: {
                fieldToMatch: "reason",
                values: ["clear", "logout", "prompt_input_exit", "other"]
            }
        },
        PermissionRequest: {
            summary: "When a permission dialog is displayed",
            description: `Input to command is JSON with tool_name, tool_input, and tool_use_id.
Output JSON with hookSpecificOutput containing decision to allow or deny.
Exit code 0 - use hook decision if provided
Other exit codes - show stderr to user only`,
            matcherMetadata: {
                fieldToMatch: "tool_name",
                values: toolNames
            }
        }
    };
}

function loadHooksConfig() {}

function getPluginHooks(): Record<string, Array<{ matcher?: string; hooks: Array<{ type: string }>; pluginName?: string }>> | null {
    return null;
}

function buildHooksByEvent(settings: any, toolNames: string[]) {
    const hooksByEvent: Record<string, Record<string, HookEntry[]>> = {
        PreToolUse: {},
        PostToolUse: {},
        PostToolUseFailure: {},
        Notification: {},
        UserPromptSubmit: {},
        SessionStart: {},
        SessionEnd: {},
        Stop: {},
        SubagentStart: {},
        SubagentStop: {},
        PreCompact: {},
        PermissionRequest: {}
    };

    const eventMetadata = getHookEventMetadata(toolNames);

    getAllHooks(settings).forEach((config) => {
        const eventBucket = hooksByEvent[config.event];
        if (!eventBucket) return;
        const matcherKey = eventMetadata[config.event as HookEvent].matcherMetadata !== undefined ? config.matcher || "" : "";
        if (!eventBucket[matcherKey]) eventBucket[matcherKey] = [];
        for (const hookConfig of config.hooks) {
            eventBucket[matcherKey].push({
                event: config.event,
                config: hookConfig,
                matcher: config.matcher,
                source: config.source
            });
        }
    });

    const pluginHooks = getPluginHooks();
    if (pluginHooks) {
        for (const [eventName, matcherGroups] of Object.entries(pluginHooks)) {
            const eventBucket = hooksByEvent[eventName];
            if (!eventBucket) continue;
            for (const matcherGroup of matcherGroups) {
                const matcherKey = matcherGroup.matcher || "";
                if (!eventBucket[matcherKey]) eventBucket[matcherKey] = [];
                for (const hook of matcherGroup.hooks) {
                    if (hook.type === "callback") {
                        eventBucket[matcherKey].push({
                            event: eventName,
                            config: { type: "command", command: "[Plugin Hook]" },
                            matcher: matcherGroup.matcher,
                            source: "pluginHook",
                            pluginName: matcherGroup.pluginName
                        });
                    }
                }
            }
        }
    }

    return hooksByEvent;
}

function getMatchersForEvent(hooksByEvent: Record<string, Record<string, HookEntry[]>>, event: HookEvent) {
    const matchers = Object.keys(hooksByEvent[event] || {});
    return matchers.sort((left, right) => left.localeCompare(right));
}

function getHooksForMatcher(hooksByEvent: Record<string, Record<string, HookEntry[]>>, event: HookEvent, matcher?: string | null) {
    const key = matcher ?? "";
    return hooksByEvent[event]?.[key] ?? [];
}

function getMatcherMetadata(event: HookEvent, toolNames: string[]) {
    return getHookEventMetadata(toolNames)[event].matcherMetadata;
}

function getHookEventSummary(event: HookEvent, toolNames: string[]) {
    return getHookEventMetadata(toolNames)[event].summary;
}

function formatActionMessage(event: string, config: HookConfig) {
    return `${event} hook: ${formatHookCommand(config)}`;
}

function getSettingsSnapshot() {
    return {
        userSettings: getSettings("userSettings"),
        projectSettings: getSettings("projectSettings"),
        localSettings: getSettings("localSettings"),
        policySettings: getSettings("policySettings")
    };
}

function useManagedHooksDiff(refreshKey: number) {
    const baselineRef = useRef<any>(null);
    if (baselineRef.current === null) {
        baselineRef.current = getSettingsSnapshot();
    }

    return useMemo(() => {
        const diff = diffHooks(baselineRef.current, getSettingsSnapshot());
        return diff.length > 0 ? diff.join("\n") : null;
    }, [refreshKey]);
}

/**
 * Main Hooks Management Controller (EJ9).
 */
export function HooksController({ toolNames, onExit }: { toolNames: string[]; onExit: (message?: any, options?: any) => void }) {
    const [messages, setMessages] = useState<string[]>([]);
    const [state, setState] = useState<HooksState>({ mode: "select-event" });
    const [refreshKey, setRefreshKey] = useState(0);
    const [disableAllHooksManaged, setDisableAllHooksManaged] = useState(() => {
        return (getSettings("localSettings") as any)?.disableAllHooks === true &&
            (getSettings("policySettings") as any)?.disableAllHooks === true;
    });
    const [restrictedByPolicy, setRestrictedByPolicy] = useState(() => {
        return (getSettings("policySettings") as any)?.allowManagedHooksOnly === true;
    });
    const [command, setCommand] = useState("");
    const [newMatcher, setNewMatcher] = useState("");
    const [appState] = useAppState();
    const { mcp } = appState;

    useEffect(() => {
        return subscribeToSettings((source) => {
            if (source !== "policySettings") return;
            const localSettings = getSettings("localSettings") as any;
            const policySettings = getSettings("policySettings") as any;
            setDisableAllHooksManaged(localSettings?.disableAllHooks === true && policySettings?.disableAllHooks === true);
            setRestrictedByPolicy(policySettings?.allowManagedHooksOnly === true);
        });
    }, []);

    const combinedToolNames = useMemo(() => {
        const mcpTools = mcp?.tools?.map((tool: any) => tool.name) ?? [];
        return [...toolNames, ...mcpTools];
    }, [toolNames, mcp?.tools]);

    const hooksByEvent = useMemo(() => buildHooksByEvent(getSettingsSnapshot(), combinedToolNames), [combinedToolNames, refreshKey]);
    const mode = state.mode;
    const selectedEvent = "event" in state ? state.event : "PreToolUse";
    const selectedMatcher = "matcher" in state ? state.matcher : null;
    const matchersForEvent = useMemo(() => getMatchersForEvent(hooksByEvent, selectedEvent), [hooksByEvent, selectedEvent]);
    const hooksForMatcher = useMemo(
        () => getHooksForMatcher(hooksByEvent, selectedEvent, selectedMatcher),
        [hooksByEvent, selectedEvent, selectedMatcher]
    );

    const hookEventMetadata = getHookEventMetadata(combinedToolNames);
    const configDifference = useManagedHooksDiff(refreshKey);

    useEffect(() => {
        loadHooksConfig();
    }, []);

    useInput((_, key) => {
        if (mode === "save-hook") return;
        if (key.escape) {
            switch (mode) {
                case "select-event":
                    if (messages.length > 0) onExit(messages.join("\n"));
                    else onExit("Hooks dialog dismissed", { display: "system" });
                    break;
                case "select-matcher":
                    setState({ mode: "select-event" });
                    break;
                case "add-matcher":
                    if ("event" in state) {
                        setState({ mode: "select-matcher", event: state.event, matcherMetadata: state.matcherMetadata });
                    }
                    setNewMatcher("");
                    break;
                case "delete-matcher":
                    if ("event" in state) {
                        setState({ mode: "select-matcher", event: state.event, matcherMetadata: state.matcherMetadata });
                    }
                    break;
                case "select-hook":
                    if ("event" in state) {
                        const metadata = getMatcherMetadata(state.event, combinedToolNames);
                        if (metadata) {
                            setState({ mode: "select-matcher", event: state.event, matcherMetadata: metadata });
                        } else {
                            setState({ mode: "select-event" });
                        }
                    }
                    break;
                case "add-hook":
                    if ("event" in state && "matcher" in state) {
                        setState({ mode: "select-hook", event: state.event, matcher: state.matcher });
                    }
                    setCommand("");
                    break;
                case "delete-hook":
                    if ("event" in state && state.mode === "delete-hook") {
                        const { hook } = state;
                        setState({ mode: "select-hook", event: state.event, matcher: hook.matcher || "" });
                    }
                    break;
            }
            return;
        }

        switch (mode) {
            case "select-event":
                if (key.return) {
                    const metadata = getMatcherMetadata(selectedEvent, combinedToolNames);
                    if (metadata) {
                        setState({ mode: "select-matcher", event: selectedEvent, matcherMetadata: metadata });
                    } else {
                        setState({ mode: "select-hook", event: selectedEvent, matcher: "" });
                    }
                }
                break;
            case "add-matcher":
                if (key.return && newMatcher.trim() && "event" in state) {
                    setState({ mode: "select-hook", event: state.event, matcher: newMatcher.trim() });
                }
                break;
            case "add-hook":
                if (key.return && command.trim() && "event" in state && "matcher" in state) {
                    const hookToSave: HookEntry = {
                        event: state.event,
                        config: { type: "command", command: command.trim() },
                        matcher: getMatcherMetadata(state.event, combinedToolNames) ? state.matcher : "",
                        source: "localSettings"
                    };
                    setState({ mode: "save-hook", event: state.event, hookToSave });
                }
                break;
        }
    });

    const handleSaveSuccess = useCallback(() => {
        if (state.mode !== "save-hook") return;
        const hook = state.hookToSave;
        setMessages((prev) => [...prev, `Added ${formatActionMessage(hook.event, hook.config)}`]);
        setState({ mode: "select-hook", event: hook.event as HookEvent, matcher: hook.matcher || "" });
        setCommand("");
        setRefreshKey((value) => value + 1);
    }, [state]);

    const handleSaveCancel = useCallback(() => {
        if (state.mode === "save-hook") {
            const hook = state.hookToSave;
            setState({ mode: "select-hook", event: hook.event as HookEvent, matcher: hook.matcher || "" });
        }
        setCommand("");
    }, [state]);

    const handleDeleteHook = useCallback(async () => {
        if (state.mode !== "delete-hook") return;
        const { hook, event } = state;
        await removeHookFromSettings(hook);
        trackEvent("tengu_hook_deleted", {
            event: hook.event,
            source: hook.source,
            has_matcher: hook.matcher ? 1 : 0
        });
        setMessages((prev) => [...prev, `Deleted ${formatActionMessage(hook.event, hook.config)}`]);
        setRefreshKey((value) => value + 1);

        const matcherKey = hook.matcher || "";
        const remaining = hooksByEvent[event]?.[matcherKey]?.filter((entry) => !areHookConfigsEqual(entry.config as any, hook.config as any));
        if (!remaining || remaining.length === 0) {
            const metadata = getMatcherMetadata(event, combinedToolNames);
            if (metadata) {
                setState({ mode: "select-matcher", event, matcherMetadata: metadata });
            } else {
                setState({ mode: "select-event" });
            }
        } else {
            setState({ mode: "select-hook", event, matcher: matcherKey });
        }
    }, [state, hooksByEvent, combinedToolNames]);

    const handleDeleteMatcher = useCallback(() => {
        if (state.mode !== "delete-matcher") return;
        setMessages((prev) => [...prev, `Deleted matcher: ${state.matcher}`]);
        setState({ mode: "select-matcher", event: state.event, matcherMetadata: state.matcherMetadata });
    }, [state]);

    const exitWithSummary = useCallback(() => {
        onExit(messages.length > 0 ? messages.join("\n") : "Hooks dialog dismissed", {
            display: messages.length === 0 ? "system" : undefined
        });
    }, [messages, onExit]);

    const totalHooksCount = useMemo(() => {
        return Object.values(hooksByEvent).reduce((total, matcherMap) => {
            return total + Object.values(matcherMap).reduce((sub, hooks) => sub + hooks.length, 0);
        }, 0);
    }, [hooksByEvent]);

    const disableAllHooks = (getSettings("localSettings") as any)?.disableAllHooks === true;

    if (disableAllHooks) {
        return (
            <HookModal title="Hook Configuration - Disabled" onCancel={exitWithSummary} borderDimColor={false} hideInputGuide={disableAllHooksManaged}>
                <Box flexDirection="column" gap={1}>
                    <Box flexDirection="column">
                        <Text>
                            All hooks are currently <Text bold>disabled</Text>
                            {disableAllHooksManaged ? " by a managed settings file" : ""}. You have{" "}
                            <Text bold>{totalHooksCount}</Text> configured hook{totalHooksCount !== 1 ? "s" : ""} that{" "}
                            {totalHooksCount !== 1 ? "are" : "is"} not running.
                        </Text>
                        <Box marginTop={1}>
                            <Text dimColor>When hooks are disabled:</Text>
                        </Box>
                        <Text dimColor>• No hook commands will execute</Text>
                        <Text dimColor>• StatusLine will not be displayed</Text>
                        <Text dimColor>• Tool operations will proceed without hook validation</Text>
                    </Box>

                    {!disableAllHooksManaged && (
                        <Box flexDirection="column">
                            <Text bold>Options:</Text>
                            <PermissionSelect
                                options={[
                                    { label: "Re-enable all hooks", value: "enable" },
                                    { label: "Exit", value: "exit" }
                                ]}
                                onChange={(value) => {
                                    if (value === "enable") {
                                        updateSettings("localSettings", { disableAllHooks: false });
                                        onExit("Re-enabled all hooks");
                                    } else {
                                        exitWithSummary();
                                    }
                                }}
                                onCancel={exitWithSummary}
                            />
                        </Box>
                    )}
                </Box>
            </HookModal>
        );
    }

    switch (state.mode) {
        case "save-hook":
            return (
                <SaveHookView
                    event={state.hookToSave.event}
                    eventSummary={hookEventMetadata[state.hookToSave.event as HookEvent].summary}
                    config={state.hookToSave.config as HookConfig}
                    matcher={state.hookToSave.matcher ?? ""}
                    onSuccess={handleSaveSuccess}
                    onCancel={handleSaveCancel}
                />
            );
        case "select-event":
            return (
                <SelectEventView
                    hookEventMetadata={hookEventMetadata}
                    totalHooksCount={totalHooksCount}
                    configDifference={configDifference}
                    restrictedByPolicy={restrictedByPolicy}
                    onSelectEvent={(event) => {
                        if (event === "disable-all") {
                            updateSettings("localSettings", { disableAllHooks: true });
                            onExit("All hooks have been disabled");
                        } else {
                            const metadata = getMatcherMetadata(event as HookEvent, combinedToolNames);
                            if (metadata) {
                                setState({ mode: "select-matcher", event: event as HookEvent, matcherMetadata: metadata });
                            } else {
                                setState({ mode: "select-hook", event: event as HookEvent, matcher: "" });
                            }
                        }
                    }}
                    onCancel={exitWithSummary}
                />
            );
        case "select-matcher":
            return (
                <ListMatchersView
                    selectedEvent={state.event}
                    matchersForSelectedEvent={matchersForEvent}
                    hooksByEventAndMatcher={hooksByEvent}
                    eventDescription={hookEventMetadata[state.event].description}
                    onSelect={(matcher: string | null) => {
                        if (matcher === null) {
                            setState({ mode: "add-matcher", event: state.event, matcherMetadata: state.matcherMetadata });
                        } else if ((hooksByEvent[state.event]?.[matcher] || []).length === 0) {
                            setState({
                                mode: "delete-matcher",
                                event: state.event,
                                matcher,
                                matcherMetadata: state.matcherMetadata
                            });
                        } else {
                            setState({ mode: "select-hook", event: state.event, matcher });
                        }
                    }}
                    onCancel={() => setState({ mode: "select-event" })}
                />
            );
        case "add-matcher":
            return (
                <AddMatcherView
                    selectedEvent={state.event}
                    newMatcher={newMatcher}
                    onChangeNewMatcher={setNewMatcher}
                    eventDescription={hookEventMetadata[state.event].description}
                    matcherMetadata={state.matcherMetadata}
                    onCancel={() => {
                        setState({ mode: "select-matcher", event: state.event, matcherMetadata: state.matcherMetadata });
                        setNewMatcher("");
                    }}
                />
            );
        case "delete-matcher":
            return (
                <ConfirmDeleteMatcher
                    selectedMatcher={state.matcher}
                    selectedEvent={state.event}
                    onDelete={handleDeleteMatcher}
                    onCancel={() => setState({ mode: "select-matcher", event: state.event, matcherMetadata: state.matcherMetadata })}
                />
            );
        case "select-hook":
            return (
                <ListHooksView
                    selectedEvent={state.event}
                    selectedMatcher={state.matcher}
                    hooksForSelectedMatcher={hooksForMatcher}
                    hookEventMetadata={hookEventMetadata[state.event]}
                    onSelect={(hook: HookEntry | null) => {
                        if (hook === null) {
                            setState({ mode: "add-hook", event: state.event, matcher: state.matcher });
                        } else {
                            setState({ mode: "delete-hook", event: state.event, hook });
                        }
                    }}
                    onCancel={() => {
                        const metadata = getMatcherMetadata(state.event, combinedToolNames);
                        if (metadata) {
                            setState({ mode: "select-matcher", event: state.event, matcherMetadata: metadata });
                        } else {
                            setState({ mode: "select-event" });
                        }
                    }}
                />
            );
        case "add-hook":
            return (
                <AddHookView
                    selectedEvent={state.event}
                    selectedMatcher={state.matcher}
                    eventDescription={getHookEventSummary(state.event, combinedToolNames)}
                    fullDescription={hookEventMetadata[state.event].description}
                    supportsMatcher={getMatcherMetadata(state.event, combinedToolNames) !== undefined}
                    command={command}
                    onChangeCommand={setCommand}
                    onCancel={() => {
                        setState({ mode: "select-hook", event: state.event, matcher: state.matcher });
                        setCommand("");
                    }}
                />
            );
        case "delete-hook":
            return (
                <ConfirmDeleteHook
                    selectedHook={state.hook}
                    eventSupportsMatcher={getMatcherMetadata(state.event, combinedToolNames) !== undefined}
                    onDelete={handleDeleteHook}
                    onCancel={() => setState({ mode: "select-hook", event: state.event, matcher: state.hook.matcher || "" })}
                />
            );
        default:
            return null;
    }
}

export const HooksTool = {
    type: "local-jsx",
    name: "hooks",
    description: "Manage hook configurations for tool events",
    isEnabled: () => true,
    isHidden: false,
    async call(onDone: (message?: string, options?: any) => void, context: any) {
        trackEvent("tengu_hooks_command", {});
        const toolPermissionContext = (await context.getAppState()).toolPermissionContext ?? { rules: [] };
        const toolNames = toolPermissionContext.rules.map((rule: any) => rule.name).filter(Boolean);
        return <HooksController toolNames={toolNames} onExit={onDone} />;
    },
    userFacingName() {
        return "hooks";
    }
};

export const FilesTool = {
    type: "local",
    name: "files",
    description: "List all files currently in context",
    isEnabled: () => false,
    isHidden: false,
    supportsNonInteractive: true,
    async call(_args: any, context: any) {
        const files = context.readFileState ? extractFilesFromReadState(context.readFileState) : [];
        if (files.length === 0) {
            return { type: "text", value: "No files in context" };
        }
        const projectRoot = process.cwd();
        return {
            type: "text",
            value: `Files in context:\n${files.map((file) => relativePath(projectRoot, file)).join("\n")}`
        };
    },
    userFacingName() {
        return "files";
    }
};

export const HooksConstants = {
    FOLDER_NAME: ".claude",
    AGENTS_DIR: "agents"
};

export function generateSkillFrontmatter(
    name: string,
    description: string,
    tools?: string[],
    model?: string,
    color?: string,
    content = ""
) {
    const normalizedDescription = description.replace(/\n/g, "\\n");
    const toolsLine =
        tools === undefined || (tools.length === 1 && tools[0] === "*")
            ? ""
            : `\ntools: ${tools.join(", ")}`;
    const modelLine = model ? `\nmodel: ${model}` : "";
    const colorLine = color ? `\ncolor: ${color}` : "";
    return `---
name: ${name}
description: ${normalizedDescription}${toolsLine}${modelLine}${colorLine}
---

${content}
`;
}

function extractFilesFromReadState(state: any): string[] {
    if (Array.isArray(state)) return state;
    if (!state) return [];
    if (Array.isArray(state.files)) return state.files;
    if (typeof state === "object") {
        return Object.values(state)
            .flat()
            .filter((value): value is string => typeof value === "string");
    }
    return [];
}

function relativePath(root: string, filePath: string) {
    if (!filePath.startsWith(root)) return filePath;
    return filePath.slice(root.length + 1);
}
