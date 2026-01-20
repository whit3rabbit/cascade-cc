import React, { useMemo, useState } from "react";
import { Box, Text, useStdout } from "ink";
import InkTextInput from "ink-text-input";
import { figures } from "../../vendor/terminalFigures.js";
import { PermissionSelect } from "../permissions/PermissionComponents.js";
import { Shortcut } from "../shared/Shortcut.js";
import ExternalLink from "./ExternalLink.js";
import { addHookToSettings } from "../../services/terminal/HookService.js";

const TextInput = InkTextInput as unknown as React.FC<{
    value: string;
    onChange: (value: string) => void;
    onSubmit?: (value: string) => void;
    showCursor?: boolean;
    columns?: number;
    cursorOffset?: number;
    onChangeCursorOffset?: (offset: number) => void;
    multiline?: boolean;
}>;

type HookConfig = { type: "command" | "prompt"; command?: string; prompt?: string };
type HookEntry = {
    event: string;
    config: HookConfig;
    matcher?: string;
    source: string;
    pluginName?: string;
};

type HookEventMetadata = {
    summary: string;
    description: string;
    matcherMetadata?: { fieldToMatch: string; values: string[] };
};

// --- Modal Header Wrapper (I6) ---
export function HookModal({
    title,
    subtitle,
    onCancel: _onCancel,
    color = "permission",
    borderDimColor = false,
    hideInputGuide = false,
    children
}: {
    title: string;
    subtitle?: string;
    onCancel: () => void;
    color?: string;
    borderDimColor?: boolean;
    hideInputGuide?: boolean;
    children: React.ReactNode;
}) {
    return (
        <Box flexDirection="column" paddingX={1} borderStyle="round" borderDimColor={borderDimColor}>
            <Box flexDirection="column" marginBottom={1}>
                <Text bold color={color}>{title}</Text>
                {subtitle && <Text dimColor>{subtitle}</Text>}
            </Box>
            {children}
            {!hideInputGuide && (
                <Box marginTop={1}>
                    <Shortcut shortcut="Esc" action="cancel" />
                </Box>
            )}
        </Box>
    );
}

export function formatHookCommand(config: HookConfig) {
    if (config.type === "command") return config.command ?? "";
    if (config.type === "prompt") return config.prompt ?? "";
    return JSON.stringify(config);
}

export function formatHookSource(source: string) {
    switch (source) {
        case "userSettings":
            return "user";
        case "projectSettings":
            return "project";
        case "localSettings":
            return "local";
        case "policySettings":
            return "policy";
        case "pluginHook":
            return "plugin";
        default:
            return source;
    }
}

export function getHookOriginLabel(source: string) {
    switch (source) {
        case "userSettings":
            return "User settings";
        case "projectSettings":
            return "Project settings";
        case "localSettings":
            return "Local settings";
        case "policySettings":
            return "Policy settings";
        case "pluginHook":
            return "Plugin hook";
        default:
            return source;
    }
}

export function getHookSourceHelp(source: string) {
    switch (source) {
        case "userSettings":
            return "Saved in ~/.claude/settings.json";
        case "projectSettings":
            return "Saved in .claude/settings.json";
        case "localSettings":
            return "Saved in .claude/settings.local.json";
        case "policySettings":
            return "Saved in managed settings";
        case "pluginHook":
            return "Provided by plugin configuration";
        default:
            return source;
    }
}

// --- Save Hook (pY9) ---
export function SaveHookView({
    event,
    eventSummary,
    config,
    matcher,
    onSuccess,
    onCancel
}: {
    event: string;
    eventSummary: string;
    config: HookConfig;
    matcher: string;
    onSuccess: () => void;
    onCancel: () => void;
}) {
    const [saving, setSaving] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const hookSources = [
        { label: "Local Settings (.claude/settings.local.json)", value: "localSettings" },
        { label: "Project Settings (.claude/settings.json)", value: "projectSettings" },
        { label: "User Settings (~/.claude/settings.json)", value: "userSettings" }
    ];

    const handleSave = async (source: string) => {
        setSaving(true);
        setError(null);
        try {
            await addHookToSettings(event, config, matcher, source);
            onSuccess();
        } catch (err: any) {
            setError(err instanceof Error ? err.message : "Failed to add hook");
            setSaving(false);
        }
    };

    if (saving) {
        return (
            <Box flexDirection="column" gap={1}>
                <Box flexDirection="row" gap={1}>
                    <Text>{figures.circleFilled}</Text>
                    <Text>Adding hook configuration…</Text>
                </Box>
            </Box>
        );
    }

    if (error) {
        return (
            <HookModal title="Failed to add hook" onCancel={onCancel} color="error" borderDimColor={false}>
                <Box flexDirection="column" gap={1}>
                    <Text>{error}</Text>
                    <PermissionSelect
                        options={[{ label: "OK", value: "ok" }]}
                        onChange={onCancel}
                        onCancel={onCancel}
                    />
                </Box>
            </HookModal>
        );
    }

    return (
        <HookModal title="Save hook configuration" onCancel={onCancel} borderDimColor={false}>
            <Box flexDirection="column" gap={1}>
                <Box flexDirection="column" marginX={2}>
                    <Text>Event: {event} - {eventSummary}</Text>
                    <Text>Matcher: {matcher}</Text>
                    <Text>{config.type === "command" ? "Command" : "Prompt"}: {formatHookCommand(config)}</Text>
                </Box>
                <Text>Where should this hook be saved?</Text>
                <PermissionSelect options={hookSources} onChange={handleSave} onCancel={onCancel} />
            </Box>
        </HookModal>
    );
}

// --- Select Event View (iY9) ---
export function SelectEventView({
    hookEventMetadata,
    totalHooksCount,
    configDifference,
    restrictedByPolicy,
    onSelectEvent,
    onCancel
}: {
    hookEventMetadata: Record<string, HookEventMetadata>;
    totalHooksCount: number;
    configDifference?: string | null;
    restrictedByPolicy: boolean;
    onSelectEvent: (event: string | "disable-all") => void;
    onCancel: () => void;
}) {
    const subtitle = `${totalHooksCount} hook${totalHooksCount !== 1 ? "s" : ""}`;
    const options = [
        ...Object.entries(hookEventMetadata).map(([key, meta]) => ({
            label: `${key} - ${meta.summary}`,
            value: key
        })),
        {
            label: <Text dimColor>Disable all hooks</Text>,
            value: "disable-all"
        }
    ];

    return (
        <HookModal title="Hooks" subtitle={subtitle} onCancel={onCancel} borderDimColor={false}>
            <Box flexDirection="column" gap={1}>
                {restrictedByPolicy && (
                    <Box flexDirection="column">
                        <Text color="suggestion">{figures.info} Hooks Restricted by Policy</Text>
                        <Text dimColor>
                            Only hooks from managed settings can run. User-defined hooks from ~/.claude/settings.json,
                            .claude/settings.json, and .claude/settings.local.json are blocked.
                        </Text>
                    </Box>
                )}

                {configDifference && (
                    <Box flexDirection="column">
                        <Text color="warning">{figures.warning} Settings Changed</Text>
                        <Text dimColor>
                            Hook settings have been modified outside of this menu. Review the following changes carefully:
                        </Text>
                        <Text dimColor>{configDifference}</Text>
                    </Box>
                )}

                <PermissionSelect
                    options={options}
                    onChange={(value) => onSelectEvent(value as any)}
                    onCancel={onCancel}
                />
            </Box>
        </HookModal>
    );
}

// --- List Matchers (aY9) ---
export function ListMatchersView({
    selectedEvent,
    matchersForSelectedEvent,
    hooksByEventAndMatcher,
    eventDescription,
    onSelect,
    onCancel
}: any) {
    const options = useMemo(() => {
        return [
            { label: `+ Add new matcher...`, value: "add-new" },
            ...matchersForSelectedEvent.map((matcher: string) => {
                const hooks = hooksByEventAndMatcher[selectedEvent]?.[matcher] || [];
                const sources = Array.from(new Set(hooks.map((hook: HookEntry) => hook.source)));
                return {
                    label: `[${sources.map((source) => formatHookSource(source as string)).join(", ")}] ${matcher}`,
                    value: matcher,
                    description: `${hooks.length} hook${hooks.length !== 1 ? "s" : ""}`
                };
            })
        ];
    }, [matchersForSelectedEvent, hooksByEventAndMatcher, selectedEvent]);

    return (
        <HookModal title={`${selectedEvent} - Tool Matchers`} subtitle={eventDescription} onCancel={onCancel} borderDimColor={false}>
            <Box flexDirection="column">
                <PermissionSelect
                    options={options}
                    onChange={(val) => onSelect(val === "add-new" ? null : val)}
                    onCancel={onCancel}
                />
                {matchersForSelectedEvent.length === 0 && (
                    <Box marginLeft={2}>
                        <Text dimColor>No matchers configured yet</Text>
                    </Box>
                )}
            </Box>
        </HookModal>
    );
}

// --- Add Matcher (sY9) ---
export function AddMatcherView({
    selectedEvent,
    newMatcher,
    onChangeNewMatcher,
    eventDescription,
    matcherMetadata,
    onCancel
}: any) {
    const [cursorOffset, setCursorOffset] = useState(newMatcher.length);
    const { stdout } = useStdout();
    const columns = stdout?.columns || 80;
    return (
        <HookModal title={`Add new matcher for ${selectedEvent}`} subtitle={eventDescription} onCancel={onCancel} borderDimColor={false}>
            <Box flexDirection="column" gap={1}>
                <Box flexDirection="column" gap={1}>
                    <Text>Possible matcher values for field {matcherMetadata.fieldToMatch}:</Text>
                    <Text dimColor>{matcherMetadata.values.join(", ")}</Text>
                </Box>
                <Box flexDirection="column">
                    <Text>Tool matcher:</Text>
                    <Box borderStyle="round" borderDimColor paddingLeft={1} paddingRight={1}>
                        <TextInput
                            value={newMatcher}
                            onChange={onChangeNewMatcher}
                            columns={columns}
                            showCursor={true}
                            cursorOffset={cursorOffset}
                            onChangeCursorOffset={setCursorOffset}
                        />
                    </Box>
                </Box>
                <Box flexDirection="column" gap={1}>
                    <Text dimColor>
                        Example Matchers:{"\n"}• Write (single tool){"\n"}• Write|Edit (multiple tools){"\n"}• Web.* (regex pattern)
                    </Text>
                </Box>
            </Box>
        </HookModal>
    );
}

// --- Add Hook (eY9) ---
export function AddHookView({
    selectedEvent,
    selectedMatcher,
    eventDescription,
    fullDescription,
    supportsMatcher,
    command,
    onChangeCommand,
    onCancel
}: any) {
    const [cursorOffset, setCursorOffset] = useState(command.length);
    const { stdout } = useStdout();
    const columns = stdout?.columns || 80;
    const firstToken = command.trim().split(/\s+/)[0] || "";
    const isRelative = Boolean(firstToken && !firstToken.startsWith("/") && !firstToken.startsWith("~") && firstToken.includes("/"));
    const hasSudo = /\bsudo\b/.test(command);

    return (
        <HookModal title="Add new hook" onCancel={onCancel} borderDimColor={false}>
            <Box flexDirection="column" gap={1}>
                <Box flexDirection="column">
                    <Text dimColor>
                        {figures.info} Hooks execute shell commands with your full user permissions. Only use hooks from trusted sources.{" "}
                        <ExternalLink url="https://code.claude.com/docs/en/hooks" label="Learn more" />
                    </Text>
                </Box>
                <Text>Event: <Text bold>{selectedEvent}</Text> - {eventDescription}</Text>
                {fullDescription && (
                    <Box>
                        <Text dimColor>{fullDescription}</Text>
                    </Box>
                )}
                {supportsMatcher && <Text>Matcher: <Text bold>{selectedMatcher}</Text></Text>}
                <Text>Command:</Text>
                <Box borderStyle="round" borderDimColor paddingLeft={1} paddingRight={1}>
                    <TextInput
                        value={command}
                        onChange={onChangeCommand}
                        columns={columns - 8}
                        showCursor={true}
                        cursorOffset={cursorOffset}
                        onChangeCursorOffset={setCursorOffset}
                        multiline={true}
                    />
                </Box>
                {(isRelative || hasSudo) && (
                    <Box flexDirection="column" gap={0}>
                        {isRelative && (
                            <Text color="warning">
                                {figures.warning} Using a relative path for the executable may be insecure. Consider using an absolute path instead.
                            </Text>
                        )}
                        {hasSudo && (
                            <Text color="warning">
                                {figures.warning} Using sudo in hooks can be dangerous and may expose your system to security risks.
                            </Text>
                        )}
                    </Box>
                )}
                <Text dimColor>
                    Examples:
                    {"\n"}• jq -r '.tool_input.file_path | select(endswith(\".go\"))' | xargs -r gofmt -w
                    {"\n"}• jq -r '\"\\(.tool_input.command) - \\(.tool_input.description // \"No description\")\"'{" >> "}~/.claude/bash-command-log.txt
                    {"\n"}• /usr/local/bin/security_check.sh
                    {"\n"}• python3 ~/hooks/validate_changes.py
                </Text>
            </Box>
        </HookModal>
    );
}

// --- Delete Matcher (QJ9) ---
export function ConfirmDeleteMatcher({
    selectedMatcher,
    selectedEvent,
    onDelete,
    onCancel
}: {
    selectedMatcher: string;
    selectedEvent: string;
    onDelete: () => void;
    onCancel: () => void;
}) {
    return (
        <HookModal title="Delete matcher?" onCancel={onCancel} borderDimColor={false}>
            <Box flexDirection="column" gap={1}>
                <Box flexDirection="column" marginX={2}>
                    <Text bold>{selectedMatcher}</Text>
                    <Text dimColor>Event: {selectedEvent}</Text>
                </Box>
                <Text>This matcher has no hooks configured. Delete it?</Text>
                <PermissionSelect
                    options={[
                        { label: "Yes", value: "yes" },
                        { label: "No", value: "no" }
                    ]}
                    onChange={(value) => (value === "yes" ? onDelete() : onCancel())}
                    onCancel={onCancel}
                />
            </Box>
        </HookModal>
    );
}

// --- List Hooks (GJ9) ---
export function ListHooksView({
    selectedEvent,
    selectedMatcher,
    hooksForSelectedMatcher,
    hookEventMetadata,
    onSelect,
    onCancel
}: {
    selectedEvent: string;
    selectedMatcher: string;
    hooksForSelectedMatcher: HookEntry[];
    hookEventMetadata: HookEventMetadata;
    onSelect: (hook: HookEntry | null) => void;
    onCancel: () => void;
}) {
    const title =
        hookEventMetadata.matcherMetadata !== undefined
            ? `${selectedEvent} - Matcher: ${selectedMatcher}`
            : selectedEvent;

    const options = [
        { label: `+ Add new hook...`, value: "add-new" },
        ...hooksForSelectedMatcher.map((hook, index) => ({
            label: hook.source === "pluginHook" ? `${formatHookCommand(hook.config)} (read-only)` : formatHookCommand(hook.config),
            value: index.toString(),
            description:
                hook.source === "pluginHook"
                    ? `${getHookOriginLabel(hook.source)} - disable ${hook.pluginName ? hook.pluginName : "plugin"} to remove`
                    : getHookOriginLabel(hook.source),
            disabled: hook.source === "pluginHook"
        }))
    ];

    return (
        <HookModal title={title} subtitle={hookEventMetadata.description} onCancel={onCancel} borderDimColor={false}>
            <Box flexDirection="column">
                <PermissionSelect
                    options={options}
                    onChange={(value) => {
                        if (value === "add-new") {
                            onSelect(null);
                            return;
                        }
                        const index = Number.parseInt(value as string, 10);
                        const hook = hooksForSelectedMatcher[index];
                        if (hook) onSelect(hook);
                    }}
                    onCancel={onCancel}
                />
                {hooksForSelectedMatcher.length === 0 && (
                    <Box marginLeft={2}>
                        <Text dimColor>No hooks configured yet</Text>
                    </Box>
                )}
            </Box>
        </HookModal>
    );
}

// --- Delete Hook (YJ9) ---
export function ConfirmDeleteHook({
    selectedHook,
    eventSupportsMatcher,
    onDelete,
    onCancel
}: {
    selectedHook: HookEntry;
    eventSupportsMatcher: boolean;
    onDelete: () => void;
    onCancel: () => void;
}) {
    return (
        <HookModal title="Delete hook?" onCancel={onCancel} borderDimColor={false}>
            <Box flexDirection="column" gap={1}>
                <Box flexDirection="column" marginX={2}>
                    <Text bold>{formatHookCommand(selectedHook.config)}</Text>
                    <Text dimColor>Event: {selectedHook.event}</Text>
                    {eventSupportsMatcher && <Text dimColor>Matcher: {selectedHook.matcher}</Text>}
                    <Text dimColor>{getHookSourceHelp(selectedHook.source)}</Text>
                </Box>
                <Text>This will remove the hook configuration from your settings.</Text>
                <PermissionSelect
                    options={[
                        { label: "Yes", value: "yes" },
                        { label: "No", value: "no" }
                    ]}
                    onChange={(value) => (value === "yes" ? onDelete() : onCancel())}
                    onCancel={onCancel}
                />
            </Box>
        </HookModal>
    );
}
