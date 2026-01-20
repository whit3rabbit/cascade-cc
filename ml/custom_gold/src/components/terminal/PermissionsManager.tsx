// Logic from chunk_538.ts (Permissions UI & Secret Redactor)

import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Box, Text, useInput } from "ink";
import { Select } from "@inkjs/ui";
import path from "node:path";
import chalk from "chalk";

import { useCtrlExit } from "../../hooks/useCtrlExit.js";
import { useTerminalFocus } from "../../hooks/useTerminalFocus.js";
import { useTerminalSize } from "../../vendor/inkContexts.js";
import { figures } from "../../vendor/terminalFigures.js";
import { Shortcut } from "../shared/Shortcut.js";
import { Indent } from "../shared/Indent.js";
import { AddDirectory } from "../onboarding/AddDirectory.js";
import { useAppState } from "../../contexts/AppStateContext.js";
import { getFileSystem } from "../../utils/file-system/fileUtils.js";
import { resolvePath } from "../../services/sandbox/pathResolver.js";
import { permissionReducer, persistPermissionUpdate, PermissionState, PermissionRule, PermissionAction, PermissionDestination } from "../../services/permissions/permissionManager.js";
import { removePersistedPermission } from "../../services/terminal/PermissionService.js";
import { getFriendlySourceName } from "../../services/terminal/HookService.js";
import { trackFeatureUsage } from "../../services/onboarding/usageTracker.js";
import { SETTING_SOURCES } from "../../services/terminal/settings.js";
import {
    AddPermissionRuleInputView,
    PermissionRuleValueDetails,
    WorkspacePermissionsView
} from "../../tools/bash/BashTool.js";
import { AddPermissionRuleView } from "../../services/terminal/HookService.js";

const SelectInput = Select as unknown as React.FC<{
    options: { label: string; value: string }[];
    onChange: (value: string) => void;
    onCancel?: () => void;
    visibleOptionCount?: number;
    isDisabled?: boolean;
    defaultFocusValue?: string;
    onUpFromFirstItem?: () => void;
}>;

function Divider({ dividerColor }: { dividerColor?: string }) {
    const { columns } = useTerminalSize();
    const line = "-".repeat(Math.max(0, columns - 2));
    return (
        <Box>
            <Text color={dividerColor} dimColor={!dividerColor}>
                {line}
            </Text>
        </Box>
    );
}

type PermissionRuleValue = {
    toolName: string;
    ruleContent?: string;
};

type PermissionRuleEntry = {
    source: string;
    ruleBehavior: "allow" | "deny" | "ask";
    ruleValue: PermissionRuleValue;
};

type RuleListOptions = {
    options: { label: string; value: string }[];
    rulesByKey: Map<string, PermissionRuleEntry>;
};

type PermissionContext = PermissionState;

function toPermissionRule(val: PermissionRuleValue): PermissionRule {
    return {
        toolName: val.toolName,
        ruleContent: val.ruleContent || ""
    };
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

function getRuleSources(): string[] {
    return [...SETTING_SOURCES, "cliArg", "command", "session"];
}

function getRulesForBehavior(context: PermissionContext, behavior: "allow" | "deny" | "ask"): PermissionRuleEntry[] {
    const key = behavior === "allow" ? "alwaysAllowRules" : behavior === "deny" ? "alwaysDenyRules" : "alwaysAskRules";
    const store = context?.[key] ?? {};

    const rules: PermissionRuleEntry[] = [];
    for (const source of getRuleSources()) {
        const values = store[source] || [];
        for (const value of values) {
            rules.push({
                source,
                ruleBehavior: behavior,
                ruleValue: parsePermissionRuleValue(value)
            });
        }
    }

    return rules;
}

function deletePermissionRule({
    rule,
    initialContext,
    setToolPermissionContext
}: {
    rule: PermissionRuleEntry;
    initialContext: PermissionContext;
    setToolPermissionContext: (next: PermissionContext) => void;
}) {
    if (rule.source === "policySettings" || rule.source === "flagSettings" || rule.source === "command") {
        throw new Error("Cannot delete permission rules from read-only settings");
    }


    const permissionRule: PermissionRule = {
        toolName: rule.ruleValue.toolName,
        ruleContent: rule.ruleValue.ruleContent || ""
    };

    const updatedContext = permissionReducer(initialContext, {
        type: "removeRules",
        rules: [permissionRule],
        behavior: rule.ruleBehavior,
        destination: rule.source as PermissionDestination
    });

    // We only call removePersistedPermission for persisted sources
    if (["localSettings", "userSettings", "projectSettings"].includes(rule.source)) {
        removePersistedPermission({
            source: rule.source as any,
            ruleBehavior: rule.ruleBehavior,
            ruleValue: formatPermissionRuleValue(rule.ruleValue)
        });
    }

    setToolPermissionContext(updatedContext);
}

// --- Remove Directory Confirm View (M89) ---
export function RemoveDirectoryConfirmView({
    directoryPath,
    onRemove,
    onCancel,
    permissionContext,
    setPermissionContext
}: {
    directoryPath: string;
    onRemove: () => void;
    onCancel: () => void;
    permissionContext: PermissionContext;
    setPermissionContext: (next: PermissionContext) => void;
}) {
    const ctrlExit = useCtrlExit();

    useInput((_, key) => {
        if (key.escape) onCancel();
    });

    const handleRemove = useCallback(() => {
        const nextContext = permissionReducer(permissionContext, {
            type: "removeDirectories",
            directories: [directoryPath],
            destination: "session"
        } as PermissionAction);

        setPermissionContext(nextContext);
        onRemove();
    }, [directoryPath, permissionContext, setPermissionContext, onRemove]);

    const handleChoice = useCallback(
        (value: string) => {
            if (value === "yes") handleRemove();
            else onCancel();
        },
        [handleRemove, onCancel]
    );

    return (
        <>
            <Box
                flexDirection="column"
                borderStyle="round"
                paddingLeft={1}
                paddingRight={1}
                borderColor="error"
            >
                <Text bold color="error">Remove directory from workspace?</Text>
                <Box marginY={1} marginX={2} flexDirection="column">
                    <Text bold>{directoryPath}</Text>
                </Box>
                <Text>Claude Code will no longer have access to files in this directory.</Text>
                <Box marginY={1}>
                    <SelectInput
                        onChange={handleChoice}
                        onCancel={onCancel}
                        options={[
                            { label: "Yes", value: "yes" },
                            { label: "No", value: "no" }
                        ]}
                    />
                </Box>
            </Box>
            <Box marginLeft={3}>
                {ctrlExit.pending ? (
                    <Text dimColor>Press {ctrlExit.keyName} again to exit</Text>
                ) : (
                    <Text dimColor>↑/↓ to select · Enter to confirm · Esc to cancel</Text>
                )}
            </Box>
        </>
    );
}

// --- Tabs (w_) ---
const TabsContext = React.createContext<{ selectedTab?: string; width?: number }>(
    { selectedTab: undefined, width: undefined }
);

export function Tabs({
    title,
    color,
    defaultTab,
    children,
    hidden,
    useFullWidth
}: {
    title?: string;
    color?: string;
    defaultTab?: string;
    children: React.ReactElement<{ id?: string; title: string }>[];
    hidden?: boolean;
    useFullWidth?: boolean;
}) {
    const { columns } = useTerminalSize();
    const tabEntries = children.map((child) => [child.props.id ?? child.props.title, child.props.title]);
    const initialIndex = defaultTab ? tabEntries.findIndex(([id]) => defaultTab === id) : 0;
    const [activeIndex, setActiveIndex] = useState(initialIndex !== -1 ? initialIndex : 0);

    useInput(
        (_input, key) => {
            if (key.tab) {
                const direction = key.shift ? -1 : 1;
                setActiveIndex((current) => (current + tabEntries.length + direction) % tabEntries.length);
            }
        },
        { isActive: !hidden }
    );

    const hint = "(tab to cycle)";
    const titleWidth = title ? title.length + 1 : 0;
    const tabsWidth = tabEntries.reduce((sum, [, label]) => sum + (label?.length ?? 0) + 3, 0);
    const hintWidth = hint.length;
    const totalWidth = titleWidth + tabsWidth + hintWidth;
    const padding = useFullWidth ? Math.max(0, columns - totalWidth - 2) : 0;
    const width = useFullWidth ? columns - 2 : undefined;

    return (
        <TabsContext.Provider
            value={{
                selectedTab: tabEntries[activeIndex]?.[0],
                width
            }}
        >
            <Box flexDirection="column">
                {!hidden && (
                    <Box flexDirection="row" gap={1}>
                        {title !== undefined && (
                            <Text bold color={color}>
                                {title}
                            </Text>
                        )}
                        {tabEntries.map(([id, label], index) => (
                            <Text
                                key={id}
                                backgroundColor={color && activeIndex === index ? color : undefined}
                                color={color && activeIndex === index ? "inverseText" : undefined}
                                bold={activeIndex === index}
                            >
                                {" "}{label}{" "}
                            </Text>
                        ))}
                        <Text dimColor>
                            <Shortcut shortcut="tab" action="cycle" parens />
                        </Text>
                        {padding > 0 && <Text>{" ".repeat(padding)}</Text>}
                    </Box>
                )}
                <Box width={width}>{children}</Box>
            </Box>
        </TabsContext.Provider>
    );
}

// --- Tab (fX) ---
export function Tab({ title, id, children }: { title: string; id?: string; children: React.ReactNode }) {
    const { selectedTab, width } = React.useContext(TabsContext);
    if (selectedTab !== (id ?? title)) return null;
    return <Box width={width}>{children}</Box>;
}

// --- useTabContext (_89) ---
export function useTabContext() {
    const { width } = React.useContext(TabsContext);
    return width;
}

// --- Search Input (nDA) ---
export function SearchInput({
    query,
    placeholder = "Search…",
    isFocused,
    isTerminalFocused,
    prefix = "⌕",
    width
}: {
    query: string;
    placeholder?: string;
    isFocused: boolean;
    isTerminalFocused: boolean;
    prefix?: string;
    width?: number;
}) {
    return (
        <Box
            flexShrink={0}
            borderStyle="round"
            borderColor={isFocused ? "suggestion" : undefined}
            borderDimColor={!isFocused}
            paddingX={1}
            width={width}
        >
            <Text dimColor={!isFocused}>
                {prefix}{" "}
                {isFocused ? (
                    query ? (
                        <>
                            <Text bold>{query}</Text>
                            {isTerminalFocused && <Text color="suggestion">█</Text>}
                        </>
                    ) : (
                        <>
                            {isTerminalFocused && <Text color="suggestion">█</Text>}
                            <Text dimColor>{placeholder}</Text>
                        </>
                    )
                ) : query ? (
                    <Text>{query}</Text>
                ) : (
                    <Text>{placeholder}</Text>
                )}
            </Text>
        </Box>
    );
}

// --- Rule Source (i97) ---
export function RuleSourceView({ rule }: { rule: PermissionRuleEntry }) {
    return <Text dimColor>{`From ${getFriendlySourceName(rule.source)}`}</Text>;
}

// --- Rule Behavior Label (n97) ---
export function formatRuleBehaviorPastTense(behavior: "allow" | "deny" | "ask") {
    switch (behavior) {
        case "allow":
            return "allowed";
        case "deny":
            return "denied";
        case "ask":
            return "ask";
    }
}

// --- Permission Rule Details (a97) ---
export function PermissionRuleDetailsView({
    rule,
    onDelete,
    onCancel
}: {
    rule: PermissionRuleEntry;
    onDelete: () => void;
    onCancel: () => void;
}) {
    const ctrlExit = useCtrlExit();

    useInput((_, key) => {
        if (key.escape) onCancel();
    });

    const ruleDetails = (
        <Box flexDirection="column" marginX={2}>
            <Text bold>{formatPermissionRuleValue(rule.ruleValue)}</Text>
            <PermissionRuleValueDetails ruleValue={rule.ruleValue} />
            <RuleSourceView rule={rule} />
        </Box>
    );

    const footer = (
        <Box marginLeft={3}>
            {ctrlExit.pending ? (
                <Text dimColor>Press {ctrlExit.keyName} again to exit</Text>
            ) : (
                <Text dimColor>Esc to cancel</Text>
            )}
        </Box>
    );

    if (rule.source === "policySettings") {
        return (
            <>
                <Box
                    flexDirection="column"
                    gap={1}
                    borderStyle="round"
                    paddingLeft={1}
                    paddingRight={1}
                    borderColor="permission"
                >
                    <Text bold color="permission">Rule details</Text>
                    {ruleDetails}
                    <Text italic>
                        This rule is configured by managed settings and cannot be modified.{"\n"}
                        Contact your system administrator for more information.
                    </Text>
                </Box>
                {footer}
            </>
        );
    }

    return (
        <>
            <Box
                flexDirection="column"
                gap={1}
                borderStyle="round"
                paddingLeft={1}
                paddingRight={1}
                borderColor="error"
            >
                <Text bold color="error">Delete {formatRuleBehaviorPastTense(rule.ruleBehavior)} tool?</Text>
                {ruleDetails}
                <Text>Are you sure you want to delete this permission rule?</Text>
                <SelectInput
                    onChange={(value) => (value === "yes" ? onDelete() : onCancel())}
                    onCancel={onCancel}
                    options={[
                        { label: "Yes", value: "yes" },
                        { label: "No", value: "no" }
                    ]}
                />
            </Box>
            {footer}
        </>
    );
}

// --- Rule List (o97) ---
export function RuleListView({
    options,
    searchQuery,
    isSearchMode,
    isFocused,
    onSelect,
    onCancel,
    lastFocusedRuleKey,
    onUpFromFirstItem
}: {
    options: { label: string; value: string }[];
    searchQuery: string;
    isSearchMode: boolean;
    isFocused: boolean;
    onSelect: (value: string) => void;
    onCancel: () => void;
    lastFocusedRuleKey?: string;
    onUpFromFirstItem?: () => void;
}) {
    const width = useTabContext();

    return (
        <Box flexDirection="column">
            <Box marginBottom={1} flexDirection="column">
                <SearchInput
                    query={searchQuery}
                    isFocused={isSearchMode}
                    isTerminalFocused={isFocused}
                    width={width}
                />
            </Box>
            <SelectInput
                options={options}
                onChange={onSelect}
                onCancel={onCancel}
                visibleOptionCount={Math.min(10, options.length)}
                isDisabled={isSearchMode}
                defaultFocusValue={lastFocusedRuleKey}
                onUpFromFirstItem={onUpFromFirstItem}
            />
        </Box>
    );
}

// --- Permissions Manager View (AH1) ---
export function PermissionsManagerView({
    onExit,
    initialTab = "allow"
}: {
    onExit: (message?: string, meta?: any) => void;
    initialTab?: "allow" | "deny" | "ask" | "workspace";
}) {
    const [statusMessages, setStatusMessages] = useState<string[]>([]);
    const [{ toolPermissionContext }, setAppState] = useAppState();
    const { isFocused, filterFocusSequences } = useTerminalFocus();
    const [selectedRule, setSelectedRule] = useState<PermissionRuleEntry | undefined>();
    const [lastFocusedRuleKey, setLastFocusedRuleKey] = useState<string | undefined>();
    const [pendingAddRule, setPendingAddRule] = useState<{
        ruleValue: PermissionRuleValue;
        ruleBehavior: "allow" | "deny" | "ask";
    } | null>(null);
    const [pendingAddRules, setPendingAddRules] = useState<{
        ruleValue: PermissionRuleValue;
        ruleBehavior: "allow" | "deny" | "ask";
    } | null>(null);
    const [isAddDirectory, setIsAddDirectory] = useState(false);
    const [removeDirectoryPath, setRemoveDirectoryPath] = useState<string | null>(null);
    const [searchQuery, setSearchQuery] = useState("");
    const [isSearchMode, setIsSearchMode] = useState(false);

    const allowRules = useMemo(() => {
        const map = new Map<string, PermissionRuleEntry>();
        getRulesForBehavior(toolPermissionContext, "allow").forEach((rule) => {
            map.set(JSON.stringify(rule), rule);
        });
        return map;
    }, [toolPermissionContext]);

    const denyRules = useMemo(() => {
        const map = new Map<string, PermissionRuleEntry>();
        getRulesForBehavior(toolPermissionContext, "deny").forEach((rule) => {
            map.set(JSON.stringify(rule), rule);
        });
        return map;
    }, [toolPermissionContext]);

    const askRules = useMemo(() => {
        const map = new Map<string, PermissionRuleEntry>();
        getRulesForBehavior(toolPermissionContext, "ask").forEach((rule) => {
            map.set(JSON.stringify(rule), rule);
        });
        return map;
    }, [toolPermissionContext]);

    const getListOptions = useCallback(
        (behavior: "allow" | "deny" | "ask" | "workspace", query = ""): RuleListOptions => {
            const rulesByKey = (() => {
                switch (behavior) {
                    case "allow":
                        return allowRules;
                    case "deny":
                        return denyRules;
                    case "ask":
                        return askRules;
                    case "workspace":
                        return new Map();
                }
            })();

            const options: { label: string; value: string }[] = [];
            if (behavior !== "workspace" && !query) {
                options.push({
                    label: `Add a new rule${figures.ellipsis}`,
                    value: "add-new-rule"
                });
            }

            const sortedKeys = Array.from(rulesByKey.keys()).sort((left, right) => {
                const leftRule = rulesByKey.get(left);
                const rightRule = rulesByKey.get(right);
                if (leftRule && rightRule) {
                    const leftLabel = formatPermissionRuleValue(leftRule.ruleValue).toLowerCase();
                    const rightLabel = formatPermissionRuleValue(rightRule.ruleValue).toLowerCase();
                    return leftLabel.localeCompare(rightLabel);
                }
                return 0;
            });

            const normalizedQuery = query.toLowerCase();
            for (const key of sortedKeys) {
                const rule = rulesByKey.get(key);
                if (!rule) continue;
                const label = formatPermissionRuleValue(rule.ruleValue);
                if (query && !label.toLowerCase().includes(normalizedQuery)) continue;
                options.push({ label, value: key });
            }

            return { options, rulesByKey };
        },
        [allowRules, denyRules, askRules]
    );

    const ctrlExit = useCtrlExit();

    useInput(
        (input, key) => {
            const isPlainInput = !key.ctrl && !key.meta;

            if (isSearchMode) {
                if (key.escape) {
                    if (searchQuery.length > 0) setSearchQuery("");
                    else setIsSearchMode(false);
                    return;
                }

                if (key.return || key.upArrow || key.downArrow) {
                    setIsSearchMode(false);
                    return;
                }

                if (key.backspace || key.delete) {
                    if (searchQuery.length === 0) {
                        setIsSearchMode(false);
                    } else {
                        setSearchQuery((value) => value.slice(0, -1));
                    }
                    return;
                }

                const filtered = filterFocusSequences(input, key);
                if (filtered && isPlainInput) setSearchQuery((value) => value + filtered);
                return;
            }

            if (input === "/" && isPlainInput) {
                setIsSearchMode(true);
                setSearchQuery("");
                return;
            }

            if (
                isPlainInput &&
                input.length > 0 &&
                input !== "j" &&
                input !== "k" &&
                input !== "m" &&
                input !== "i" &&
                !/^\s+$/.test(input)
            ) {
                const filtered = filterFocusSequences(input, key);
                if (filtered) {
                    setIsSearchMode(true);
                    setSearchQuery(filtered);
                }
            }
        },
        {
            isActive: !selectedRule && !pendingAddRule && !pendingAddRules && !isAddDirectory && !removeDirectoryPath
        }
    );

    const handleSelectRule = useCallback(
        (value: string, behavior: "allow" | "deny" | "ask") => {
            const { rulesByKey } = getListOptions(behavior);
            if (value === "add-new-rule") {
                setPendingAddRule({ ruleValue: { toolName: "" }, ruleBehavior: behavior });
                return;
            }
            setSelectedRule(rulesByKey.get(value));
        },
        [getListOptions]
    );

    const cancelAddRule = useCallback(() => {
        setPendingAddRule(null);
    }, []);

    const submitAddRule = useCallback((ruleValue: PermissionRuleValue, behavior: "allow" | "deny" | "ask") => {
        setPendingAddRules({ ruleValue, ruleBehavior: behavior });
        setPendingAddRule(null);
    }, []);

    const handleAddRules = useCallback((rules: PermissionRuleEntry[]) => {
        setPendingAddRules(null);
        for (const rule of rules) {
            setStatusMessages((prev) => [
                ...prev,
                `Added ${rule.ruleBehavior} rule ${chalk.bold(formatPermissionRuleValue(rule.ruleValue))}`
            ]);
        }
    }, []);

    const cancelAddRules = useCallback(() => {
        setPendingAddRules(null);
    }, []);

    const handleDeleteRule = () => {
        if (!selectedRule) return;

        const { options } = getListOptions(selectedRule.ruleBehavior);
        const selectedKey = JSON.stringify(selectedRule);
        const ruleKeys = options
            .filter((option) => option.value !== "add-new-rule")
            .map((option) => option.value);
        const index = ruleKeys.indexOf(selectedKey);
        let nextFocus: string | undefined;

        if (index !== -1) {
            if (index < ruleKeys.length - 1) nextFocus = ruleKeys[index + 1];
            else if (index > 0) nextFocus = ruleKeys[index - 1];
        }

        setLastFocusedRuleKey(nextFocus);
        deletePermissionRule({
            rule: selectedRule,
            initialContext: toolPermissionContext,
            setToolPermissionContext: (next) => {
                setAppState((prev) => ({
                    ...prev,
                    toolPermissionContext: next
                }));
            }
        });
        setStatusMessages((prev) => [
            ...prev,
            `Deleted ${selectedRule.ruleBehavior} rule ${chalk.bold(formatPermissionRuleValue(selectedRule.ruleValue))}`
        ]);
        setSelectedRule(undefined);
    };

    if (selectedRule) {
        return (
            <PermissionRuleDetailsView
                rule={selectedRule}
                onDelete={handleDeleteRule}
                onCancel={() => setSelectedRule(undefined)}
            />
        );
    }

    if (pendingAddRule) {
        return (
            <AddPermissionRuleInputView
                onCancel={cancelAddRule}
                onSubmit={submitAddRule}
                ruleBehavior={pendingAddRule.ruleBehavior}
            />
        );
    }

    if (pendingAddRules) {
        return (
            <AddPermissionRuleView
                onAddRules={handleAddRules}
                onCancel={cancelAddRules}
                ruleValues={[pendingAddRules.ruleValue]}
                ruleBehavior={pendingAddRules.ruleBehavior}
                initialContext={toolPermissionContext}
                setToolPermissionContext={(next: PermissionContext) => {
                    setAppState((prev) => ({
                        ...prev,
                        toolPermissionContext: next
                    }));
                }}
            />
        );
    }

    if (isAddDirectory) {
        return (
            <AddDirectory
                onAddDirectory={(directoryPath, remember) => {
                    const update = {
                        type: "addDirectories",
                        directories: [directoryPath],
                        destination: remember ? "localSettings" : "session"
                    } as PermissionAction;
                    const updatedContext = permissionReducer(toolPermissionContext, update);

                    setAppState((prev) => ({
                        ...prev,
                        toolPermissionContext: updatedContext
                    }));

                    if (remember) persistPermissionUpdate(update);

                    setStatusMessages((prev) => [
                        ...prev,
                        `Added directory ${chalk.bold(directoryPath)} to workspace${remember ? " and saved to local settings" : " for this session"}`
                    ]);
                    setIsAddDirectory(false);
                }}
                onCancel={() => setIsAddDirectory(false)}
                permissionContext={toolPermissionContext}
            />
        );
    }

    if (removeDirectoryPath) {
        return (
            <RemoveDirectoryConfirmView
                directoryPath={removeDirectoryPath}
                onRemove={() => {
                    setStatusMessages((prev) => [
                        ...prev,
                        `Removed directory ${chalk.bold(removeDirectoryPath)} from workspace`
                    ]);
                    setRemoveDirectoryPath(null);
                }}
                onCancel={() => setRemoveDirectoryPath(null)}
                permissionContext={toolPermissionContext}
                setPermissionContext={(next) => {
                    setAppState((prev) => ({
                        ...prev,
                        toolPermissionContext: next
                    }));
                }}
            />
        );
    }

    function getTabDescription(tab: "allow" | "deny" | "ask" | "workspace") {
        switch (tab) {
            case "allow":
                return "Claude Code won't ask before using allowed tools.";
            case "deny":
                return "Claude Code will always reject requests to use denied tools.";
            case "ask":
                return "Claude Code will always ask for confirmation before using these tools.";
            case "workspace":
                return "Claude Code can read files in the workspace, and make edits when auto-accept edits is on.";
        }
    }

    function renderTabBody(tab: "allow" | "deny" | "ask" | "workspace") {
        if (tab === "workspace") {
            return (
                <WorkspacePermissionsView
                    onExit={onExit}
                    getToolPermissionContext={() => toolPermissionContext}
                    onRequestAddDirectory={() => setIsAddDirectory(true)}
                    onRequestRemoveDirectory={(directoryPath) => setRemoveDirectoryPath(directoryPath)}
                />
            );
        }

        const { options } = getListOptions(tab, searchQuery);

        return (
            <RuleListView
                options={options}
                searchQuery={searchQuery}
                isSearchMode={isSearchMode}
                isFocused={isFocused}
                onSelect={(value) => handleSelectRule(value, tab)}
                onCancel={() => {
                    if (statusMessages.length > 0) {
                        onExit(statusMessages.join("\n"));
                    } else {
                        onExit("Permissions dialog dismissed", { display: "system" });
                    }
                }}
                lastFocusedRuleKey={lastFocusedRuleKey}
                onUpFromFirstItem={() => setIsSearchMode(true)}
            />
        );
    }

    return (
        <Box flexDirection="column" flexShrink={0}>
            <Divider dividerColor="permission" />
            <Box paddingX={1} flexDirection="column" flexShrink={0}>
                <Tabs
                    title="Permissions:"
                    color="permission"
                    defaultTab={initialTab}
                    hidden={!!selectedRule || !!pendingAddRule || !!pendingAddRules || isAddDirectory || !!removeDirectoryPath}
                    useFullWidth
                >
                    <Tab id="allow" title="Allow">
                        <Box flexDirection="column" flexShrink={0}>
                            <Text>{getTabDescription("allow")}</Text>
                            {renderTabBody("allow")}
                        </Box>
                    </Tab>
                    <Tab id="ask" title="Ask">
                        <Box flexDirection="column">
                            <Text>{getTabDescription("ask")}</Text>
                            {renderTabBody("ask")}
                        </Box>
                    </Tab>
                    <Tab id="deny" title="Deny">
                        <Box flexDirection="column">
                            <Text>{getTabDescription("deny")}</Text>
                            {renderTabBody("deny")}
                        </Box>
                    </Tab>
                    <Tab id="workspace" title="Workspace">
                        <Box flexDirection="column">
                            <Text>{getTabDescription("workspace")}</Text>
                            {renderTabBody("workspace")}
                        </Box>
                    </Tab>
                </Tabs>
                <Box marginTop={1}>
                    <Text dimColor>
                        {ctrlExit.pending ? (
                            <>Press {ctrlExit.keyName} again to exit</>
                        ) : (
                            <>Press ↑↓ to navigate · Enter to select · Type to search · Esc to cancel</>
                        )}
                    </Text>
                </Box>
            </Box>
        </Box>
    );
}

// --- Add Directory Result View (s97) ---
export function AddDirectoryResultView({
    message,
    args,
    onDone
}: {
    message: string;
    args: string;
    onDone: () => void;
}) {
    useEffect(() => {
        const timer = setTimeout(onDone, 0);
        return () => clearTimeout(timer);
    }, [onDone]);

    return (
        <Box flexDirection="column">
            <Text dimColor>{`> /add-dir ${args}`}</Text>
            <Indent>
                <Text>{message}</Text>
            </Indent>
        </Box>
    );
}

export function validateDirectoryPath(rawPath: string, permissionContext: PermissionContext) {
    if (!rawPath) {
        return { resultType: "emptyPath" } as const;
    }

    const absolutePath = resolvePath(rawPath);
    const fileSystem = getFileSystem();

    if (!fileSystem.existsSync(absolutePath)) {
        return {
            resultType: "pathNotFound",
            directoryPath: rawPath,
            absolutePath
        } as const;
    }

    if (!fileSystem.statSync(absolutePath).isDirectory()) {
        return {
            resultType: "notADirectory",
            directoryPath: rawPath,
            absolutePath
        } as const;
    }

    const workingDirectories = getWorkingDirectories(permissionContext);
    for (const workingDir of workingDirectories) {
        if (isPathWithinDirectory(absolutePath, workingDir)) {
            return {
                resultType: "alreadyInWorkingDirectory",
                directoryPath: rawPath,
                workingDir
            } as const;
        }
    }

    return {
        resultType: "success",
        absolutePath
    } as const;
}

export function formatAddDirectoryValidationMessage(
    result: ReturnType<typeof validateDirectoryPath>
): string {
    switch (result.resultType) {
        case "emptyPath":
            return "Please provide a directory path.";
        case "pathNotFound":
            return `Path ${chalk.bold(result.absolutePath)} was not found.`;
        case "notADirectory": {
            const parentDir = path.dirname(result.absolutePath);
            return `${chalk.bold(result.directoryPath)} is not a directory. Did you mean to add the parent directory ${chalk.bold(parentDir)}?`;
        }
        case "alreadyInWorkingDirectory":
            return `${chalk.bold(result.directoryPath)} is already accessible within the existing working directory ${chalk.bold(result.workingDir)}.`;
        case "success":
            return `Added ${chalk.bold(result.absolutePath)} as a working directory.`;
    }
}


function getWorkingDirectories(permissionContext: PermissionContext): string[] {
    const workingDirectories = new Set<string>();
    const additional = permissionContext?.additionalWorkingDirectories;

    if (additional instanceof Map) {
        for (const dir of additional.values()) {
            if (dir && dir.path) workingDirectories.add(dir.path);
        }
    }

    if (workingDirectories.size === 0) {
        workingDirectories.add(process.cwd());
    }

    return Array.from(workingDirectories);
}

function isPathWithinDirectory(targetPath: string, baseDir: string): boolean {
    const resolvedTarget = path.resolve(targetPath);
    const resolvedBase = path.resolve(baseDir);
    const relativePath = path.relative(resolvedBase, resolvedTarget);

    return relativePath === "" || (!relativePath.startsWith("..") && !path.isAbsolute(relativePath));
}

// --- /add-dir Command (t97) ---
export const AddDirectoryCommand = {
    type: "local-jsx",
    name: "add-dir",
    description: "Add a new working directory",
    argumentHint: "<path>",
    isEnabled: () => true,
    isHidden: false,
    async call(onDone: (message?: string, meta?: any) => void, context: any, args: string) {
        trackFeatureUsage("multi-directory");
        const trimmed = args.trim();

        if (!trimmed) {
            return <PermissionsManagerView onExit={onDone} initialTab="workspace" />;
        }

        const appState = await context.getAppState();
        const validation = validateDirectoryPath(trimmed, appState.toolPermissionContext);

        if (validation.resultType !== "success") {
            const message = formatAddDirectoryValidationMessage(validation);
            return (
                <AddDirectoryResultView
                    message={message}
                    args={args}
                    onDone={() => onDone(message)}
                />
            );
        }

        return (
            <AddDirectory
                directoryPath={validation.absolutePath}
                permissionContext={appState.toolPermissionContext}
                onAddDirectory={async (directoryPath, remember) => {
                    const update = {
                        type: "addDirectories",
                        directories: [directoryPath],
                        destination: remember ? "localSettings" : "session"
                    } as PermissionAction;

                    const latestState = await context.getAppState();
                    const updatedContext = permissionReducer(latestState.toolPermissionContext, update);

                    context.setAppState((prev: any) => ({
                        ...prev,
                        toolPermissionContext: updatedContext
                    }));

                    let message: string;
                    if (remember) {
                        try {
                            persistPermissionUpdate(update);
                            message = `Added ${chalk.bold(directoryPath)} as a working directory and saved to local settings`;
                        } catch (error) {
                            message = `Added ${chalk.bold(directoryPath)} as a working directory. Failed to save to local settings: ${error instanceof Error ? error.message : "Unknown error"}`;
                        }
                    } else {
                        message = `Added ${chalk.bold(directoryPath)} as a working directory for this session`;
                    }

                    const finalMessage = `${message} ${chalk.dim("· /permissions to manage")}`;
                    onDone(finalMessage);
                }}
                onCancel={() => {
                    onDone(`Did not add ${chalk.bold(validation.absolutePath)} as a working directory.`);
                }}
            />
        );
    },
    userFacingName() {
        return "add-dir";
    }
};

// --- Secret Redactor (c4A) ---
export function redactSecrets(text: string): string {
    let redacted = text;
    redacted = redacted.replace(/"(sk-ant[^\s"']{24,})"/g, '"[REDACTED_API_KEY]"');
    redacted = redacted.replace(/(?<![A-Za-z0-9"'])(sk-ant-?[A-Za-z0-9_-]{10,})(?![A-Za-z0-9"'])/g, "[REDACTED_API_KEY]");
    redacted = redacted.replace(/AWS key: "(AWS[A-Z0-9]{20,})"/g, 'AWS key: "[REDACTED_AWS_KEY]"');
    redacted = redacted.replace(/(AKIA[A-Z0-9]{16})/g, "[REDACTED_AWS_KEY]");
    redacted = redacted.replace(/(?<![A-Za-z0-9])(AIza[A-Za-z0-9_-]{35})(?![A-Za-z0-9])/g, "[REDACTED_GCP_KEY]");
    redacted = redacted.replace(/(?<![A-Za-z0-9])([a-z0-9-]+@[a-z0-9-]+\.iam\.gserviceaccount\.com)(?![A-Za-z0-9])/g, "[REDACTED_GCP_SERVICE_ACCOUNT]");
    redacted = redacted.replace(/(["']?x-api-key["']?\s*[:=]\s*["']?)[^"',\s)}\]]+/gi, "$1[REDACTED_API_KEY]");
    redacted = redacted.replace(/(["']?authorization["']?\s*[:=]\s*["']?(bearer\s+)?)[^"',\s)}\]]+/gi, "$1[REDACTED_TOKEN]");
    redacted = redacted.replace(/(AWS[_-][A-Za-z0-9_]+\s*[=:]\s*)["']?[^"',\s)}\]]+["']?/gi, "$1[REDACTED_AWS_VALUE]");
    redacted = redacted.replace(/(GOOGLE[_-][A-Za-z0-9_]+\s*[=:]\s*)["']?[^"',\s)}\]]+["']?/gi, "$1[REDACTED_GCP_VALUE]");
    redacted = redacted.replace(/((API[-_]?KEY|TOKEN|SECRET|PASSWORD)\s*[=:]\s*)["']?[^"',\s)}\]]+["']?/gi, "$1[REDACTED]");
    return redacted;
}
