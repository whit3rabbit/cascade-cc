// Logic from chunk_566.ts (Agents Manager & Marketplace UI)

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Box, useInput } from "ink";
import { Text } from "../../vendor/inkText.js";
import { figures } from "../../vendor/terminalFigures.js";
import { PermissionSelect } from "../permissions/PermissionComponents.js";
import { useCtrlExit } from "../../hooks/useCtrlExit.js";
import { useAppState, type AppState } from "../../contexts/AppStateContext.js";
import { CreateAgentWizard } from "./CreateAgentWizard.js";
import { formatModelName } from "../../services/claude/modelSettings.js";
import { getFileSystem } from "../../utils/file-system/fileUtils.js";
import InkTextInput from "ink-text-input";
import { resolve as resolvePath } from "path";
import { homedir } from "os";

const TextInput = InkTextInput as unknown as React.FC<{
    value: string;
    onChange: (value: string) => void;
    onSubmit?: (value: string) => void;
    placeholder?: string;
    showCursor?: boolean;
    columns?: number;
    cursorOffset?: number;
    onChangeCursorOffset?: (offset: number) => void;
    focus?: boolean;
}>;

type ToolDefinition = {
    name: string;
    isHidden?: boolean;
};

type ToolUseContext = {
    options?: {
        tools?: ToolDefinition[];
    };
    getAppState?: () => Promise<AppState>;
};

type AgentDefinition = {
    agentType: string;
    whenToUse: string;
    tools?: string[];
    source: string;
    baseDir: string;
    model: string;
    getSystemPrompt: () => string;
    color?: string;
};

type AgentViewState =
    | { mode: "list-agents"; source: string }
    | { mode: "create-agent"; source?: string }
    | { mode: "agent-menu"; agent: AgentDefinition; previousMode: AgentViewState }
    | { mode: "view-agent"; agent: AgentDefinition; previousMode: AgentViewState }
    | { mode: "edit-agent"; agent: AgentDefinition; previousMode: AgentViewState }
    | { mode: "delete-confirm"; agent: AgentDefinition; previousMode: AgentViewState };

type AgentsToolProps = {
    tools: ToolDefinition[];
    onExit: (message?: string, meta?: any) => void;
};

type AgentEditMenuProps = {
    agent: AgentDefinition;
    tools: ToolDefinition[];
    onSaved: (message: string) => void;
    onBack: () => void;
};

type AgentDetailsViewProps = {
    agent: AgentDefinition;
    tools: ToolDefinition[];
    onBack: () => void;
};

type AgentToolsEditorProps = {
    tools: ToolDefinition[];
    initialTools?: string[];
    onComplete: (tools: string[] | undefined) => void;
};

type AgentColorPickerProps = {
    agentName: string;
    currentColor: string;
    onConfirm: (color: string) => void;
};

type AgentModelPickerProps = {
    initialModel: string;
    onComplete: (model: string) => void;
};

type AddMarketplaceViewProps = {
    inputValue: string;
    setInputValue: (value: string) => void;
    cursorOffset: number;
    setCursorOffset: (offset: number) => void;
    error: string | null;
    setError: (error: string | null) => void;
    result: string | null;
    setResult: (value: string | null) => void;
    setViewState: (value: any) => void;
    onAddComplete?: () => Promise<void> | void;
    cliMode?: boolean;
};

const AGENT_SOURCE_LABELS: Record<string, string> = {
    "built-in": "Built-in",
    userSettings: "User settings",
    projectSettings: "Project settings",
    policySettings: "Policy settings",
    localSettings: "Local settings",
    flagSettings: "Flag settings",
    plugin: "Plugin"
};

const DEFAULT_AGENT_SOURCES = [
    "built-in",
    "userSettings",
    "projectSettings",
    "policySettings",
    "flagSettings",
    "plugin"
];

function formatAgentSourceLabel(source: string): string {
    return AGENT_SOURCE_LABELS[source] ?? source;
}

function getAgentSourceSummary(agent: AgentDefinition): string {
    return `Source: ${formatAgentSourceLabel(agent.source)}`;
}

function shouldHideSystemPrompt(agent: AgentDefinition): boolean {
    if (agent.source === "built-in") return true;
    const prompt = agent.getSystemPrompt?.();
    return !prompt;
}

function resolveAgentColorSwatch(agentType: string, color?: string): string | null {
    if (!color || color === "automatic") return null;
    return color;
}

function getAgentFilePath(agent: AgentDefinition): string {
    if (agent.baseDir && agent.baseDir !== "built-in") {
        return resolvePath(agent.baseDir, `${agent.agentType}.md`);
    }
    return resolvePath(process.cwd(), `${agent.agentType}.md`);
}

async function openAgentInEditor(agent: AgentDefinition): Promise<void> {
    const targetPath = getAgentFilePath(agent);
    const editor = (globalThis as any).openExternalEditor;
    if (typeof editor === "function") {
        editor(targetPath);
        return;
    }
    throw new Error("Unable to open editor: openExternalEditor is not available");
}

function canEditAgent(agent: AgentDefinition): boolean {
    return agent.source !== "built-in" && agent.source !== "policySettings";
}

function canWriteAgentConfig(agent: AgentDefinition): boolean {
    return agent.source !== "built-in";
}

async function saveAgentDefinition(
    agent: AgentDefinition,
    whenToUse: string,
    tools: string[] | undefined,
    systemPrompt: string,
    color: string | undefined,
    model: string
): Promise<void> {
    void agent;
    void whenToUse;
    void tools;
    void systemPrompt;
    void color;
    void model;
}

function setAgentColorPreference(agentType: string, color: string): void {
    void agentType;
    void color;
}

function rebuildActiveAgents(allAgents: AgentDefinition[]): AgentDefinition[] {
    return allAgents;
}

function formatEmphasisLabel(label: string): string {
    return label;
}

function getAgentToolStatus(agent: AgentDefinition, tools: ToolDefinition[]) {
    if (!agent.tools) {
        return {
            hasWildcard: true,
            validTools: tools.map((tool) => tool.name),
            invalidTools: [] as string[]
        };
    }

    const toolNames = new Set(tools.map((tool) => tool.name));
    const validTools = agent.tools.filter((toolName) => toolNames.has(toolName));
    const invalidTools = agent.tools.filter((toolName) => !toolNames.has(toolName));

    return {
        hasWildcard: false,
        validTools,
        invalidTools
    };
}

function normalizeAgentTools(
    toolDefs: ToolDefinition[],
    mcpTools: ToolDefinition[] | undefined,
    permissionContext: AppState["toolPermissionContext"] | undefined
): ToolDefinition[] {
    void permissionContext;
    const combined = [...toolDefs, ...(mcpTools ?? [])];
    const seen = new Set<string>();
    return combined.filter((tool) => {
        if (seen.has(tool.name)) return false;
        seen.add(tool.name);
        return !tool.isHidden;
    });
}

function getSelectableToolNames(tools: ToolDefinition[]): string[] {
    return tools.map((tool) => tool.name).sort();
}

function getAgentSourceGroups(allAgents: AgentDefinition[]) {
    const groups: Record<string, AgentDefinition[]> = {};
    DEFAULT_AGENT_SOURCES.forEach((source) => {
        groups[source] = [];
    });

    for (const agent of allAgents) {
        const groupKey = groups[agent.source] ? agent.source : "other";
        groups[groupKey] = groups[groupKey] ?? [];
        groups[groupKey].push(agent);
    }

    groups.all = allAgents;
    return groups;
}

function AgentToolsEditor({ tools, initialTools, onComplete }: AgentToolsEditorProps): React.ReactElement {
    const toolNames = useMemo(() => getSelectableToolNames(tools), [tools]);
    const [selected, setSelected] = useState<Set<string>>(
        new Set(initialTools ?? toolNames)
    );
    const [index, setIndex] = useState(0);

    useInput((input, key) => {
        if (key.upArrow) setIndex((value) => Math.max(0, value - 1));
        else if (key.downArrow) setIndex((value) => Math.min(toolNames.length - 1, value + 1));
        else if (key.return || input === " ") {
            const name = toolNames[index];
            const next = new Set(selected);
            if (next.has(name)) next.delete(name);
            else next.add(name);
            setSelected(next);
        } else if (key.escape) {
            onComplete(Array.from(selected));
        }
    });

    return (
        <Box flexDirection="column">
            <Text bold>Edit tools</Text>
            <Box flexDirection="column" marginTop={1}>
                {toolNames.map((name, idx) => {
                    const isSelected = selected.has(name);
                    return (
                        <Text key={name} color={idx === index ? "suggestion" : undefined}>
                            {idx === index ? figures.pointer : " "} {isSelected ? figures.checkboxOn : figures.checkboxOff}{" "}
                            {name}
                        </Text>
                    );
                })}
            </Box>
            <Box marginTop={1}>
                <Text dimColor>Enter/Space to toggle · Esc to save</Text>
            </Box>
        </Box>
    );
}

function AgentColorPicker({ agentName, currentColor, onConfirm }: AgentColorPickerProps): React.ReactElement {
    const colors = ["automatic", "blue", "green", "magenta", "cyan", "yellow", "red"];
    const [index, setIndex] = useState(Math.max(colors.indexOf(currentColor), 0));

    useInput((input, key) => {
        if (key.upArrow) setIndex((value) => Math.max(0, value - 1));
        else if (key.downArrow) setIndex((value) => Math.min(colors.length - 1, value + 1));
        else if (key.return) onConfirm(colors[index]);
    });

    return (
        <Box flexDirection="column">
            <Text bold>Edit color</Text>
            <Text dimColor>Select a highlight color for {agentName}</Text>
            <Box flexDirection="column" marginTop={1}>
                {colors.map((color, idx) => (
                    <Text key={color} color={idx === index ? "suggestion" : undefined}>
                        {idx === index ? figures.pointer : " "} {color}
                    </Text>
                ))}
            </Box>
        </Box>
    );
}

function AgentModelPicker({ initialModel, onComplete }: AgentModelPickerProps): React.ReactElement {
    const models = ["sonnet", "opus", "haiku", "inherit"];
    const [index, setIndex] = useState(Math.max(models.indexOf(initialModel), 0));

    useInput((input, key) => {
        if (key.upArrow) setIndex((value) => Math.max(0, value - 1));
        else if (key.downArrow) setIndex((value) => Math.min(models.length - 1, value + 1));
        else if (key.return) onComplete(models[index]);
    });

    return (
        <Box flexDirection="column">
            <Text bold>Edit model</Text>
            <Box flexDirection="column" marginTop={1}>
                {models.map((model, idx) => (
                    <Text key={model} color={idx === index ? "suggestion" : undefined}>
                        {idx === index ? figures.pointer : " "} {formatModelName(model)}
                    </Text>
                ))}
            </Box>
        </Box>
    );
}

function InlineSpinner(): React.ReactElement {
    return <Text dimColor>{figures.info} </Text>;
}

function AgentEditMenu({ agent, tools, onSaved, onBack }: AgentEditMenuProps): React.ReactElement | null {
    const [, setAppState] = useAppState();
    const [viewMode, setViewMode] = useState<"menu" | "edit-tools" | "edit-model" | "edit-color">("menu");
    const [selectedIndex, setSelectedIndex] = useState(0);
    const [error, setError] = useState<string | null>(null);
    const [color, setColor] = useState(agent.color);

    const openInEditor = useCallback(async () => {
        try {
            await openAgentInEditor(agent);
            onSaved(
                `Opened ${agent.agentType} in editor. If you made edits, restart to load the latest version.`
            );
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to open editor");
        }
    }, [agent, onSaved]);

    const applyUpdates = useCallback(
        async (updates: { tools?: string[]; color?: string; model?: string } = {}) => {
            const nextColor = updates.color ?? color;
            const hasToolsUpdate = updates.tools !== undefined;
            const hasModelUpdate = updates.model !== undefined;
            const hasColorUpdate = nextColor !== agent.color;

            if (!hasToolsUpdate && !hasModelUpdate && !hasColorUpdate) return false;

            try {
                if (!canEditAgent(agent) && !canWriteAgentConfig(agent)) return false;
                await saveAgentDefinition(
                    agent,
                    agent.whenToUse,
                    updates.tools ?? agent.tools,
                    agent.getSystemPrompt(),
                    nextColor,
                    updates.model ?? agent.model
                );

                if (hasColorUpdate && nextColor) {
                    setAgentColorPreference(agent.agentType, nextColor);
                }

                setAppState((state) => {
                    const updatedAgents = state.agentDefinitions.allAgents.map((entry) =>
                        entry.agentType === agent.agentType
                            ? {
                                ...entry,
                                tools: updates.tools ?? entry.tools,
                                color: nextColor,
                                model: updates.model ?? entry.model
                            }
                            : entry
                    );

                    return {
                        ...state,
                        agentDefinitions: {
                            ...state.agentDefinitions,
                            activeAgents: rebuildActiveAgents(updatedAgents),
                            allAgents: updatedAgents
                        }
                    };
                });

                onSaved(`Updated agent: ${formatEmphasisLabel(agent.agentType)}`);
                return true;
            } catch (err) {
                setError(err instanceof Error ? err.message : "Failed to save agent");
                return false;
            }
        },
        [agent, color, onSaved, setAppState]
    );

    const menuItems = useMemo(
        () => [
            { label: "Open in editor", action: openInEditor },
            { label: "Edit tools", action: () => setViewMode("edit-tools") },
            { label: "Edit model", action: () => setViewMode("edit-model") },
            { label: "Edit color", action: () => setViewMode("edit-color") }
        ],
        [openInEditor]
    );

    const handleBack = useCallback(() => {
        setError(null);
        if (viewMode === "menu") onBack();
        else setViewMode("menu");
    }, [viewMode, onBack]);

    const handleMenuInput = useCallback(
        (key: any) => {
            if (key.upArrow) setSelectedIndex((value) => Math.max(0, value - 1));
            else if (key.downArrow) setSelectedIndex((value) => Math.min(menuItems.length - 1, value + 1));
            else if (key.return) {
                const item = menuItems[selectedIndex];
                if (item) item.action();
            }
        },
        [menuItems, selectedIndex]
    );

    useInput((input, key) => {
        if (key.escape) {
            handleBack();
            return;
        }
        if (viewMode === "menu") handleMenuInput(key);
    });

    const renderMenu = () => (
        <Box flexDirection="column">
            <Text dimColor>{getAgentSourceSummary(agent)}</Text>
            <Box marginTop={1} flexDirection="column">
                {menuItems.map((item, idx) => (
                    <Text key={item.label} color={idx === selectedIndex ? "suggestion" : undefined}>
                        {idx === selectedIndex ? `${figures.pointer} ` : "  "}
                        {item.label}
                    </Text>
                ))}
            </Box>
            {error && (
                <Box marginTop={1}>
                    <Text color="error">{error}</Text>
                </Box>
            )}
        </Box>
    );

    switch (viewMode) {
        case "menu":
            return renderMenu();
        case "edit-tools":
            return (
                <AgentToolsEditor
                    tools={tools}
                    initialTools={agent.tools}
                    onComplete={async (selectedTools) => {
                        setViewMode("menu");
                        await applyUpdates({ tools: selectedTools });
                    }}
                />
            );
        case "edit-color":
            return (
                <AgentColorPicker
                    agentName={agent.agentType}
                    currentColor={color || agent.color || "automatic"}
                    onConfirm={async (nextColor) => {
                        setColor(nextColor);
                        setViewMode("menu");
                        await applyUpdates({ color: nextColor });
                    }}
                />
            );
        case "edit-model":
            return (
                <AgentModelPicker
                    initialModel={agent.model}
                    onComplete={async (model) => {
                        setViewMode("menu");
                        await applyUpdates({ model });
                    }}
                />
            );
        default:
            return null;
    }
}

function AgentDetailsView({ agent, tools, onBack }: AgentDetailsViewProps): React.ReactElement {
    const toolStatus = getAgentToolStatus(agent, tools);
    const sourceLabel = getAgentSourceSummary(agent);
    const colorSwatch = resolveAgentColorSwatch(agent.agentType, agent.color);

    useInput((input, key) => {
        if (key.escape || key.return) onBack();
    });

    const renderTools = () => {
        if (toolStatus.hasWildcard) return <Text>All tools</Text>;
        if (!agent.tools || agent.tools.length === 0) return <Text>None</Text>;

        return (
            <>
                {toolStatus.validTools.length > 0 && (
                    <Text>{toolStatus.validTools.join(", ")}</Text>
                )}
                {toolStatus.invalidTools.length > 0 && (
                    <Text color="warning">
                        {figures.warning} Unrecognized: {toolStatus.invalidTools.join(", ")}
                    </Text>
                )}
            </>
        );
    };

    return (
        <Box flexDirection="column" gap={1}>
            <Text dimColor>{sourceLabel}</Text>
            <Box flexDirection="column">
                <Text bold>Description (tells Claude when to use this agent):</Text>
                <Box marginLeft={2}>
                    <Text>{agent.whenToUse}</Text>
                </Box>
            </Box>
            <Box>
                <Text bold>Tools</Text>: {renderTools()}
            </Box>
            <Text>
                <Text bold>Model</Text>: {formatModelName(agent.model)}
            </Text>
            {colorSwatch && (
                <Box>
                    <Text bold>Color</Text>: {" "}
                    <Text backgroundColor={colorSwatch} color="inverseText">
                        {" "}{agent.agentType}{" "}
                    </Text>
                </Box>
            )}
            {!shouldHideSystemPrompt(agent) && (
                <>
                    <Box>
                        <Text bold>System prompt</Text>:
                    </Box>
                    <Box marginLeft={2} marginRight={2}>
                        <Text wrap="wrap">{agent.getSystemPrompt()}</Text>
                    </Box>
                </>
            )}
        </Box>
    );
}

function ExitHintFooter({
    instructions = "Press ↑↓ to navigate · Enter to select · Esc to go back"
}: {
    instructions?: string;
}): React.ReactElement {
    const exitState = useCtrlExit();

    return (
        <Box marginLeft={3}>
            <Text dimColor>
                {exitState.pending ? `Press ${exitState.keyName} again to exit` : instructions}
            </Text>
        </Box>
    );
}

function AgentListView({
    source,
    agents,
    onBack,
    onSelect,
    onCreateNew,
    changes
}: {
    source: string;
    agents: (AgentDefinition & { overriddenBy?: string })[];
    onBack: () => void;
    onSelect: (agent: AgentDefinition) => void;
    onCreateNew: () => void;
    changes: string[];
}): React.ReactElement {
    const [index, setIndex] = useState(0);
    const entries = useMemo(() => {
        return [
            { label: "+ Create new agent", action: onCreateNew },
            ...agents.map((agent) => ({
                label: `${agent.agentType} (${formatAgentSourceLabel(agent.source)})${agent.overriddenBy ? ` · overridden by ${formatAgentSourceLabel(agent.overriddenBy)}` : ""
                    }`,
                action: () => onSelect(agent)
            }))
        ];
    }, [agents, onCreateNew, onSelect]);

    useInput((input, key) => {
        if (key.escape) {
            onBack();
            return;
        }
        if (key.upArrow) setIndex((value) => Math.max(0, value - 1));
        else if (key.downArrow) setIndex((value) => Math.min(entries.length - 1, value + 1));
        else if (key.return) {
            const entry = entries[index];
            if (entry) entry.action();
        }
    });

    return (
        <Box flexDirection="column" gap={1}>
            <Text bold>Agents ({formatAgentSourceLabel(source)})</Text>
            <Box flexDirection="column" marginTop={1}>
                {entries.map((entry, idx) => (
                    <Text key={entry.label} color={idx === index ? "suggestion" : undefined}>
                        {idx === index ? figures.pointer : " "} {entry.label}
                    </Text>
                ))}
            </Box>
            {changes.length > 0 && (
                <Box marginTop={1}>
                    <Text dimColor>{changes[changes.length - 1]}</Text>
                </Box>
            )}
        </Box>
    );
}

export function AgentsManager({ tools, onExit }: AgentsToolProps): React.ReactElement | null {
    const [viewState, setViewState] = useState<AgentViewState>({
        mode: "list-agents",
        source: "all"
    });
    const [appState, setAppState] = useAppState();
    const { allAgents, activeAgents } = appState.agentDefinitions;
    const [changes, setChanges] = useState<string[]>([]);
    const availableTools = normalizeAgentTools(
        tools,
        appState.mcp.tools,
        appState.toolPermissionContext
    );

    useCtrlExit();

    const sourceGroups = useMemo(() => getAgentSourceGroups(allAgents), [allAgents]);

    useInput((input, key) => {
        if (!key.escape) return;
        const changeSummary =
            changes.length > 0
                ? `Agent changes:\n${changes.join("\n")}`
                : undefined;
        switch (viewState.mode) {
            case "list-agents":
                onExit(changeSummary ?? "Agents dialog dismissed", {
                    display: changes.length === 0 ? "system" : undefined
                });
                break;
            case "create-agent":
            case "view-agent":
                return;
            default:
                if ("previousMode" in viewState) setViewState(viewState.previousMode);
        }
    });

    const handleAgentSaved = useCallback((message: string) => {
        setChanges((prev) => [...prev, message]);
        setViewState({ mode: "list-agents", source: "all" });
    }, []);

    const deleteAgent = useCallback(
        async (agent: AgentDefinition) => {
            try {
                await deleteAgentDefinition(agent);
                setAppState((state) => {
                    const remaining = state.agentDefinitions.allAgents.filter(
                        (entry) => !(entry.agentType === agent.agentType && entry.source === agent.source)
                    );
                    return {
                        ...state,
                        agentDefinitions: {
                            ...state.agentDefinitions,
                            allAgents: remaining,
                            activeAgents: rebuildActiveAgents(remaining)
                        }
                    };
                });
                setChanges((prev) => [...prev, `Deleted agent: ${formatEmphasisLabel(agent.agentType)}`]);
                setViewState({ mode: "list-agents", source: "all" });
            } catch (err) {
                logError(err instanceof Error ? err : Error("Failed to delete agent"));
            }
        },
        [setAppState]
    );

    switch (viewState.mode) {
        case "list-agents": {
            const source = viewState.source;
            const agents =
                source === "all"
                    ? [
                        ...sourceGroups["built-in"],
                        ...sourceGroups.userSettings,
                        ...sourceGroups.projectSettings,
                        ...sourceGroups.policySettings,
                        ...sourceGroups.flagSettings,
                        ...sourceGroups.plugin
                    ]
                    : sourceGroups[source] ?? [];

            const activeAgentByType = new Map<string, AgentDefinition>();
            activeAgents.forEach((agent) => activeAgentByType.set(agent.agentType, agent));

            const decoratedAgents = agents.map((agent) => {
                const active = activeAgentByType.get(agent.agentType);
                const overriddenBy = active && active.source !== agent.source ? active.source : undefined;
                return {
                    ...agent,
                    overriddenBy
                };
            });

            return (
                <>
                    <AgentListView
                        source={source}
                        agents={decoratedAgents}
                        onBack={() => {
                            const changeSummary =
                                changes.length > 0
                                    ? `Agent changes:\n${changes.join("\n")}`
                                    : undefined;
                            onExit(changeSummary ?? "Agents dialog dismissed", {
                                display: changes.length === 0 ? "system" : undefined
                            });
                        }}
                        onSelect={(agent) =>
                            setViewState({
                                mode: "agent-menu",
                                agent,
                                previousMode: viewState
                            })
                        }
                        onCreateNew={() => setViewState({ mode: "create-agent" })}
                        changes={changes}
                    />
                    <ExitHintFooter />
                </>
            );
        }
        case "create-agent":
            return (
                <CreateAgentWizard
                    tools={availableTools}
                    existingAgents={activeAgents}
                    onComplete={handleAgentSaved}
                    onCancel={() => setViewState({ mode: "list-agents", source: "all" })}
                />
            );
        case "agent-menu": {
            const agent =
                allAgents.find(
                    (entry) => entry.agentType === viewState.agent.agentType && entry.source === viewState.agent.source
                ) || viewState.agent;
            const isBuiltIn = agent.source === "built-in";
            const options = [
                { label: "View agent", value: "view" },
                ...(!isBuiltIn
                    ? [
                        { label: "Edit agent", value: "edit" },
                        { label: "Delete agent", value: "delete" }
                    ]
                    : []),
                { label: "Back", value: "back" }
            ];

            const handleSelect = (value: string) => {
                switch (value) {
                    case "view":
                        setViewState({
                            mode: "view-agent",
                            agent,
                            previousMode: viewState.previousMode
                        });
                        break;
                    case "edit":
                        setViewState({
                            mode: "edit-agent",
                            agent,
                            previousMode: viewState.previousMode
                        });
                        break;
                    case "delete":
                        setViewState({
                            mode: "delete-confirm",
                            agent,
                            previousMode: viewState.previousMode
                        });
                        break;
                    case "back":
                        setViewState(viewState.previousMode);
                        break;
                }
            };

            return (
                <>
                    <Panel title={viewState.agent.agentType}>
                        <Box flexDirection="column" marginTop={1}>
                            <PermissionSelect
                                options={options}
                                onChange={handleSelect}
                                onCancel={() => setViewState(viewState.previousMode)}
                            />
                            {changes.length > 0 && (
                                <Box marginTop={1}>
                                    <Text dimColor>{changes[changes.length - 1]}</Text>
                                </Box>
                            )}
                        </Box>
                    </Panel>
                    <ExitHintFooter />
                </>
            );
        }
        case "view-agent": {
            const agent =
                allAgents.find(
                    (entry) => entry.agentType === viewState.agent.agentType && entry.source === viewState.agent.source
                ) || viewState.agent;
            return (
                <>
                    <Panel title={agent.agentType}>
                        <AgentDetailsView
                            agent={agent}
                            tools={availableTools}
                            onBack={() =>
                                setViewState({
                                    mode: "agent-menu",
                                    agent,
                                    previousMode: viewState.previousMode
                                })
                            }
                        />
                    </Panel>
                    <ExitHintFooter instructions="Press Enter or Esc to go back" />
                </>
            );
        }
        case "delete-confirm": {
            const options = [
                { label: "Yes, delete", value: "yes" },
                { label: "No, cancel", value: "no" }
            ];
            return (
                <>
                    <Panel title="Delete agent" titleColor="error" borderColor="error">
                        <Text>
                            Are you sure you want to delete the agent{" "}
                            <Text bold>{viewState.agent.agentType}</Text>?
                        </Text>
                        <Box marginTop={1}>
                            <Text dimColor>Source: {viewState.agent.source}</Text>
                        </Box>
                        <Box marginTop={1}>
                            <PermissionSelect
                                options={options}
                                onChange={(value) => {
                                    if (value === "yes") deleteAgent(viewState.agent);
                                    else if ("previousMode" in viewState) setViewState(viewState.previousMode);
                                }}
                                onCancel={() => {
                                    if ("previousMode" in viewState) setViewState(viewState.previousMode);
                                }}
                            />
                        </Box>
                    </Panel>
                    <ExitHintFooter instructions="Press ↑↓ to navigate, Enter to select, Esc to cancel" />
                </>
            );
        }
        case "edit-agent": {
            const agent =
                allAgents.find(
                    (entry) => entry.agentType === viewState.agent.agentType && entry.source === viewState.agent.source
                ) || viewState.agent;
            return (
                <>
                    <Panel title={`Edit agent: ${agent.agentType}`}>
                        <AgentEditMenu
                            agent={agent}
                            tools={availableTools}
                            onSaved={(message) => {
                                handleAgentSaved(message);
                                setViewState(viewState.previousMode);
                            }}
                            onBack={() => setViewState(viewState.previousMode)}
                        />
                    </Panel>
                    <ExitHintFooter />
                </>
            );
        }
        default:
            return null;
    }
}

function Panel({
    title,
    titleColor = "text",
    borderColor = "background",
    children
}: {
    title: string;
    titleColor?: string;
    borderColor?: string;
    children: React.ReactNode;
}): React.ReactElement {
    return (
        <Box borderStyle="round" borderColor={borderColor} flexDirection="column" paddingX={1}>
            <Text bold color={titleColor}>
                {title}
            </Text>
            <Box flexDirection="column" marginTop={1}>
                {children}
            </Box>
        </Box>
    );
}

export const AgentsToolDefinition = {
    type: "local-jsx",
    name: "agents",
    description: "Manage agent configurations",
    isEnabled: () => true,
    isHidden: false,
    async call(onExit: (message?: string, meta?: any) => void, context: ToolUseContext) {
        const appState = (await context.getAppState?.()) ?? null;
        const permissionContext = appState?.toolPermissionContext;
        const toolDefinitions = normalizeAgentTools(
            context.options?.tools ?? [],
            appState?.mcp?.tools,
            permissionContext
        );
        return <AgentsManager tools={toolDefinitions} onExit={onExit} />;
    },
    userFacingName() {
        return "agents";
    }
};

export function parseMarketplaceSource(raw: string) {
    const trimmed = raw.trim();
    const fileSystem = getFileSystem();
    const sshMatch = trimmed.match(/^([a-zA-Z0-9._-]+@[^:]+:.+?(?:\.git)?)(#(.+))?$/);

    if (sshMatch?.[1]) {
        const url = sshMatch[1];
        const ref = sshMatch[3];
        return ref ? { source: "git", url, ref } : { source: "git", url };
    }

    if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
        const parts = trimmed.match(/^([^#]+)(#(.+))?$/);
        const url = parts?.[1] || trimmed;
        const ref = parts?.[3];

        if (url.endsWith(".git")) {
            return ref ? { source: "git", url, ref } : { source: "git", url };
        }

        let parsedUrl: URL | null = null;
        try {
            parsedUrl = new URL(url);
        } catch {
            return { source: "url", url };
        }

        if (parsedUrl.hostname === "github.com" || parsedUrl.hostname === "www.github.com") {
            if (parsedUrl.pathname.match(/^\/([^/]+\/[^/]+?)(\/|\.git|$)/)?.[1]) {
                const gitUrl = url.endsWith(".git") ? url : `${url}.git`;
                return ref ? { source: "git", url: gitUrl, ref } : { source: "git", url: gitUrl };
            }
        }

        return { source: "url", url };
    }

    if (
        trimmed.startsWith("./") ||
        trimmed.startsWith("../") ||
        trimmed.startsWith("/") ||
        trimmed.startsWith("~")
    ) {
        const resolved = resolvePath(trimmed.startsWith("~") ? trimmed.replace(/^~/, homedir()) : trimmed);
        if (!fileSystem.existsSync(resolved)) {
            return { error: `Path does not exist: ${resolved}` };
        }
        const stat = fileSystem.statSync(resolved);
        if (stat.isFile()) {
            if (resolved.endsWith(".json")) return { source: "file", path: resolved };
            return {
                error: `File path must point to a .json file (marketplace.json), but got: ${resolved}`
            };
        }
        if (stat.isDirectory()) return { source: "directory", path: resolved };
        return { error: `Path is neither a file nor a directory: ${resolved}` };
    }

    if (trimmed.includes("/") && !trimmed.startsWith("@")) {
        if (trimmed.includes(":")) return null;
        const parts = trimmed.match(/^([^#]+)(#(.+))?$/);
        const repo = parts?.[1] || trimmed;
        const ref = parts?.[3];
        return ref ? { source: "github", repo, ref } : { source: "github", repo };
    }

    return null;
}

async function addMarketplaceSource(source: any, onProgress: (message: string) => void) {
    onProgress("Updating marketplace registry...");
    if (source.source === "github") {
        return { name: source.repo };
    }
    if (source.source === "git") {
        return { name: source.url };
    }
    if (source.source === "url") {
        return { name: source.url };
    }
    if (source.source === "file" || source.source === "directory") {
        return { name: source.path };
    }
    return { name: "marketplace" };
}

function resetMarketplaceCache(): void { }

function trackMarketplaceAdded(_event: string, _payload: Record<string, any>): void { }

function logError(_error: Error): void { }

export function AddMarketplaceView({
    inputValue,
    setInputValue,
    cursorOffset,
    setCursorOffset,
    error,
    setError,
    result,
    setResult,
    setViewState,
    onAddComplete,
    cliMode = false
}: AddMarketplaceViewProps): React.ReactElement {
    const didAutoSubmit = useRef(false);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [progressMessage, setProgressMessage] = useState("");

    const handleSubmit = async () => {
        const trimmed = inputValue.trim();
        if (!trimmed) {
            setError("Please enter a marketplace source");
            return;
        }
        const parsed = parseMarketplaceSource(trimmed);
        if (!parsed) {
            setError("Invalid marketplace source format. Try: owner/repo, https://..., or ./path");
            return;
        }
        if ("error" in parsed) {
            setError(parsed.error ?? null);
            return;
        }

        setError(null);
        try {
            setIsSubmitting(true);
            setProgressMessage("");
            const { name } = await addMarketplaceSource(parsed, (message: string) => {
                setProgressMessage(message);
            });
            resetMarketplaceCache();
            const sourceType = parsed.source === "github" ? parsed.repo : parsed.source;
            trackMarketplaceAdded("tengu_marketplace_added", {
                source_type: sourceType
            });
            if (onAddComplete) await onAddComplete();
            setProgressMessage("");
            setIsSubmitting(false);
            if (cliMode) setResult(`Successfully added marketplace: ${name}`);
            else setViewState({ type: "browse-marketplace", targetMarketplace: name });
        } catch (err) {
            const errorObj = err instanceof Error ? err : new Error(String(err));
            logError(errorObj);
            setError(errorObj.message);
            setProgressMessage("");
            setIsSubmitting(false);
            if (cliMode) setResult(`Error: ${errorObj.message}`);
            else setResult(null);
        }
    };

    useEffect(() => {
        if (inputValue && !didAutoSubmit.current && !error && !result) {
            didAutoSubmit.current = true;
            void handleSubmit();
        }
    }, []);

    return (
        <Box flexDirection="column">
            <Box flexDirection="column" paddingX={1} borderStyle="round">
                <Box marginBottom={1}>
                    <Text bold>Add Marketplace</Text>
                </Box>
                <Box flexDirection="column">
                    <Text>Enter marketplace source:</Text>
                    <Text dimColor>Examples:</Text>
                    <Text dimColor> • owner/repo (GitHub)</Text>
                    <Text dimColor> • git@github.com:owner/repo.git (SSH)</Text>
                    <Text dimColor> • https://example.com/marketplace.json</Text>
                    <Text dimColor> • ./path/to/marketplace</Text>
                    <Box marginTop={1}>
                        <TextInput
                            value={inputValue}
                            onChange={setInputValue}
                            onSubmit={() => void handleSubmit()}
                            columns={80}
                            cursorOffset={cursorOffset}
                            onChangeCursorOffset={setCursorOffset}
                            focus={true}
                            showCursor={true}
                        />
                    </Box>
                </Box>
                {isSubmitting && (
                    <Box marginTop={1}>
                        <InlineSpinner />
                        <Text>{progressMessage || "Adding marketplace to configuration..."}</Text>
                    </Box>
                )}
                {error && (
                    <Box marginTop={1}>
                        <Text color="error">{error}</Text>
                    </Box>
                )}
                {result && (
                    <Box marginTop={1}>
                        <Text>{result}</Text>
                    </Box>
                )}
            </Box>
            <Box marginLeft={3}>
                <Text dimColor italic>Enter to add · Esc to cancel</Text>
            </Box>
        </Box>
    );
}

async function deleteAgentDefinition(agent: AgentDefinition): Promise<void> {
    void agent;
}
