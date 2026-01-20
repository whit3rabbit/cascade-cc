// Logic from chunk_565.ts (Agent Creation Wizard - Final Steps & Orchestration)

import React, { useMemo, useState } from "react";
import { Box, Text, useInput } from "ink";
import { figures } from "../../vendor/terminalFigures.js";
import { Wizard, WizardLayout, useWizard, AgentLocationStep, AgentMethodStep } from "./Wizard.js";
import { ShortcutHint, ShortcutGroup } from "../shared/Shortcut.js";
import {
    AgentDescriptionStep,
    AgentGenerationStep,
    AgentIdentifierStep,
    AgentSystemPromptStep,
    validateAgentConfig,
    groupToolsByServer
} from "./AgentWizardSteps.js";
import { formatModelName } from "../../services/claude/modelSettings.js";
import { useAppState } from "../../contexts/AppStateContext.js";

const TOOL_SELECTION_SEPARATOR = "-".repeat(40);

const TOOL_COLORS: Record<string, string> = {
    blue: "blue",
    green: "green",
    magenta: "magenta",
    cyan: "cyan",
    yellow: "yellow",
    red: "red"
};

const TOOL_COLOR_OPTIONS = ["automatic", ...Object.keys(TOOL_COLORS)];

type ToolDefinition = {
    name: string;
};

type ToolBucket = {
    id: string;
    name: string;
    tools: ToolDefinition[];
};

type ToolCategoryDefinition = {
    READ_ONLY: { name: string; toolNames: Set<string> };
    EDIT: { name: string; toolNames: Set<string> };
    EXECUTION: { name: string; toolNames: Set<string> };
    MCP: { name: string; toolNames: Set<string>; isMcp?: boolean };
    OTHER: { name: string; toolNames: Set<string> };
};

type WizardAgent = {
    agentType: string;
    whenToUse: string;
    getSystemPrompt: () => string;
    tools?: string[];
    model?: string;
    color?: string;
    source: string;
};

type AgentWizardProps = {
    tools: ToolDefinition[];
    existingAgents: WizardAgent[];
    onComplete: (message: string) => void;
    onCancel: () => void;
};

function getToolCategoryDefinitions(): ToolCategoryDefinition {
    return {
        READ_ONLY: {
            name: "Read-only tools",
            toolNames: new Set()
        },
        EDIT: {
            name: "Edit tools",
            toolNames: new Set()
        },
        EXECUTION: {
            name: "Execution tools",
            toolNames: new Set()
        },
        MCP: {
            name: "MCP tools",
            toolNames: new Set(),
            isMcp: true
        },
        OTHER: {
            name: "Other tools",
            toolNames: new Set()
        }
    };
}

function isMcpTool(tool: ToolDefinition): boolean {
    return tool.name.startsWith("mcp__");
}

function parseMcpToolName(toolName: string): { serverName: string; toolName: string } | null {
    const match = toolName.match(/^mcp__([^_]+)__(.+)$/);
    if (!match) return null;
    return { serverName: match[1], toolName: match[2] };
}

function buildToolBuckets(tools: ToolDefinition[]): ToolBucket[] {
    const categories = getToolCategoryDefinitions();
    const buckets = {
        readOnly: [] as ToolDefinition[],
        edit: [] as ToolDefinition[],
        execution: [] as ToolDefinition[],
        mcp: [] as ToolDefinition[],
        other: [] as ToolDefinition[]
    };

    tools.forEach((tool) => {
        if (isMcpTool(tool)) {
            buckets.mcp.push(tool);
            return;
        }
        if (categories.READ_ONLY.toolNames.has(tool.name)) {
            buckets.readOnly.push(tool);
            return;
        }
        if (categories.EDIT.toolNames.has(tool.name)) {
            buckets.edit.push(tool);
            return;
        }
        if (categories.EXECUTION.toolNames.has(tool.name)) {
            buckets.execution.push(tool);
            return;
        }
        if (tool.name !== "agent") buckets.other.push(tool);
    });

    return [
        { id: "bucket-readonly", name: categories.READ_ONLY.name, tools: buckets.readOnly },
        { id: "bucket-edit", name: categories.EDIT.name, tools: buckets.edit },
        { id: "bucket-execution", name: categories.EXECUTION.name, tools: buckets.execution },
        { id: "bucket-mcp", name: categories.MCP.name, tools: buckets.mcp },
        { id: "bucket-other", name: categories.OTHER.name, tools: buckets.other }
    ];
}

type ToolMenuEntry = {
    id: string;
    label: string;
    action: () => void;
    isContinue?: boolean;
    isToggle?: boolean;
    isHeader?: boolean;
};

export function AgentToolsEditor({
    tools,
    initialTools,
    onComplete,
    onCancel
}: {
    tools: ToolDefinition[];
    initialTools?: string[];
    onComplete: (tools?: string[]) => void;
    onCancel?: () => void;
}) {
    const availableTools = useMemo(() => tools, [tools]);
    const initial = !initialTools || initialTools.includes("*") ? availableTools.map((tool) => tool.name) : initialTools;
    const [selected, setSelected] = useState<string[]>(initial);
    const [cursor, setCursor] = useState(0);
    const [showAdvanced, setShowAdvanced] = useState(false);

    const activeToolNames = useMemo(() => {
        const allowed = new Set(availableTools.map((tool) => tool.name));
        return selected.filter((name) => allowed.has(name));
    }, [selected, availableTools]);

    const selectedSet = new Set(activeToolNames);
    const allSelected = activeToolNames.length === availableTools.length && availableTools.length > 0;
    const buckets = useMemo(() => buildToolBuckets(availableTools), [availableTools]);

    const toggleSingle = (toolName?: string) => {
        if (!toolName) return;
        setSelected((prev) => (prev.includes(toolName) ? prev.filter((name) => name !== toolName) : [...prev, toolName]));
    };

    const toggleMultiple = (toolNames: string[], enable: boolean) => {
        setSelected((prev) => {
            if (enable) {
                const additions = toolNames.filter((name) => !prev.includes(name));
                return [...prev, ...additions];
            }
            return prev.filter((name) => !toolNames.includes(name));
        });
    };

    const handleContinue = () => {
        const allToolNames = availableTools.map((tool) => tool.name);
        const resolved =
            activeToolNames.length === allToolNames.length && allToolNames.every((name) => activeToolNames.includes(name))
                ? undefined
                : activeToolNames;
        onComplete(resolved);
    };

    const menuEntries: ToolMenuEntry[] = [];
    menuEntries.push({ id: "continue", label: "Continue", action: handleContinue, isContinue: true });
    menuEntries.push({
        id: "bucket-all",
        label: `${allSelected ? figures.checkboxOn : figures.checkboxOff} All tools`,
        action: () => {
            const toolNames = availableTools.map((tool) => tool.name);
            toggleMultiple(toolNames, !allSelected);
        }
    });

    buckets.forEach((bucket) => {
        if (bucket.tools.length === 0) return;
        const isBucketSelected = bucket.tools.filter((tool) => selectedSet.has(tool.name)).length === bucket.tools.length;
        menuEntries.push({
            id: bucket.id,
            label: `${isBucketSelected ? figures.checkboxOn : figures.checkboxOff} ${bucket.name}`,
            action: () => {
                const names = bucket.tools.map((tool) => tool.name);
                toggleMultiple(names, !isBucketSelected);
            }
        });
    });

    const advancedStartIndex = menuEntries.length;
    menuEntries.push({
        id: "toggle-individual",
        label: showAdvanced ? "Hide advanced options" : "Show advanced options",
        action: () => {
            setShowAdvanced((prev) => !prev);
            if (showAdvanced && cursor > advancedStartIndex) setCursor(advancedStartIndex);
        },
        isToggle: true
    });

    const mcpGroups = useMemo(() => groupToolsByServer(availableTools), [availableTools]);

    if (showAdvanced) {
        if (mcpGroups.length > 0) {
            menuEntries.push({ id: "mcp-servers-header", label: "MCP Servers:", action: () => { }, isHeader: true });
            mcpGroups.forEach((group) => {
                const isGroupSelected =
                    group.tools.filter((tool) => selectedSet.has(tool.name)).length === group.tools.length;
                menuEntries.push({
                    id: `mcp-server-${group.serverName}`,
                    label: `${isGroupSelected ? figures.checkboxOn : figures.checkboxOff} ${group.serverName} (${group.tools.length} tool${group.tools.length === 1 ? "" : "s"
                        })`,
                    action: () => {
                        const names = group.tools.map((tool) => tool.name);
                        toggleMultiple(names, !isGroupSelected);
                    }
                });
            });
            menuEntries.push({ id: "tools-header", label: "Individual Tools:", action: () => { }, isHeader: true });
        }
        availableTools.forEach((tool) => {
            let displayName = tool.name;
            if (tool.name.startsWith("mcp__")) {
                const parsed = parseMcpToolName(tool.name);
                displayName = parsed ? `${parsed.toolName} (${parsed.serverName})` : tool.name;
            }
            menuEntries.push({
                id: `tool-${tool.name}`,
                label: `${selectedSet.has(tool.name) ? figures.checkboxOn : figures.checkboxOff} ${displayName}`,
                action: () => toggleSingle(tool.name)
            });
        });
    }

    useInput((_input, key) => {
        if (key.return) {
            const entry = menuEntries[cursor];
            if (entry && !entry.isHeader) entry.action();
        } else if (key.escape) {
            if (onCancel) onCancel();
            else onComplete(initialTools);
        } else if (key.upArrow) {
            let next = cursor - 1;
            while (next > 0 && menuEntries[next]?.isHeader) next -= 1;
            setCursor(Math.max(0, next));
        } else if (key.downArrow) {
            let next = cursor + 1;
            while (next < menuEntries.length - 1 && menuEntries[next]?.isHeader) next += 1;
            setCursor(Math.min(menuEntries.length - 1, next));
        }
    });

    return (
        <Box flexDirection="column" marginTop={1}>
            <Text color={cursor === 0 ? "suggestion" : undefined} bold={cursor === 0}>
                {cursor === 0 ? `${figures.pointer} ` : "  "}[ Continue ]
            </Text>
            <Text dimColor>{TOOL_SELECTION_SEPARATOR}</Text>
            {menuEntries.slice(1).map((entry, index) => {
                const realIndex = index + 1;
                const isSelected = realIndex === cursor;
                return (
                    <React.Fragment key={entry.id}>
                        {entry.isToggle && <Text dimColor>{TOOL_SELECTION_SEPARATOR}</Text>}
                        {entry.isHeader && index > 0 && <Box marginTop={1} />}
                        <Text color={entry.isHeader ? undefined : isSelected ? "suggestion" : undefined} dimColor={entry.isHeader} bold={entry.isToggle && isSelected}>
                            {entry.isHeader ? "" : isSelected ? `${figures.pointer} ` : "  "}
                            {entry.isToggle ? `[ ${entry.label} ]` : entry.label}
                        </Text>
                    </React.Fragment>
                );
            })}
            <Box marginTop={1} flexDirection="column">
                <Text dimColor>
                    {allSelected
                        ? "All tools selected"
                        : `${selectedSet.size} of ${availableTools.length} tools selected`}
                </Text>
            </Box>
        </Box>
    );
}

export function AgentToolsStep({ tools }: { tools: ToolDefinition[] }) {
    const { goNext, goBack, updateWizardData, wizardData } = useWizard();

    const handleComplete = (selection?: string[]) => {
        updateWizardData({ selectedTools: selection });
        goNext();
    };

    return (
        <WizardLayout
            subtitle="Select tools"
            footerText={
                <ShortcutGroup>
                    <ShortcutHint shortcut="Enter" action="toggle selection" />
                    <ShortcutHint shortcut="↑↓" action="navigate" />
                    <ShortcutHint shortcut="Esc" action="go back" />
                </ShortcutGroup>
            }
        >
            <AgentToolsEditor
                tools={tools}
                initialTools={wizardData.selectedTools}
                onComplete={handleComplete}
                onCancel={goBack}
            />
        </WizardLayout>
    );
}

function getModelOptions(): { label: string; value: string }[] {
    return [
        { label: "Sonnet", value: "sonnet" },
        { label: "Opus", value: "opus" },
        { label: "Haiku", value: "haiku" },
        { label: "Inherit", value: "inherit" }
    ];
}

export function AgentModelPicker({
    initialModel,
    onComplete,
    onCancel
}: {
    initialModel?: string;
    onComplete: (model: string) => void;
    onCancel: () => void;
}) {
    const options = useMemo(() => getModelOptions(), []);
    const defaultValue = useMemo(() => {
        if (initialModel && options.some((option) => option.value === initialModel)) return initialModel;
        return "sonnet";
    }, [initialModel, options]);
    const [index, setIndex] = useState(
        Math.max(
            0,
            options.findIndex((option) => option.value === defaultValue)
        )
    );

    useInput((_input, key) => {
        if (key.upArrow) setIndex((value) => Math.max(0, value - 1));
        if (key.downArrow) setIndex((value) => Math.min(options.length - 1, value + 1));
        if (key.return) onComplete(options[index].value);
        if (key.escape) onCancel();
    });

    return (
        <Box flexDirection="column">
            <Box marginBottom={1}>
                <Text dimColor>Model determines the agent's reasoning capabilities and speed.</Text>
            </Box>
            <Box flexDirection="column">
                {options.map((option, idx) => (
                    <Text key={option.value} color={idx === index ? "suggestion" : undefined}>
                        {idx === index ? `${figures.pointer} ` : "  "}{option.label}
                    </Text>
                ))}
            </Box>
        </Box>
    );
}

export function AgentModelStep() {
    const { goNext, goBack, updateWizardData, wizardData } = useWizard();
    const handleComplete = (model: string) => {
        updateWizardData({ selectedModel: model });
        goNext();
    };

    return (
        <WizardLayout
            subtitle="Select model"
            footerText={
                <ShortcutGroup>
                    <ShortcutHint shortcut="↑↓" action="navigate" />
                    <ShortcutHint shortcut="Enter" action="select" />
                    <ShortcutHint shortcut="Esc" action="go back" />
                </ShortcutGroup>
            }
        >
            <AgentModelPicker initialModel={wizardData.selectedModel} onComplete={handleComplete} onCancel={goBack} />
        </WizardLayout>
    );
}

export function AgentColorPicker({
    agentName,
    currentColor = "automatic",
    onConfirm
}: {
    agentName: string;
    currentColor?: string;
    onConfirm: (color?: string) => void;
}) {
    const [index, setIndex] = useState(Math.max(0, TOOL_COLOR_OPTIONS.indexOf(currentColor)));

    useInput((_input, key) => {
        if (key.upArrow) setIndex((value) => (value > 0 ? value - 1 : TOOL_COLOR_OPTIONS.length - 1));
        else if (key.downArrow) setIndex((value) => (value < TOOL_COLOR_OPTIONS.length - 1 ? value + 1 : 0));
        else if (key.return) {
            const selectedColor = TOOL_COLOR_OPTIONS[index];
            onConfirm(selectedColor === "automatic" ? undefined : selectedColor);
        }
    });

    const selectedColor = TOOL_COLOR_OPTIONS[index];

    return (
        <Box flexDirection="column" gap={1}>
            <Box flexDirection="column">
                {TOOL_COLOR_OPTIONS.map((color, idx) => {
                    const isActive = idx === index;
                    return (
                        <Box key={color} flexDirection="row" gap={1}>
                            <Text color={isActive ? "suggestion" : undefined}>{isActive ? figures.pointer : " "}</Text>
                            {color === "automatic" ? (
                                <Text bold={isActive}>Automatic color</Text>
                            ) : (
                                <Box gap={1}>
                                    <Text backgroundColor={TOOL_COLORS[color]} color="inverseText">
                                        {" "}
                                    </Text>
                                    <Text bold={isActive}>{color.charAt(0).toUpperCase() + color.slice(1)}</Text>
                                </Box>
                            )}
                        </Box>
                    );
                })}
            </Box>
            <Box marginTop={1}>
                <Text>Preview: </Text>
                {selectedColor === "automatic" ? (
                    <Text inverse bold>
                        {" "}{agentName}{" "}
                    </Text>
                ) : (
                    <Text backgroundColor={TOOL_COLORS[selectedColor]} color="inverseText" bold>
                        {" "}{agentName}{" "}
                    </Text>
                )}
            </Box>
        </Box>
    );
}

export function AgentColorStep() {
    const { goNext, goBack, updateWizardData, wizardData } = useWizard();

    useInput((_input, key) => {
        if (key.escape) goBack();
    });

    const handleConfirm = (color?: string) => {
        updateWizardData({
            selectedColor: color,
            finalAgent: {
                agentType: wizardData.agentType,
                whenToUse: wizardData.whenToUse,
                getSystemPrompt: () => wizardData.systemPrompt,
                tools: wizardData.selectedTools,
                ...(wizardData.selectedModel ? { model: wizardData.selectedModel } : {}),
                ...(color ? { color } : {}),
                source: wizardData.location
            }
        });
        goNext();
    };

    return (
        <WizardLayout
            subtitle="Choose background color"
            footerText={
                <ShortcutGroup>
                    <ShortcutHint shortcut="↑↓" action="navigate" />
                    <ShortcutHint shortcut="Enter" action="select" />
                    <ShortcutHint shortcut="Esc" action="go back" />
                </ShortcutGroup>
            }
        >
            <Box marginTop={1}>
                <AgentColorPicker agentName={wizardData.agentType || "agent"} currentColor="automatic" onConfirm={handleConfirm} />
            </Box>
        </WizardLayout>
    );
}

function formatAgentPath({ source, agentType }: { source: string; agentType: string }): string {
    if (source === "projectSettings") return `.claude/agents/${agentType}.md`;
    if (source === "userSettings") return `~/.claude/agents/${agentType}.md`;
    return `${agentType}.md`;
}

function formatToolList(tools?: string[]): string {
    if (tools === undefined) return "All tools";
    if (tools.length === 0) return "None";
    if (tools.length === 1) return tools[0] || "None";
    if (tools.length === 2) return tools.join(" and ");
    return `${tools.slice(0, -1).join(", ")}, and ${tools[tools.length - 1]}`;
}

export function AgentConfirmView({
    tools,
    existingAgents,
    onSave,
    onSaveAndEdit,
    error
}: {
    tools: ToolDefinition[];
    existingAgents: WizardAgent[];
    onSave: () => void;
    onSaveAndEdit: () => void;
    error: string | null;
}) {
    const { goBack, wizardData } = useWizard();

    useInput((input, key) => {
        if (key.escape) goBack();
        else if (input === "s" || key.return) onSave();
        else if (input === "e") onSaveAndEdit();
    });

    const agent = wizardData.finalAgent as WizardAgent;
    const validation = validateAgentConfig(agent, tools, existingAgents);

    return (
        <WizardLayout
            subtitle="Confirm and save"
            footerText={
                <ShortcutGroup>
                    <ShortcutHint shortcut="s/Enter" action="save" />
                    <ShortcutHint shortcut="e" action="edit in your editor" />
                    <ShortcutHint shortcut="Esc" action="cancel" />
                </ShortcutGroup>
            }
        >
            <Box flexDirection="column" marginTop={1}>
                <Text>
                    <Text bold>Name</Text>: {agent.agentType}
                </Text>
                <Text>
                    <Text bold>Location</Text>: {formatAgentPath({ source: wizardData.location, agentType: agent.agentType })}
                </Text>
                <Text>
                    <Text bold>Tools</Text>: {formatToolList(agent.tools)}
                </Text>
                <Text>
                    <Text bold>Model</Text>: {formatModelName(agent.model ?? "sonnet")}
                </Text>
                <Box marginTop={1}>
                    <Text>
                        <Text bold>Description</Text> (tells Claude when to use this agent):
                    </Text>
                </Box>
                <Box marginLeft={2} marginTop={1}>
                    <Text>{agent.whenToUse.length > 240 ? `${agent.whenToUse.slice(0, 240)}...` : agent.whenToUse}</Text>
                </Box>
                <Box marginTop={1}>
                    <Text>
                        <Text bold>System prompt</Text>:
                    </Text>
                </Box>
                <Box marginLeft={2} marginTop={1}>
                    <Text>
                        {(() => {
                            const prompt = agent.getSystemPrompt();
                            return prompt.length > 240 ? `${prompt.slice(0, 240)}...` : prompt;
                        })()}
                    </Text>
                </Box>
                {validation.warnings.length > 0 && (
                    <Box marginTop={1} flexDirection="column">
                        <Text color="warning">Warnings:</Text>
                        {validation.warnings.map((warning, index) => (
                            <Text key={index} dimColor>
                                {" "}• {warning}
                            </Text>
                        ))}
                    </Box>
                )}
                {validation.errors.length > 0 && (
                    <Box marginTop={1} flexDirection="column">
                        <Text color="error">Errors:</Text>
                        {validation.errors.map((err, index) => (
                            <Text key={index} color="error">
                                {" "}• {err}
                            </Text>
                        ))}
                    </Box>
                )}
                {error && (
                    <Box marginTop={1}>
                        <Text color="error">{error}</Text>
                    </Box>
                )}
                <Box marginTop={2}>
                    <Text color="success">
                        Press <Text bold>s</Text> or <Text bold>Enter</Text> to save, <Text bold>e</Text> to save and edit
                    </Text>
                </Box>
            </Box>
        </WizardLayout>
    );
}

async function saveAgentDefinition(
    location: string,
    agentType: string,
    whenToUse: string,
    tools: string[] | undefined,
    systemPrompt: string,
    includeFrontmatter: boolean,
    color?: string,
    model?: string
): Promise<void> {
    void location;
    void agentType;
    void whenToUse;
    void tools;
    void systemPrompt;
    void includeFrontmatter;
    void color;
    void model;
}

function rebuildActiveAgents(agents: WizardAgent[]): WizardAgent[] {
    return agents;
}

function openExternalEditor(path: string) {
    const editor = (globalThis as any).openExternalEditor;
    if (typeof editor === "function") editor(path);
}

function getAgentFilePath(agent: WizardAgent): string {
    return formatAgentPath({ source: agent.source, agentType: agent.agentType });
}

export function AgentConfirmStep({
    tools,
    existingAgents,
    onComplete
}: {
    tools: ToolDefinition[];
    existingAgents: WizardAgent[];
    onComplete: (message: string) => void;
}) {
    const { wizardData } = useWizard();
    const [error, setError] = useState<string | null>(null);
    const [, setAppState] = useAppState();

    const handleSave = async () => {
        if (!wizardData?.finalAgent) return;
        try {
            await saveAgentDefinition(
                wizardData.location,
                wizardData.finalAgent.agentType,
                wizardData.finalAgent.whenToUse,
                wizardData.finalAgent.tools,
                wizardData.finalAgent.getSystemPrompt(),
                true,
                wizardData.finalAgent.color,
                wizardData.finalAgent.model
            );
            setAppState((state: any) => {
                if (!wizardData.finalAgent) return state;
                const allAgents = state.agentDefinitions.allAgents.concat(wizardData.finalAgent);
                return {
                    ...state,
                    agentDefinitions: {
                        ...state.agentDefinitions,
                        activeAgents: rebuildActiveAgents(allAgents),
                        allAgents
                    }
                };
            });
            onComplete(`Created agent: ${wizardData.finalAgent.agentType}`);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to save agent");
        }
    };

    const handleSaveAndEdit = async () => {
        if (!wizardData?.finalAgent) return;
        try {
            await saveAgentDefinition(
                wizardData.location,
                wizardData.finalAgent.agentType,
                wizardData.finalAgent.whenToUse,
                wizardData.finalAgent.tools,
                wizardData.finalAgent.getSystemPrompt(),
                true,
                wizardData.finalAgent.color,
                wizardData.finalAgent.model
            );
            setAppState((state: any) => {
                if (!wizardData.finalAgent) return state;
                const allAgents = state.agentDefinitions.allAgents.concat(wizardData.finalAgent);
                return {
                    ...state,
                    agentDefinitions: {
                        ...state.agentDefinitions,
                        activeAgents: rebuildActiveAgents(allAgents),
                        allAgents
                    }
                };
            });
            const filePath = getAgentFilePath(wizardData.finalAgent);
            openExternalEditor(filePath);
            onComplete(
                `Created agent: ${wizardData.finalAgent.agentType} and opened in editor. If you made edits, restart to load the latest version.`
            );
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to save agent");
        }
    };

    return (
        <AgentConfirmView
            tools={tools}
            existingAgents={existingAgents}
            onSave={handleSave}
            onSaveAndEdit={handleSaveAndEdit}
            error={error}
        />
    );
}

export function CreateAgentWizard({ tools, existingAgents, onComplete, onCancel }: AgentWizardProps) {
    return (
        <Wizard
            steps={[
                AgentLocationStep,
                AgentMethodStep,
                AgentGenerationStep,
                () => <AgentIdentifierStep existingAgents={existingAgents} />,
                AgentSystemPromptStep,
                AgentDescriptionStep,
                () => <AgentToolsStep tools={tools} />,
                AgentModelStep,
                AgentColorStep,
                () => <AgentConfirmStep tools={tools} existingAgents={existingAgents} onComplete={onComplete} />
            ]}
            initialData={{}}
            onComplete={() => { }}
            onCancel={onCancel}
            title="Create new agent"
            showStepCounter={false}
        />
    );
}
