// Logic from chunk_564.ts (Agent Creation Wizard - Steps)

import React, { useMemo, useRef, useState } from "react";
import { Box, Text, useInput } from "ink";
import { figures } from "../../vendor/terminalFigures.js";
import { Shortcut } from "../shared/Shortcut.js";
import { WizardLayout, useWizard } from "./Wizard.js";
import InkTextInput from "ink-text-input";
import { useAppState } from "../../contexts/AppStateContext.js";
import { normalizeModelId } from "../../services/claude/modelSettings.js";
import { getDefaultModelName } from "../../services/claude/claudeUtils.js";

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
};

type AgentDraft = {
    agentType: string;
    whenToUse: string;
    tools?: string[];
    source: string;
    getSystemPrompt: () => string;
};

type GeneratedAgent = {
    identifier: string;
    whenToUse: string;
    systemPrompt: string;
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

function formatAgentSourceLabel(source: string): string {
    return AGENT_SOURCE_LABELS[source] ?? source;
}

export const AGENT_GENERATION_SYSTEM_PROMPT = `You are an elite AI agent architect specializing in crafting high-performance agent configurations. Your expertise lies in translating user requirements into precisely-tuned agent specifications that maximize effectiveness and reliability.

**Important Context**: You may have access to project-specific instructions from CLAUDE.md files and other context that may include coding standards, project structure, and custom requirements. Consider this context when creating agents to ensure they align with the project's established patterns and practices.

When a user describes what they want an agent to do, you will:

1. **Extract Core Intent**: Identify the fundamental purpose, key responsibilities, and success criteria for the agent. Look for both explicit requirements and implicit needs. Consider any project-specific context from CLAUDE.md files. For agents that are meant to review code, you should assume that the user is asking to review recently written code and not the whole codebase, unless the user has explicitly instructed you otherwise.

2. **Design Expert Persona**: Create a compelling expert identity that embodies deep domain knowledge relevant to the task. The persona should inspire confidence and guide the agent's decision-making approach.

3. **Architect Comprehensive Instructions**: Develop a system prompt that:
   - Establishes clear behavioral boundaries and operational parameters
   - Provides specific methodologies and best practices for task execution
   - Anticipates edge cases and provides guidance for handling them
   - Incorporates any specific requirements or preferences mentioned by the user
   - Defines output format expectations when relevant
   - Aligns with project-specific coding standards and patterns from CLAUDE.md

4. **Optimize for Performance**: Include:
   - Decision-making frameworks appropriate to the domain
   - Quality control mechanisms and self-verification steps
   - Efficient workflow patterns
   - Clear escalation or fallback strategies

5. **Create Identifier**: Design a concise, descriptive identifier that:
   - Uses lowercase letters, numbers, and hyphens only
   - Is typically 2-4 words joined by hyphens
   - Clearly indicates the agent's primary function
   - Is memorable and easy to type
   - Avoids generic terms like "helper" or "assistant"

6 **Example agent descriptions**:
  - in the 'whenToUse' field of the JSON object, you should include examples of when this agent should be used.
  - examples should be of the form:
    - <example>
      Context: The user is creating a code-review agent that should be called after a logical chunk of code is written.
      user: "Please write a function that checks if a number is prime"
      assistant: "Here is the relevant function: "
      <function call omitted for brevity only for this example>
      <commentary>
      Since the user is greeting, use the \${n3} tool to launch the greeting-responder agent to respond with a friendly joke.
      </commentary>
      assistant: "Now let me use the code-reviewer agent to review the code"
    </example>
    - <example>
      Context: User is creating an agent to respond to the word "hello" with a friendly jok.
      user: "Hello"
      assistant: "I'm going to use the \${n3} tool to launch the greeting-responder agent to respond with a friendly joke"
      <commentary>
      Since the user is greeting, use the greeting-responder agent to respond with a friendly joke.
      </commentary>
    </example>
  - If the user mentioned or implied that the agent should be used proactively, you should include examples of this.
- NOTE: Ensure that in the examples, you are making the assistant use the Agent tool and not simply respond directly to the task.

Your output must be a valid JSON object with exactly these fields:
{
  "identifier": "A unique, descriptive identifier using lowercase letters, numbers, and hyphens (e.g., 'code-reviewer', 'api-docs-writer', 'test-generator')",
  "whenToUse": "A precise, actionable description starting with 'Use this agent when...' that clearly defines the triggering conditions and use cases. Ensure you include examples as described above.",
  "systemPrompt": "The complete system prompt that will govern the agent's behavior, written in second person ('You are...', 'You will...') and structured for maximum clarity and effectiveness"
}

Key principles for your system prompts:
- Be specific rather than generic - avoid vague instructions
- Include concrete examples when they would clarify behavior
- Balance comprehensiveness with clarity - every instruction should add value
- Ensure the agent has enough context to handle variations of the core task
- Make the agent proactive in seeking clarification when needed
- Build in quality assurance and self-correction mechanisms

Remember: The agents you create should be autonomous experts capable of handling their designated tasks with minimal additional guidance. Your system prompts are their complete operational manual.
`;

function createAbortController(): AbortController {
    return new AbortController();
}

function useCurrentModelId(): string {
    const [appState] = useAppState();
    const { mainLoopModel, mainLoopModelForSession } = appState;

    return useMemo(() => {
        const fallback = getDefaultModelName();
        return normalizeModelId(mainLoopModelForSession ?? mainLoopModel ?? fallback);
    }, [mainLoopModel, mainLoopModelForSession]);
}

async function generateAgentDefinition(
    prompt: string,
    modelId: string,
    existingIdentifiers: string[],
    signal?: AbortSignal
): Promise<GeneratedAgent> {
    void modelId;
    void existingIdentifiers;
    if (signal?.aborted) throw new Error("Generation cancelled");

    return {
        identifier: "generated-agent",
        whenToUse: `Use this agent when ${prompt}.`,
        systemPrompt: AGENT_GENERATION_SYSTEM_PROMPT
    };
}

// --- Agent Type Validation (qO0) ---
export function validateAgentType(name: string): string | null {
    if (!name) return "Agent type is required";
    if (!/^[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]$/.test(name)) {
        return "Agent type must start and end with alphanumeric characters and contain only letters, numbers, and hyphens";
    }
    if (name.length < 3) return "Agent type must be at least 3 characters long";
    if (name.length > 50) return "Agent type must be less than 50 characters";
    return null;
}

function getAgentToolStatus(agent: AgentDraft, tools: ToolDefinition[]) {
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

// --- Agent Config Validation (mJ9) ---
export function validateAgentConfig(agent: AgentDraft, tools: ToolDefinition[], existingAgents: AgentDraft[]) {
    const errors: string[] = [];
    const warnings: string[] = [];

    if (!agent.agentType) {
        errors.push("Agent type is required");
    } else {
        const validation = validateAgentType(agent.agentType);
        if (validation) errors.push(validation);
        const duplicate = existingAgents.find(
            (entry) => entry.agentType === agent.agentType && entry.source !== agent.source
        );
        if (duplicate) {
            errors.push(
                `Agent type "${agent.agentType}" already exists in ${formatAgentSourceLabel(duplicate.source)}`
            );
        }
    }

    if (!agent.whenToUse) {
        errors.push("Description (description) is required");
    } else if (agent.whenToUse.length < 10) {
        warnings.push("Description should be more descriptive (at least 10 characters)");
    } else if (agent.whenToUse.length > 5000) {
        warnings.push("Description is very long (over 5000 characters)");
    }

    if (agent.tools !== undefined && !Array.isArray(agent.tools)) {
        errors.push("Tools must be an array");
    } else {
        if (agent.tools === undefined) warnings.push("Agent has access to all tools");
        else if (agent.tools.length === 0)
            warnings.push("No tools selected - agent will have very limited capabilities");
        const toolStatus = getAgentToolStatus(agent, tools);
        if (toolStatus.invalidTools.length > 0) {
            errors.push(`Invalid tools: ${toolStatus.invalidTools.join(", ")}`);
        }
    }

    const prompt = agent.getSystemPrompt();
    if (!prompt) {
        errors.push("System prompt is required");
    } else if (prompt.length < 20) {
        errors.push("System prompt is too short (minimum 20 characters)");
    } else if (prompt.length > 10000) {
        warnings.push("System prompt is very long (over 10,000 characters)");
    }

    return {
        isValid: errors.length === 0,
        errors,
        warnings
    };
}

// --- Agent Generation Step (gJ9) ---
export function AgentGenerationStep() {
    const { updateWizardData, goBack, goToStep, wizardData } = useWizard();
    const [value, setValue] = useState(wizardData.generationPrompt || "");
    const [isGenerating, setIsGenerating] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [cursorOffset, setCursorOffset] = useState(value.length);
    const modelId = useCurrentModelId();
    const abortRef = useRef<AbortController | null>(null);

    useInput((_input, key) => {
        if (!key.escape) return;
        if (isGenerating && abortRef.current) {
            abortRef.current.abort();
            abortRef.current = null;
            setIsGenerating(false);
            setError("Generation cancelled");
            return;
        }

        if (!isGenerating) {
            updateWizardData({
                generationPrompt: "",
                agentType: "",
                systemPrompt: "",
                whenToUse: "",
                generatedAgent: undefined,
                wasGenerated: false
            });
            setValue("");
            setError(null);
            goBack();
        }
    });

    const handleSubmit = async () => {
        const trimmed = value.trim();
        if (!trimmed) {
            setError("Please describe what the agent should do");
            return;
        }

        setError(null);
        setIsGenerating(true);
        updateWizardData({ generationPrompt: trimmed, isGenerating: true });

        const controller = createAbortController();
        abortRef.current = controller;

        try {
            const generated = await generateAgentDefinition(trimmed, modelId, [], controller.signal);
            updateWizardData({
                agentType: generated.identifier,
                whenToUse: generated.whenToUse,
                systemPrompt: generated.systemPrompt,
                generatedAgent: generated,
                isGenerating: false,
                wasGenerated: true
            });
            goToStep(6);
        } catch (err) {
            if (err instanceof Error && !err.message.includes("No assistant message found")) {
                setError(err.message || "Failed to generate agent");
            }
            updateWizardData({ isGenerating: false });
        } finally {
            setIsGenerating(false);
            abortRef.current = null;
        }
    };

    const subtitle =
        "Describe what this agent should do and when it should be used (be comprehensive for best results)";

    if (isGenerating) {
        return (
            <WizardLayout
                subtitle={subtitle}
                footerText={
                    <>
                        <Shortcut shortcut="Esc" action="cancel" />
                    </>
                }
            >
                <Box marginTop={1} flexDirection="row" alignItems="center">
                    <Text color="suggestion">{figures.info} Generating agent from description...</Text>
                </Box>
            </WizardLayout>
        );
    }

    return (
        <WizardLayout
            subtitle={subtitle}
            footerText={
                <>
                    <Shortcut shortcut="Enter" action="submit" />
                    {"  "}
                    <Shortcut shortcut="Esc" action="go back" />
                </>
            }
        >
            <Box flexDirection="column" marginTop={1}>
                {error && (
                    <Box marginBottom={1}>
                        <Text color="error">{error}</Text>
                    </Box>
                )}
                <TextInput
                    value={value}
                    onChange={setValue}
                    onSubmit={handleSubmit}
                    placeholder="e.g., Help me write unit tests for my code..."
                    columns={80}
                    cursorOffset={cursorOffset}
                    onChangeCursorOffset={setCursorOffset}
                    focus={true}
                    showCursor={true}
                />
            </Box>
        </WizardLayout>
    );
}

// --- Agent Identifier Step (dJ9) ---
export function AgentIdentifierStep(_props?: { existingAgents?: any[] }) {
    const { goNext, goBack, updateWizardData, wizardData } = useWizard();
    const [value, setValue] = useState(wizardData.agentType || "");
    const [error, setError] = useState<string | null>(null);
    const [cursorOffset, setCursorOffset] = useState(value.length);

    useInput((_input, key) => {
        if (key.escape) goBack();
    });

    return (
        <WizardLayout
            subtitle="Agent type (identifier)"
            footerText={
                <>
                    <Shortcut shortcut="Type" action="enter text" />
                    {"  "}
                    <Shortcut shortcut="Enter" action="continue" />
                    {"  "}
                    <Shortcut shortcut="Esc" action="go back" />
                </>
            }
        >
            <Box flexDirection="column" marginTop={1}>
                <Text>Enter a unique identifier for your agent:</Text>
                <Box marginTop={1}>
                    <TextInput
                        value={value}
                        onChange={setValue}
                        onSubmit={(val: string) => {
                            const trimmed = val.trim();
                            const err = validateAgentType(trimmed);
                            if (err) {
                                setError(err);
                                return;
                            }
                            setError(null);
                            updateWizardData({ agentType: trimmed });
                            goNext();
                        }}
                        placeholder="e.g., code-reviewer, tech-lead, etc"
                        columns={60}
                        cursorOffset={cursorOffset}
                        onChangeCursorOffset={setCursorOffset}
                        focus={true}
                        showCursor={true}
                    />
                </Box>
                {error && (
                    <Box marginTop={1}>
                        <Text color="error">{error}</Text>
                    </Box>
                )}
            </Box>
        </WizardLayout>
    );
}

// --- Agent System Prompt Step (cJ9) ---
export function AgentSystemPromptStep() {
    const { goNext, goBack, updateWizardData, wizardData } = useWizard();
    const [value, setValue] = useState(wizardData.systemPrompt || "");
    const [cursorOffset, setCursorOffset] = useState(value.length);
    const [error, setError] = useState<string | null>(null);

    useInput((_input, key) => {
        if (key.escape) goBack();
    });

    return (
        <WizardLayout
            subtitle="System prompt"
            footerText={
                <>
                    <Shortcut shortcut="Type" action="enter text" />
                    {"  "}
                    <Shortcut shortcut="Enter" action="continue" />
                    {"  "}
                    <Shortcut shortcut="Esc" action="go back" />
                </>
            }
        >
            <Box flexDirection="column" marginTop={1}>
                <Text>Enter the system prompt for your agent:</Text>
                <Text dimColor>Be comprehensive for best results</Text>
                <Box marginTop={1}>
                    <TextInput
                        value={value}
                        onChange={setValue}
                        onSubmit={() => {
                            const trimmed = value.trim();
                            if (!trimmed) {
                                setError("System prompt is required");
                                return;
                            }
                            setError(null);
                            updateWizardData({ systemPrompt: trimmed });
                            goNext();
                        }}
                        placeholder="You are a helpful code reviewer who..."
                        columns={80}
                        cursorOffset={cursorOffset}
                        onChangeCursorOffset={setCursorOffset}
                        focus={true}
                        showCursor={true}
                    />
                </Box>
                {error && (
                    <Box marginTop={1}>
                        <Text color="error">{error}</Text>
                    </Box>
                )}
            </Box>
        </WizardLayout>
    );
}

// --- Agent Description Step (iJ9) ---
export function AgentDescriptionStep() {
    const { goNext, goBack, updateWizardData, wizardData } = useWizard();
    const [value, setValue] = useState(wizardData.whenToUse || "");
    const [cursorOffset, setCursorOffset] = useState(value.length);
    const [error, setError] = useState<string | null>(null);

    useInput((_input, key) => {
        if (key.escape) goBack();
    });

    return (
        <WizardLayout
            subtitle="Description (tell Claude when to use this agent)"
            footerText={
                <>
                    <Shortcut shortcut="Type" action="enter text" />
                    {"  "}
                    <Shortcut shortcut="Enter" action="continue" />
                    {"  "}
                    <Shortcut shortcut="Esc" action="go back" />
                </>
            }
        >
            <Box flexDirection="column" marginTop={1}>
                <Text>When should Claude use this agent?</Text>
                <Box marginTop={1}>
                    <TextInput
                        value={value}
                        onChange={setValue}
                        onSubmit={(val: string) => {
                            const trimmed = val.trim();
                            if (!trimmed) {
                                setError("Description is required");
                                return;
                            }
                            setError(null);
                            updateWizardData({ whenToUse: trimmed });
                            goNext();
                        }}
                        placeholder="e.g., use this agent after you're done writing code..."
                        columns={80}
                        cursorOffset={cursorOffset}
                        onChangeCursorOffset={setCursorOffset}
                        focus={true}
                        showCursor={true}
                    />
                </Box>
                {error && (
                    <Box marginTop={1}>
                        <Text color="error">{error}</Text>
                    </Box>
                )}
            </Box>
        </WizardLayout>
    );
}

function isMcpTool(tool: ToolDefinition): boolean {
    return tool.name.startsWith("mcp__");
}

function parseMcpToolName(toolName: string): { serverName: string } | null {
    const match = toolName.match(/^mcp__([^_]+)__/);
    if (!match) return null;
    return { serverName: match[1] };
}

// --- MCP Tool Grouping (UG7) ---
export function groupToolsByServer(tools: ToolDefinition[]) {
    const grouped = new Map<string, ToolDefinition[]>();

    tools.forEach((tool) => {
        if (!isMcpTool(tool)) return;
        const parsed = parseMcpToolName(tool.name);
        if (!parsed?.serverName) return;
        const existing = grouped.get(parsed.serverName) || [];
        existing.push(tool);
        grouped.set(parsed.serverName, existing);
    });

    return Array.from(grouped.entries())
        .map(([serverName, serverTools]) => ({
            serverName,
            tools: serverTools
        }))
        .sort((a, b) => a.serverName.localeCompare(b.serverName));
}
