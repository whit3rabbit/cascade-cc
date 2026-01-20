import React, { useMemo } from 'react';
import { Box, Text } from 'ink';
import { ToolCallItem } from './ToolCallItem.js';
import { useTerminalSize } from '../../hooks/terminal/useTerminalControllerHooks.js';
// import { MessageAdapter as ParsedMessage } from './MessageAdapter.js'; // Not used yet
import { Shortcut } from '../shared/Shortcut.js';

// --- Stubbed Helper Functions (referenced in chunk_526.ts but external) ---
const formatCompactNumber = (n: number) => n < 1000 ? n : (n / 1000).toFixed(1) + 'k'; // L3 stub

// --- Helper Functions from chunk_526.ts (OB7, MB7, wq0, qq0, etc.) ---

function getToolUsageStats(progressMessages: any[]) {
    const toolUseCount = progressMessages.filter(msg =>
        msg.data.message?.message?.content?.some((c: any) => c.type === "tool_use")
    ).length;

    const lastAssistantMsg = [...progressMessages].reverse().find(msg => msg.data.message?.type === "assistant");
    let tokens = null;

    if (lastAssistantMsg?.data?.message?.type === "assistant") {
        const usage = lastAssistantMsg.data.message.message.usage;
        if (usage) {
            tokens = (usage.cache_creation_input_tokens ?? 0) +
                (usage.cache_read_input_tokens ?? 0) +
                usage.input_tokens +
                usage.output_tokens;
        }
    }

    return { toolUseCount, tokens };
}

function getLastToolInfo(progressMessages: any[], tools: any[]) {
    const lastUserMsg = [...progressMessages].reverse().find(msg => {
        const m = msg.data.message;
        return m?.type === "user" && m.message?.content?.some((c: any) => c.type === "tool_result");
    });

    if (lastUserMsg?.data.message.type === "user") {
        const toolResult = lastUserMsg.data.message.message.content.find((c: any) => c.type === "tool_result");
        if (toolResult?.type === "tool_result") {
            const toolUseId = toolResult.tool_use_id;
            const assistantMsg = progressMessages.find(msg => {
                const m = msg.data.message;
                return m?.type === "assistant" && m.message?.content?.some((c: any) => c.type === "tool_use" && c.id === toolUseId);
            });

            if (assistantMsg?.data.message.type === "assistant") {
                const toolUse = assistantMsg.data.message.message.content.find((c: any) => c.type === "tool_use" && c.id === toolUseId);
                if (toolUse?.type === "tool_use") {
                    const tool = tools.find((t: any) => t.name === toolUse.name);
                    if (!tool) return toolUse.name;

                    const parseResult = tool.inputSchema.safeParse(toolUse.input);
                    const userFacingName = tool.userFacingName(parseResult.success ? parseResult.data : undefined);

                    if (tool.getToolUseSummary) {
                        const summary = tool.getToolUseSummary(parseResult.success ? parseResult.data : undefined);
                        if (summary) return `${userFacingName}: ${summary}`;
                    }
                    return userFacingName;
                }
            }
        }
    }
    return null;
}

function getAgentType(param: any) {
    // JX1.agentType stub - assuming "orchestrator" or similar constant
    if (param?.subagent_type && param.subagent_type !== "orchestrator") {
        return param.subagent_type;
    }
    return "Task";
}

function getAgentColor(param: any) {
    if (!param?.subagent_type) return undefined;
    // vKA(param.subagent_type) stub - assuming color mapping
    const colors: Record<string, string> = {
        "planner": "magenta",
        "researcher": "cyan",
        "coder": "blue",
        "writer": "green"
    };
    return colors[param.subagent_type];
}

const MAX_VISIBLE_MESSAGES = 3;
const MESSAGE_ROW_HEIGHT = 9; // qB7
const MESSAGE_HEADER_HEIGHT = 7; // NB7


// --- InProgressToolList (PV1) ---

export function InProgressToolList({
    progressMessages,
    tools,
    verbose,
    terminalSize,
    inProgressToolCallCount
}: {
    progressMessages: any[],
    tools: any[],
    verbose?: boolean,
    terminalSize?: { rows: number, cols: number },
    inProgressToolCallCount?: number // Z
}) {
    if (!progressMessages.length) {
        return (
            <Box height={1}>
                <Text dimColor>Initializing…</Text>
            </Box>
        );
    }

    const estimatedHeight = (inProgressToolCallCount ?? 1) * MESSAGE_ROW_HEIGHT + MESSAGE_HEADER_HEIGHT;
    const isTooLarge = !verbose && terminalSize && terminalSize.rows && terminalSize.rows < estimatedHeight;

    if (isTooLarge) {
        const { toolUseCount, tokens } = getToolUsageStats(progressMessages);
        return (
            <Box height={1}>
                <Text dimColor>
                    In progress… · <Text bold>{toolUseCount}</Text> tool {toolUseCount === 1 ? "use" : "uses"}
                    {tokens ? ` · ${formatCompactNumber(tokens)} tokens` : ""} · <Shortcut shortcut="ctrl+o" action="expand" parens />
                </Text>
            </Box>
        );
    }

    const visibleMessages = verbose ? progressMessages : progressMessages.slice(-MAX_VISIBLE_MESSAGES);
    const hiddenCount = progressMessages.length - visibleMessages.length;
    const initialPrompt = progressMessages[0]?.data?.prompt;

    return (
        <Box>
            <Box flexDirection="column">
                <Box>
                    {verbose && initialPrompt && (
                        <Box marginBottom={1}>
                            <Text>Prompt: {initialPrompt}</Text>
                            {/* IbA was prompt component, simplified heavily here */}
                        </Box>
                    )}
                    {visibleMessages.map((msg: any) => {
                        if (msg.type === "summary") {
                            // Logic from Dr2 stubbed
                            const isActive = msg.isActive;
                            const status = isActive ? "Searching..." : `Found ${msg.searchCount} results`;
                            return (
                                <Box key={msg.uuid} height={1} overflow="hidden">
                                    <Text dimColor>{status}</Text>
                                </Box>
                            );
                        }

                        // RS -> ToolUseRenderer? Or MessageRenderer?
                        // Assuming ToolCallItem-like logic or actually calls ToolCallItem if it's a tool use
                        // The original passes messages to RS. 
                        // For now, let's render a basic placeholder if it's not a tool use itself, 
                        // but if it is, we need to handle it.
                        // Actually, visibleMessages here are mostly `progress` type from `AgentOutputAndToolUse` wrapper.

                        // Let's assume we render message content:
                        return (
                            <Box key={msg.message?.uuid ?? msg.uuid} flexDirection="column">
                                <Text dimColor>
                                    [{msg.data?.type || msg.type}] {msg.data?.message?.message?.content?.[0]?.text?.substring(0, 50)}...
                                </Text>
                            </Box>
                        );
                    })}
                </Box>
                {hiddenCount > 0 && (
                    <Text dimColor>+{hiddenCount} more tool {hiddenCount === 1 ? "use" : "uses"} </Text>
                )}
            </Box>
        </Box>
    );
}

// --- AgentToolUseList (v99) ---

function AgentToolUseListItem({
    agentType,
    description,
    toolUseCount,
    tokens,
    color,
    isLast,
    isResolved,
    isError,
    isAsync,
    shouldAnimate,
    lastToolInfo,
    hideType
}: any) {
    // ToolCallItem-like display for agents
    return (
        <Box flexDirection="column" paddingLeft={1}>
            <Box>
                <Text color={isError ? 'red' : color || 'green'}>
                    {!hideType && <Text bold>[{agentType}] </Text>}
                    {description || "Task"}
                    {" "}
                    <Text dimColor>
                        {isResolved
                            ? `(Done${tokens ? `, ${formatCompactNumber(tokens)} tokens` : ""})`
                            : `(Running${toolUseCount > 0 ? `, ${toolUseCount} tools` : ""}...)`
                        }
                    </Text>
                </Text>
            </Box>
            {isAsync && !isResolved && <Text dimColor>  ↳ Running in background</Text>}
            {!isResolved && lastToolInfo && <Text dimColor>  ↳ {lastToolInfo}</Text>}
        </Box>
    );
}

export function AgentToolUseList({
    toolUses,
    shouldAnimate,
    tools
}: {
    toolUses: any[],
    shouldAnimate: boolean,
    tools: any[] // G
}) {
    const processedToolUses = toolUses.map(({ param, isResolved, isError, progressMessages }: any) => {
        const stats = getToolUsageStats(progressMessages); // OB7(F)
        const lastTool = getLastToolInfo(progressMessages, tools); // MB7(F, G)

        let subagentType = "Task";
        let description = undefined;
        let color = undefined;
        let isAsync = false;

        // Param is likely the tool input args for the subagent tool
        if (param && param.input) {
            subagentType = getAgentType(param.input); // wq0
            description = param.input.description;
            color = getAgentColor(param.input); // qq0
            isAsync = param.input.run_in_background === true;
        }

        return {
            id: param.id,
            agentType: subagentType,
            description,
            toolUseCount: stats.toolUseCount,
            tokens: stats.tokens,
            isResolved,
            isError,
            isAsync,
            color,
            lastToolInfo: lastTool
        };
    });

    const hasUnresolved = toolUses.some((t: any) => !t.isResolved);
    const hasError = toolUses.some((t: any) => t.isError);
    const isFinished = !hasUnresolved;

    // Check if all agents are of same type to optionally hide redundant labels
    const allSameType = processedToolUses.length > 0 && processedToolUses.every((item: any) => item.agentType === processedToolUses[0]?.agentType);
    const commonType = allSameType ? processedToolUses[0]?.agentType : null;
    const allAsync = processedToolUses.every((item: any) => item.isAsync);

    return (
        <Box flexDirection="column" marginTop={1}>
            <Box flexDirection="row">
                <Box marginRight={1}>
                    {/* C4A replacement */}
                    {hasUnresolved ? (shouldAnimate ? <Text color="yellow">⟳</Text> : <Text color="yellow">•</Text>) : (hasError ? <Text color="red">✖</Text> : <Text color="green">✔</Text>)}
                </Box>
                <Text>
                    {isFinished ? (
                        <>
                            <Text bold>{toolUses.length}</Text> {commonType ? `${commonType} agents` : "agents"} {allAsync ? "launched" : "finished"}
                        </>
                    ) : (
                        <>
                            Running <Text bold>{toolUses.length}</Text> {commonType ? `${commonType} agents` : "agents"}…
                        </>
                    )}
                </Text>
            </Box>

            {processedToolUses.map((item: any, index: number) => (
                <AgentToolUseListItem
                    key={item.id}
                    {...item}
                    isLast={index === processedToolUses.length - 1}
                    shouldAnimate={shouldAnimate}
                    hideType={allSameType}
                />
            ))}
        </Box>
    );
}

// --- Main Helper Wrapper (x99) ---

export function AgentOutputAndToolUse({ messages, tools, verbose }: any) {
    const progressMessages = messages.filter((m: any) => m.type === "progress" || m.type === "tool_use");

    // The original x99/y99 wraps PV1.
    // If it's a subagent result, it might use y99/o3 logic?
    // For now we map to InProgressToolList as per x99 logic.

    return (
        <>
            <InProgressToolList
                progressMessages={progressMessages}
                tools={tools}
                verbose={verbose}
            />
        </>
    );
}
