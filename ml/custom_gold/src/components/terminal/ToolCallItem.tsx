import React, { useMemo } from 'react';
import { Box, Text } from 'ink';
import { useTerminalSize } from '../../hooks/terminal/useTerminalControllerHooks.js';
import { useTheme } from '../../services/terminal/themeManager.js';
import { useAppState } from '../../contexts/AppStateContext.js';

// --- Helper Components (stubs for missing dependencies) ---

const ToolExecutionIndicator: React.FC<{ shouldAnimate: boolean; isUnresolved: boolean; isError: boolean }> = ({ shouldAnimate, isUnresolved, isError }) => {
    if (isError) return <Text color="red">✖</Text>;
    if (isUnresolved) return <Text color="yellow">{shouldAnimate ? '⟳' : '•'}</Text>;
    return <Text color="green">✔</Text>;
};

const HookEvent: React.FC<any> = () => null; // Stub for AK1

// --- Helper Functions ---

function renderToolUseMessage(tool: any, input: any, { theme, verbose, commands }: any) {
    try {
        const result = tool.inputSchema.safeParse(input);
        if (!result.success) return "";
        return tool.renderToolUseMessage(result.data, {
            theme,
            verbose,
            commands
        });
    } catch (error) {
        console.error(`Error rendering tool use message for ${tool.name}: ${error}`);
        return "";
    }
}

function renderToolUseProgress(tool: any, tools: any[], messages: any[], toolUseId: string, progressMessages: any[], { verbose, inProgressToolCallCount }: any, terminalSize: any) {
    const relevantProgress = progressMessages.filter((msg: any) => msg.data.type !== "hook_progress");
    try {
        const renderedProgress = tool.renderToolUseProgressMessage(relevantProgress, {
            tools,
            verbose,
            terminalSize,
            inProgressToolCallCount: inProgressToolCallCount ?? 1
        });
        return (
            <>
                <Box>
                    <HookEvent
                        hookEvent="PreToolUse"
                        messages={messages}
                        toolUseID={toolUseId}
                        verbose={verbose}
                    />
                </Box>
                {renderedProgress}
            </>
        );
    } catch (error) {
        console.error(`Error rendering tool use progress message for ${tool.name}: ${error}`);
        return null;
    }
}

function renderQueuedMessage(tool: any) {
    try {
        return tool.renderToolUseQueuedMessage?.();
    } catch (error) {
        console.error(`Error rendering tool use queued message for ${tool.name}: ${error}`);
        return null;
    }
}

// --- Main Component ---

export function ToolCallItem({
    toolCall,
    addMargin,
    tools,
    commands,
    verbose,
    erroredToolUseIDs,
    inProgressToolUseIDs,
    resolvedToolUseIDs,
    progressMessagesForMessage,
    shouldAnimate,
    shouldShowDot,
    inProgressToolCallCount,
    messages
}: any) {
    const terminalSize = useTerminalSize();
    const appState = useAppState()[0]; // useAppState returns [state, setState]
    const theme = useTheme();

    const pendingWorkerRequest = appState.pendingWorkerRequest;

    if (!tools) {
        console.error(`Tools array is undefined for tool ${toolCall.name}`);
        return null;
    }

    const tool = tools.find((t: any) => t.name === toolCall.name);
    if (!tool) {
        console.error(`Tool ${toolCall.name} not found`);
        return null;
    }

    const isResolved = resolvedToolUseIDs.has(toolCall.id);
    const isUnresolved = !inProgressToolUseIDs.has(toolCall.id) && !isResolved;
    const isPendingWorker = pendingWorkerRequest?.toolUseId === toolCall.id;

    const inputResult = tool.inputSchema.safeParse(toolCall.input);
    const userFacingName = tool.userFacingName(inputResult.success ? inputResult.data : undefined);
    const nameBgColor = tool.userFacingNameBackgroundColor?.(inputResult.success ? inputResult.data : undefined);

    if (userFacingName === "") return null;

    const toolMessage = inputResult.success ? renderToolUseMessage(tool, inputResult.data, {
        theme,
        verbose,
        commands
    }) : null;

    if (toolMessage === null) return null;

    return (
        <Box
            flexDirection="row"
            justifyContent="space-between"
            marginTop={addMargin ? 1 : 0}
            width="100%"
        >
            <Box flexDirection="column">
                <Box flexDirection="row" flexWrap="nowrap" minWidth={userFacingName.length + (shouldShowDot ? 2 : 0)}>
                    {shouldShowDot && (isUnresolved ? (
                        <Box minWidth={2}>
                            <Text dimColor>•</Text>
                        </Box>
                    ) : (
                        <ToolExecutionIndicator
                            shouldAnimate={shouldAnimate}
                            isUnresolved={!isResolved}
                            isError={erroredToolUseIDs.has(toolCall.id)}
                        />
                    ))}

                    <Box flexShrink={0}>
                        <Text
                            bold
                            wrap="truncate-end"
                            backgroundColor={nameBgColor}
                            color={nameBgColor ? "inverse" : undefined} // ink uses 'inverse' or contrast color
                        >
                            {userFacingName}
                        </Text>
                    </Box>

                    {toolMessage !== "" && (
                        <Box flexWrap="nowrap">
                            <Text>({toolMessage})</Text>
                        </Box>
                    )}

                    {inputResult.success && tool.renderToolUseTag && tool.renderToolUseTag(inputResult.data)}
                </Box>

                {!isResolved && !isUnresolved && (isPendingWorker ? (
                    <Box height={1}>
                        <Text dimColor>Waiting for permission…</Text>
                    </Box>
                ) : (
                    renderToolUseProgress(tool, tools, messages, toolCall.id, progressMessagesForMessage, {
                        verbose,
                        inProgressToolCallCount
                    }, terminalSize)
                ))}

                {!isResolved && isUnresolved && renderQueuedMessage(tool)}
            </Box>
        </Box>
    );
}

// Re-export stubs/utils if needed by other files (though we moved them)
// For compatibility with previous mocked version if anything imports them:
export { uuidv4 } from '../../utils/uuid.js';
// export { Emitter } from '../../utils/events/SimpleEmitter'; 
