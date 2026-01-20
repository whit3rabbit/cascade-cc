// Logic from chunk_507.ts (Message View Adapter & Tool Grouping)

import React from 'react';
import { Box, Text } from 'ink';
import { MessageContentRenderer, AttachmentRenderer } from './MessageRenderer.js';
import { ToolCallItem } from './ToolCallItem.js';

// --- Helper for Tool Grouping ---
function isSearchOrReadCommand(toolName: string, input: any) {
    // Stub logic from Ir2
    // Real logic handles: grep, glob, various file reads
    // For now we assume names match the tool names
    const isSearch = toolName === "grep" || toolName === "glob" || toolName.includes("search");
    const isRead = toolName === "read_file" || toolName.includes("read");
    return { isSearch, isRead, isCollapsible: isSearch || isRead };
}

// --- Message Adapter (cA7) ---
export function MessageViewAdapter({
    message,
    tools,
    verbose,
    normalizedMessages,
    resolvedToolUseIDs,
    erroredToolUseIDs,
    inProgressToolUseIDs
}: any) {
    const addMargin = true; // defaulting to true for spacing

    switch (message.type) {
        case "attachment":
            return (
                <AttachmentRenderer
                    attachment={message.attachment}
                    addMargin={addMargin}
                    verbose={verbose}
                />
            );
        case "assistant":
            return (
                <Box flexDirection="column" width="100%">
                    {message.message.content.map((content: any, i: number) => (
                        <AssistantContent
                            key={i}
                            content={content}
                            tools={tools}
                            verbose={verbose}
                            erroredToolUseIDs={erroredToolUseIDs}
                            inProgressToolUseIDs={inProgressToolUseIDs}
                            resolvedToolUseIDs={resolvedToolUseIDs}
                        />
                    ))}
                </Box>
            );
        case "user":
            return (
                <Box flexDirection="column" width="100%">
                    {message.message.content.map((content: any, i: number) => (
                        <UserContent key={i} content={content} verbose={verbose} />
                    ))}
                </Box>
            );
        case "collapsed_read_search":
            return <CollapsedToolSummary
                message={message}
                resolvedToolUseIDs={resolvedToolUseIDs}
                erroredToolUseIDs={erroredToolUseIDs}
                verbose={verbose}
                tools={tools}
                normalizedMessages={normalizedMessages}
            />;
        case "grouped_tool_use":
            return <GroupedToolUse
                message={message}
                tools={tools}
                normalizedMessages={normalizedMessages}
                resolvedToolUseIDs={resolvedToolUseIDs}
                erroredToolUseIDs={erroredToolUseIDs}
                inProgressToolUseIDs={inProgressToolUseIDs}
                shouldAnimate={true}
            />;
        case "system":
            if (message.subtype === "compact_boundary") {
                return (
                    <Box borderStyle="single" borderColor="gray" paddingX={1} marginY={1}>
                        <Text dimColor>Conversation compacted · ctrl+o for history</Text>
                    </Box>
                );
            }
            if (message.subtype === "local_command") {
                return <MessageContentRenderer param={{ type: "text", text: message.content }} isVerbose={verbose} />;
            }
            // Generic system message fallback
            return <Box marginY={1}><Text dimColor>{message.content}</Text></Box>;
        default:
            return null;
    }
}

// --- Assistant Content Switch (iA7) ---
function AssistantContent({
    content,
    tools,
    verbose,
    erroredToolUseIDs,
    inProgressToolUseIDs,
    resolvedToolUseIDs
}: any) {
    if (content.type === "tool_use") {
        const tool = tools?.find((t: any) => t.name === content.name);
        const isResolved = resolvedToolUseIDs?.has(content.id);
        const isError = erroredToolUseIDs?.has(content.id);
        const isInProgress = inProgressToolUseIDs?.has(content.id);

        return (
            <ToolCallItem
                toolCall={content}
                tool={tool}
                isResolved={isResolved}
                isError={isError}
                isInProgress={isInProgress}
                shouldAnimate={isInProgress}
                verbose={verbose}
            />
        );
    }

    // text, thinking, redacted_thinking
    return (
        <MessageContentRenderer
            param={content}
            isVerbose={verbose}
            addMargin={true}
        />
    );
}

// --- User Content (lA7) ---
function UserContent({ content, verbose }: any) {
    if (content.type === "image") {
        return <Box marginTop={1}><Text dimColor>[Image]</Text></Box>;
    }
    if (content.type === "tool_result") {
        // Render tool result
        // Simplified for now
        return (
            <Box marginTop={1} borderStyle="round" borderColor="gray" paddingX={1}>
                <Text dimColor>Tool Result for {content.tool_use_id}</Text>
                {verbose && <Text>{JSON.stringify(content.content)}</Text>}
            </Box>
        );
    }
    // text
    return <MessageContentRenderer param={content} isVerbose={verbose} addMargin={true} />;
}


// --- Collapsed Tools (Fr2) ---
function CollapsedToolSummary({ message, verbose, tools, normalizedMessages, resolvedToolUseIDs, erroredToolUseIDs }: any) {
    const { searchCount, readCount, messages } = message;

    if (verbose) {
        // Expand the collapsed group
        // Logic from Fr2 verbose branch: filter messages and render individually
        const assistantMessages = messages.filter((m: any) => m.type === "assistant" || m.type === "grouped_tool_use");
        // Flatten grouped
        const expanded: any[] = [];
        for (const m of assistantMessages) {
            if (m.type === "grouped_tool_use") expanded.push(...m.messages);
            else expanded.push(m);
        }

        return (
            <Box flexDirection="column">
                {expanded.map((am: any) => {
                    const content = am.message.content[0];
                    if (content?.type !== "tool_use") return null;
                    return (
                        <AssistantContent
                            key={content.id}
                            content={content}
                            tools={tools}
                            verbose={verbose}
                        // Pass through IDs
                        />
                    );
                })}
            </Box>
        );
    }

    const parts = [];
    // "Searched for X patterns"
    if (searchCount > 0) {
        parts.push(
            <Text key="search">
                Searched for <Text bold>{searchCount}</Text> {searchCount === 1 ? "pattern" : "patterns"}
            </Text>
        );
    }
    // "Read Y files"
    if (readCount > 0) {
        if (parts.length > 0) parts.push(<Text key="comma">, </Text>);
        parts.push(
            <Text key="read">
                Read <Text bold>{readCount}</Text> {readCount === 1 ? "file" : "files"}
            </Text>
        );
    }

    // Dot indicator
    const hasError = false; // Stub error detection logic

    return (
        <Box flexDirection="row" marginTop={1}>
            <Box marginRight={1}>
                <Text color={hasError ? "red" : "green"}>•</Text>
            </Box>
            <Box>
                {parts}
                <Text dimColor> (ctrl+o to expand)</Text>
            </Box>
        </Box>
    );
}

// --- Grouped Tool Use (Yr2) ---
function GroupedToolUse({
    message,
    tools,
    shouldAnimate
}: any) {
    const toolName = message.toolName;
    const toolDef = tools?.find((t: any) => t.name === toolName);

    // If the tool definition provides a custom grouper render, use it
    // Note: In typical Mcp/Tool definitions, `renderGroupedToolUse` might exist
    if (toolDef?.renderGroupedToolUse) {
        // Transform message format to what renderGroupedToolUse expects
        // Stub logic
        return (
            <Box marginTop={1}>
                <Text>Grouped Tool Use: {toolName} (custom render)</Text>
            </Box>
        );
    }

    // Fallback: render individual messages
    return (
        <Box flexDirection="column">
            {message.messages.map((m: any, i: number) => {
                const content = m.message.content[0];
                return (
                    <AssistantContent
                        key={i}
                        content={content}
                        tools={tools}
                        verbose={false}
                    />
                );
            })}
        </Box>
    );
}

