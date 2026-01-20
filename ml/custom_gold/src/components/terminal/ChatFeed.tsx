// Logic from chunk_511.ts (Chat Feed & Task Dialog)

import React, { useContext, useMemo, useState } from "react";
import { Box, Text, useInput, Static } from "ink";
import { MessageViewAdapter } from "./MessageAdapter.js";
import { AppDashboard, SideConversationView, isMessageUnresolved } from "./AppDashboard.js";
import { NoticeList } from "./NoticeList.js";
import { TerminalProgressBar } from "./AgentDetails.js";
import { figures } from "../../vendor/terminalFigures.js";
import { useCtrlExit } from "../../hooks/useCtrlExit.js";
import { useAppState } from "../../contexts/AppStateContext.js";
import { Separator } from "./Separator.js";

const MAX_TRANSCRIPT_TAIL = 10;
const INPUT_TRUNCATE_LIMIT = 10000;
const INPUT_TRUNCATE_WINDOW = 1000;

const TerminalFocusContext = React.createContext<boolean>(false);

type MessageItem = {
    uuid?: string;
    type: string;
    message?: any;
    subtype?: string;
    messages?: any[];
    timestamp?: string;
    model?: string;
};

type ToolDefinition = {
    name: string;
};

type ToolUseWrapper = {
    contentBlock: { id: string };
};

type ChatFeedProps = {
    messages: any[];
    normalizedMessageHistory: any[];
    tools: ToolDefinition[];
    commands: any[];
    verbose: boolean;
    toolJSX?: { shouldContinueAnimation?: boolean } | null;
    toolUseConfirmQueue: any[];
    inProgressToolUseIDs: Set<string>;
    isMessageSelectorVisible: boolean;
    conversationId: string;
    screen: string;
    screenToggleId: string;
    streamingToolUses: ToolUseWrapper[];
    showAllInTranscript?: boolean;
    agentDefinitions: any;
    onOpenRateLimitOptions?: () => void;
    sideQuestionResponse?: { question: string; response?: string };
    hideLogo?: boolean;
    isLoading: boolean;
    renderCustomMessage?: (message: MessageItem) => React.ReactNode;
};

type FeedItem = { type: "static" | "transient"; jsx: React.ReactElement };

type TaskItem = {
    id: string;
    type: "local_bash" | "remote_agent" | "local_agent";
    label: string;
    status: string;
    task: any;
};

type ToolUseContext = {
    abortController?: AbortController;
    getAppState?: () => Promise<any>;
    setAppState?: any;
};

function normalizeMessages(messages: any[]): MessageItem[] {
    return messages as MessageItem[];
}

function shouldIncludeMessage(message: MessageItem): boolean {
    return message.type !== "progress";
}

function getResolvedToolUseIds(_messages: MessageItem[]): Record<string, boolean> {
    return {};
}

function getErroredToolUseIds(_messages: MessageItem[]): Set<string> {
    return new Set();
}

function getPendingToolUseIds(_messages: MessageItem[]): Set<string> {
    return new Set();
}

function filterToolUseMessages(messages: MessageItem[]): MessageItem[] {
    return messages;
}

function mergeStreamingToolUses(messages: MessageItem[], toolUses: MessageItem[]): MessageItem[] {
    return [...messages, ...toolUses];
}

function buildCollapsedMessages(messages: MessageItem[], _tools: ToolDefinition[], _verbose: boolean) {
    return { messages };
}

function buildDisplayMessages(messages: MessageItem[], _verbose: boolean, _tools: ToolDefinition[]) {
    return messages;
}

function getProgressMessagesForMessage(_message: MessageItem, _progressByMessage: Map<string, any>) {
    return [] as any[];
}

function getToolUseIdsForMessage(_message: MessageItem, _progressByMessage: Map<string, any>) {
    return new Set<string>();
}

function isCollapsedGroupInProgress(_message: MessageItem, _inProgress: Set<string>): boolean {
    return false;
}

function getMessageKey(message: MessageItem): string {
    return message.uuid ?? message.type;
}

function getToolUseId(message: MessageItem): string | null {
    if (!message) return null;
    if (message.message?.content?.[0]?.id) return message.message.content[0].id;
    return message.message?.id ?? null;
}

function resolveCollapsedMessage(message: MessageItem): MessageItem {
    return message;
}

function buildMessageProgressMap(_messages: MessageItem[], _displayMessages: MessageItem[]) {
    return new Map();
}

function joinWithSeparator(items: React.ReactElement[], separator: (value: string) => React.ReactElement) {
    if (items.length <= 1) return items;
    const output: React.ReactElement[] = [];
    items.forEach((item, index) => {
        output.push(item);
        if (index < items.length - 1) output.push(separator(String(index)));
    });
    return output;
}

import { ShortcutHint, ShortcutGroup } from "../shared/Shortcut.js";

export const ChatFeedView = React.memo(function ChatFeedView({
    messages,
    normalizedMessageHistory,
    tools,
    commands,
    verbose,
    toolJSX,
    toolUseConfirmQueue,
    inProgressToolUseIDs,
    isMessageSelectorVisible,
    conversationId,
    screen,
    screenToggleId,
    streamingToolUses,
    showAllInTranscript = false,
    agentDefinitions,
    onOpenRateLimitOptions,
    sideQuestionResponse,
    hideLogo = false,
    isLoading
}: ChatFeedProps) {
    const columns = process.stdout?.columns ?? 80;
    const focus = useContext(TerminalFocusContext);
    const mergedHistory = useMemo(
        () => [...normalizedMessageHistory, ...normalizeMessages(messages).filter(shouldIncludeMessage)],
        [messages, normalizedMessageHistory]
    );
    const resolvedToolUseIds = useMemo(() => new Set(Object.keys(getResolvedToolUseIds(mergedHistory))), [
        mergedHistory
    ]);
    const erroredToolUseIds = useMemo(() => getErroredToolUseIds(mergedHistory), [mergedHistory]);
    const _queuedToolUseIds = useMemo(() => getPendingToolUseIds(mergedHistory), [mergedHistory]);
    const filteredToolUses = useMemo(
        () =>
            streamingToolUses.filter((toolUse) => {
                if (inProgressToolUseIDs.has(toolUse.contentBlock.id)) return false;
                if (
                    mergedHistory.some(
                        (message) =>
                            message.type === "assistant" &&
                            message.message?.content?.[0]?.type === "tool_use" &&
                            message.message?.content?.[0]?.id === toolUse.contentBlock.id
                    )
                ) {
                    return false;
                }
                return true;
            }),
        [streamingToolUses, inProgressToolUseIDs, mergedHistory]
    );
    const streamingMessages = useMemo(
        () => filteredToolUses.flatMap((toolUse) => normalizeMessages([{ contentBlock: toolUse.contentBlock }])),
        [filteredToolUses]
    );

    const feedItems = useMemo(() => {
        const isTranscript = screen === "transcript";
        const shouldLimitTail = isTranscript && !showAllInTranscript;
        const baseMessages = verbose ? mergedHistory : filterToolUseMessages(mergedHistory);
        const filtered = baseMessages.filter((message) => message.type !== "progress");
        const merged = mergeStreamingToolUses(
            filtered.filter((message) => message.type !== "progress"),
            streamingMessages
        );
        const tailMessages = shouldLimitTail ? merged.slice(-MAX_TRANSCRIPT_TAIL) : merged;
        const shouldShowTruncation = shouldLimitTail && merged.length > MAX_TRANSCRIPT_TAIL;

        const header: FeedItem = {
            type: "static",
            jsx: (
                <Box flexDirection="column" gap={1} key={`logo-${conversationId}-${screenToggleId}`}>
                    <AppDashboard isBeforeFirstMessage={false} />
                    <NoticeList agentDefinitions={agentDefinitions} />
                </Box>
            )
        };

        const headerItems = hideLogo ? [] : [header];
        const truncationItems = shouldShowTruncation
            ? [
                {
                    type: "static" as const,
                    jsx: (
                        <Separator
                            key={`truncation-indicator-${conversationId}-${screenToggleId}`}
                            title={`Ctrl+E to show ${merged.length - MAX_TRANSCRIPT_TAIL} previous messages`}
                        />
                    )
                }
            ]
            : [];

        const hideIndicatorItems =
            isTranscript && showAllInTranscript && merged.length > MAX_TRANSCRIPT_TAIL
                ? [
                    {
                        type: "static" as const,
                        jsx: (
                            <Separator
                                key={`hide-indicator-${conversationId}-${screenToggleId}`}
                                title={`Ctrl+E to hide ${merged.length - MAX_TRANSCRIPT_TAIL} previous messages`}
                            />
                        )
                    }
                ]
                : [];

        const { messages: collapsed } = buildCollapsedMessages(tailMessages, tools, verbose);
        const displayMessages = buildDisplayMessages(collapsed, verbose, tools);
        const progressMap = buildMessageProgressMap(mergedHistory, tailMessages);
        const streamingIds = new Set(streamingToolUses.map((toolUse) => toolUse.contentBlock.id));
        const canAnimate = (!toolJSX || toolJSX.shouldContinueAnimation !== false) &&
            toolUseConfirmQueue.length === 0 &&
            !isMessageSelectorVisible;
        const lastDisplayIndex = displayMessages.length - 1;

        const messageItems = displayMessages.map((message, index) => {
            const isGrouped = message.type === "grouped_tool_use";
            const isCollapsed = message.type === "collapsed_read_search";
            const isActiveCollapsed = isCollapsed && index === lastDisplayIndex && isLoading;
            const previous = index > 0 ? displayMessages[index - 1] : null;
            const isUserContinuation = message.type === "user" && previous?.type === "user";
            const displayMessage = isGrouped ? (message as any).displayMessage : isCollapsed ? resolveCollapsedMessage(message) : message;
            const progressMessages = isGrouped || isCollapsed ? [] : getProgressMessagesForMessage(message, progressMap);
            const resolvedIds = isGrouped || isCollapsed ? new Set<string>() : getToolUseIdsForMessage(message, progressMap);
            const itemType = isMessageUnresolved(
                message,
                streamingIds,
                resolvedToolUseIds,
                inProgressToolUseIDs,
                resolvedIds,
                screen,
                progressMap
            )
                ? "static"
                : "transient";
            let shouldAnimate = false;

            if (canAnimate) {
                if (isGrouped) {
                    shouldAnimate = (message as any).messages?.some((entry: any) => {
                        const content = entry.message.content[0];
                        return content?.type === "tool_use" && inProgressToolUseIDs.has(content.id);
                    });
                } else if (isCollapsed) {
                    shouldAnimate = isCollapsedGroupInProgress(message, inProgressToolUseIDs);
                } else {
                    const toolUseId = getToolUseId(message);
                    shouldAnimate = !toolUseId || inProgressToolUseIDs.has(toolUseId);
                }
            }

            return {
                type: itemType as "static" | "transient",
                jsx: (
                    <Box
                        key={`${getMessageKey(message)}-${conversationId}-${screenToggleId}`}
                        width={columns}
                        flexDirection="row"
                        flexWrap="nowrap"
                        alignItems="flex-start"
                        justifyContent="space-between"
                        gap={1}
                    >
                        <MessageViewAdapter
                            message={message}
                            messages={mergedHistory}
                            addMargin={true}
                            tools={tools}
                            commands={commands}
                            verbose={verbose}
                            erroredToolUseIDs={erroredToolUseIds}
                            inProgressToolUseIDs={inProgressToolUseIDs}
                            progressMessagesForMessage={progressMessages}
                            shouldAnimate={shouldAnimate}
                            shouldShowDot={true}
                            resolvedToolUseIDs={resolvedToolUseIds}
                            isTranscriptMode={isTranscript}
                            isStatic={itemType === "static"}
                            onOpenRateLimitOptions={onOpenRateLimitOptions}
                            isActiveCollapsedGroup={isActiveCollapsed}
                            isUserContinuation={isUserContinuation}
                        />
                        <MessageTimestamp message={displayMessage} isTranscriptMode={isTranscript} />
                        <MessageModelLabel message={displayMessage} isTranscriptMode={isTranscript} />
                    </Box>
                )
            };
        });

        return [...headerItems, ...truncationItems, ...hideIndicatorItems, ...messageItems];
    }, [
        hideLogo,
        screen,
        showAllInTranscript,
        verbose,
        mergedHistory,
        streamingMessages,
        conversationId,
        screenToggleId,
        agentDefinitions,
        columns,
        streamingToolUses,
        resolvedToolUseIds,
        tools,
        commands,
        erroredToolUseIds,
        inProgressToolUseIDs,
        toolJSX,
        toolUseConfirmQueue.length,
        isMessageSelectorVisible,
        onOpenRateLimitOptions,
        isLoading
    ]);

    const hasInProgress = inProgressToolUseIDs.size > 0;

    if (focus) {
        return (
            <>
                {feedItems.map((item) => item.jsx)}
                {sideQuestionResponse && (
                    <SideConversationView
                        question={sideQuestionResponse.question}
                        response={sideQuestionResponse.response}
                    />
                )}
                <TerminalProgressBar state={hasInProgress ? "indeterminate" : "completed"} percentage={0} />
            </>
        );
    }

    const staticItems = feedItems.filter((item) => item.type === "static");
    const transientItems = feedItems.filter((item) => item.type === "transient");

    return (
        <>
            <Static items={staticItems}>
                {(item, index) => <Box key={index}>{item.jsx}</Box>}
            </Static>
            {transientItems.map((item) => item.jsx)}
            {sideQuestionResponse && (
                <SideConversationView
                    question={sideQuestionResponse.question}
                    response={sideQuestionResponse.response}
                />
            )}
            <TerminalProgressBar state={hasInProgress ? "indeterminate" : "completed"} percentage={0} />
        </>
    );
}, areChatFeedPropsEqual);

export const ChatFeed = ChatFeedView;

function areChatFeedPropsEqual(prev: ChatFeedProps, next: ChatFeedProps): boolean {
    const keys = Object.keys(prev) as (keyof ChatFeedProps)[];
    for (const key of keys) {
        if (key === "onOpenRateLimitOptions") continue;
        if (prev[key] !== next[key]) {
            if (key === "streamingToolUses") {
                const prevUses = prev.streamingToolUses;
                const nextUses = next.streamingToolUses;
                if (
                    prevUses.length === nextUses.length &&
                    prevUses.every((use, index) => use.contentBlock === nextUses[index]?.contentBlock)
                ) {
                    continue;
                }
            }
            return false;
        }
    }
    return true;
}

export function ShellStatusView({ shell }: { shell: any }) {
    switch (shell.status) {
        case "completed":
            return (
                <Text color="success" dimColor>
                    done
                </Text>
            );
        case "failed":
            return (
                <Text color="error" dimColor>
                    error
                </Text>
            );
        case "killed":
            return (
                <Text color="error" dimColor>
                    killed
                </Text>
            );
        case "running":
        case "pending": {
            const output = getShellOutput(shell.id);
            const lastLine = getLastShellLine(output);
            if (!lastLine) return <Text dimColor>no output</Text>;
            return <Text dimColor>{wrapText(lastLine, 20, true)}</Text>;
        }
        default:
            return null;
    }
}

function getShellOutput(_shellId: string): string {
    return "";
}

function getLastShellLine(output: string): string {
    if (!output) return "";
    const lines = output.split("\n");
    for (let i = lines.length - 1; i >= 0; i -= 1) {
        const line = lines[i]?.trim();
        if (line) return line;
    }
    return "";
}

export function TaskStatusView({ task }: { task: any }) {
    switch (task.type) {
        case "local_bash":
            return (
                <Text>
                    {wrapText(task.command, 40, true)} <ShellStatusView shell={task} />
                </Text>
            );
        case "remote_agent":
            return (
                <Text>
                    {wrapText(task.title, 40, true)} <TerminalProgressBar state={task.status} percentage={0} />
                </Text>
            );
        case "local_agent":
            return (
                <Text>
                    {wrapText(task.description, 40, true)}{" "}
                    <Text dimColor>
                        ({task.status}
                        {task.status === "completed" && !task.notified && ", unread"})
                    </Text>
                </Text>
            );
        default:
            return null;
    }
}

export function BackgroundTasksDialog({ onDone, toolUseContext }: { onDone: any; toolUseContext: ToolUseContext }) {
    const [state, setAppState] = useAppState();
    const [viewState, setViewState] = useState<{ mode: "list" } | { mode: "detail"; itemId: string }>({
        mode: "list"
    });
    const [selectedIndex, setSelectedIndex] = useState(0);
    const tasks = state.tasks ?? {};
    const items = Object.values(tasks).map(toTaskItem).sort(sortTaskItems);
    const bashTasks = items.filter((item) => item.type === "local_bash");
    const remoteTasks = items.filter((item) => item.type === "remote_agent");
    const agentTasks = items.filter((item) => item.type === "local_agent");
    const orderedItems = useMemo(() => [...bashTasks, ...remoteTasks, ...agentTasks], [bashTasks, remoteTasks, agentTasks]);
    const selectedItem = orderedItems[selectedIndex] || null;

    useInput((input, key) => {
        if (viewState.mode !== "list") return;
        if (key.escape) {
            onDone("Background tasks dialog dismissed", { display: "system" });
            return;
        }
        if (key.upArrow) {
            setSelectedIndex((value) => Math.max(0, value - 1));
            return;
        }
        if (key.downArrow) {
            setSelectedIndex((value) => Math.min(orderedItems.length - 1, value + 1));
            return;
        }
        if (!selectedItem) return;
        if (key.return) {
            setViewState({ mode: "detail", itemId: selectedItem.id });
            return;
        }
        if (input === "k") {
            if (selectedItem.type === "local_bash" && selectedItem.status === "running") {
                void killShell(selectedItem.id, toolUseContext, setAppState);
            } else if (selectedItem.type === "local_agent" && selectedItem.status === "running") {
                void killAgent(selectedItem.id, toolUseContext, setAppState);
            }
        }
    });

    const exitState = useCtrlExit();

    void setViewState;

    if (viewState.mode !== "list" && tasks) {
        const task = tasks[viewState.itemId];
        if (!task) return null;
        switch (task.type) {
            case "local_bash":
                return (
                    <Box>
                        <Text>Shell details not implemented.</Text>
                    </Box>
                );
            case "local_agent":
                return (
                    <Box>
                        <Text>Agent details not implemented.</Text>
                    </Box>
                );
            case "remote_agent":
                return (
                    <Box>
                        <Text>Remote session details not implemented.</Text>
                    </Box>
                );
            default:
                return null;
        }
    }

    const runningShells = bashTasks.filter((item) => item.status === "running").length;
    const runningSessions = remoteTasks.filter((item) => item.status === "running" || item.status === "pending").length;
    const runningAgents = agentTasks.filter((item) => item.status === "running").length;
    const statusItems = joinWithSeparator(
        [
            ...(runningShells > 0
                ? [
                    <Text key="shells">
                        {runningShells} {runningShells !== 1 ? "active shells" : "active shell"}
                    </Text>
                ]
                : []),
            ...(runningSessions > 0
                ? [
                    <Text key="sessions">
                        {runningSessions} {runningSessions !== 1 ? "active sessions" : "active session"}
                    </Text>
                ]
                : []),
            ...(runningAgents > 0
                ? [
                    <Text key="agents">
                        {runningAgents} {runningAgents !== 1 ? "active agents" : "active agent"}
                    </Text>
                ]
                : [])
        ],
        (value) => <Text key={`separator-${value}`}> · </Text>
    );

    const shortcuts = [
        <ShortcutHint key="upDown" shortcut="↑/↓" action="select" />,
        <ShortcutHint key="enter" shortcut="Enter" action="view" />,
        ...(selectedItem?.type === "local_bash" || selectedItem?.type === "local_agent") &&
            selectedItem.status === "running"
            ? [<ShortcutHint key="kill" shortcut="k" action="kill" />]
            : [],
        <ShortcutHint key="esc" shortcut="Esc" action="close" />
    ];

    return (
        <Box width="100%" flexDirection="column">
            <Box
                borderStyle="round"
                borderColor="background"
                flexDirection="column"
                marginTop={1}
                paddingLeft={1}
                paddingRight={1}
                width="100%"
            >
                <Text color="background" bold>
                    Background tasks
                </Text>
                <Text dimColor>{statusItems}</Text>
                {items.length === 0 ? (
                    <Text dimColor>No tasks currently running</Text>
                ) : (
                    <Box flexDirection="column" marginTop={1}>
                        {bashTasks.length > 0 && (
                            <Box flexDirection="column">
                                {(remoteTasks.length > 0 || agentTasks.length > 0) && (
                                    <Text dimColor>
                                        <Text bold>  Bashes</Text> ({bashTasks.length})
                                    </Text>
                                )}
                                <Box flexDirection="column">
                                    {bashTasks.map((item, index) => (
                                        <TaskRow key={item.id} item={item} isSelected={index === selectedIndex} />
                                    ))}
                                </Box>
                            </Box>
                        )}
                        {remoteTasks.length > 0 && (
                            <Box flexDirection="column" marginTop={bashTasks.length > 0 ? 1 : 0}>
                                <Text dimColor>
                                    <Text bold>  Remote sessions</Text> ({remoteTasks.length})
                                </Text>
                                <Box flexDirection="column">
                                    {remoteTasks.map((item, index) => (
                                        <TaskRow
                                            key={item.id}
                                            item={item}
                                            isSelected={bashTasks.length + index === selectedIndex}
                                        />
                                    ))}
                                </Box>
                            </Box>
                        )}
                        {agentTasks.length > 0 && (
                            <Box
                                flexDirection="column"
                                marginTop={bashTasks.length > 0 || remoteTasks.length > 0 ? 1 : 0}
                            >
                                <Text dimColor>
                                    <Text bold>  Async agents</Text> ({agentTasks.length})
                                </Text>
                                <Box flexDirection="column">
                                    {agentTasks.map((item, index) => (
                                        <TaskRow
                                            key={item.id}
                                            item={item}
                                            isSelected={bashTasks.length + remoteTasks.length + index === selectedIndex}
                                        />
                                    ))}
                                </Box>
                            </Box>
                        )}
                    </Box>
                )}
            </Box>
            <Box marginLeft={2}>
                {exitState.pending ? (
                    <Text dimColor>Press {exitState.keyName} again to exit</Text>
                ) : (
                    <Text dimColor>
                        <ShortcutGroup>{shortcuts}</ShortcutGroup>
                    </Text>
                )}
            </Box>
        </Box>
    );
}

function toTaskItem(task: any): TaskItem {
    switch (task.type) {
        case "local_bash":
            return { id: task.id, type: "local_bash", label: task.command, status: task.status, task };
        case "remote_agent":
            return { id: task.id, type: "remote_agent", label: task.title, status: task.status, task };
        case "local_agent":
            return { id: task.id, type: "local_agent", label: task.description, status: task.status, task };
        default:
            return { id: task.id, type: "local_bash", label: "", status: "unknown", task };
    }
}

function sortTaskItems(a: TaskItem, b: TaskItem) {
    if (a.status === "running" && b.status !== "running") return -1;
    if (a.status !== "running" && b.status === "running") return 1;
    return (b.task?.startTime ?? 0) - (a.task?.startTime ?? 0);
}

function TaskRow({ item, isSelected }: { item: TaskItem; isSelected: boolean }) {
    return (
        <Box flexDirection="row" gap={1}>
            <Text color={isSelected ? "suggestion" : undefined}>
                {isSelected ? `${figures.pointer} ` : "  "}
                <TaskStatusView task={item.task} />
            </Text>
        </Box>
    );
}

function StaticMessageList({ items }: { items: FeedItem[] }) {
    return <>{items.map((item) => item.jsx)}</>;
}

function wrapText(value: string, maxWidth: number, truncate = false): string {
    if (!value) return "";
    if (value.length <= maxWidth) return value;
    if (!truncate) return value;
    return value.slice(0, Math.max(0, maxWidth - 1)) + "…";
}

function killShell(_id: string, _context: ToolUseContext, _setAppState: any) {
    return Promise.resolve();
}

function killAgent(_id: string, _context: ToolUseContext, _setAppState: any) {
    return Promise.resolve();
}

function truncateInputText(input: string, index: number) {
    if (input.length <= INPUT_TRUNCATE_LIMIT) return { truncatedText: input, placeholderContent: "" };
    const headLength = Math.floor(INPUT_TRUNCATE_WINDOW / 2);
    const tailLength = Math.floor(INPUT_TRUNCATE_WINDOW / 2);
    const head = input.slice(0, headLength);
    const tail = input.slice(-tailLength);
    const body = input.slice(headLength, -tailLength);
    const placeholder = buildTruncationPlaceholder(index, countLines(body));
    return { truncatedText: `${head}${placeholder}${tail}`, placeholderContent: body };
}

function buildTruncationPlaceholder(index: number, lineCount: number) {
    return `[...Truncated text #${index} +${lineCount} lines...]`;
}

function applyInputTruncation(input: string, pastedContents: Record<number, any>) {
    const indices = Object.keys(pastedContents).map(Number);
    const nextIndex = indices.length > 0 ? Math.max(...indices) + 1 : 1;
    const { truncatedText, placeholderContent } = truncateInputText(input, nextIndex);
    if (!placeholderContent) return { newInput: input, newPastedContents: pastedContents };
    return {
        newInput: truncatedText,
        newPastedContents: {
            ...pastedContents,
            [nextIndex]: {
                id: nextIndex,
                type: "text",
                content: placeholderContent
            }
        }
    };
}

function countLines(value: string): number {
    if (!value) return 0;
    return value.split("\n").length;
}

export function useLargeInputTruncation({
    input,
    pastedContents,
    onInputChange,
    setCursorOffset,
    setPastedContents
}: {
    input: string;
    pastedContents: Record<number, any>;
    onInputChange: (value: string) => void;
    setCursorOffset: (offset: number) => void;
    setPastedContents: (value: Record<number, any>) => void;
}) {
    const [hasTruncated, setHasTruncated] = useState(false);

    React.useEffect(() => {
        if (hasTruncated) return;
        if (input.length <= INPUT_TRUNCATE_LIMIT) return;
        const { newInput, newPastedContents } = applyInputTruncation(input, pastedContents);
        onInputChange(newInput);
        setCursorOffset(newInput.length);
        setPastedContents(newPastedContents);
        setHasTruncated(true);
    }, [input, hasTruncated, pastedContents, onInputChange, setPastedContents, setCursorOffset]);

    React.useEffect(() => {
        if (input === "") setHasTruncated(false);
    }, [input]);
}

import { runGitCommand } from "../../utils/git/GitUtils.js";

function formatFileFrequency(files: string[], limit = 20) {
    const counts = new Map<string, number>();
    for (const file of files) counts.set(file, (counts.get(file) || 0) + 1);
    return Array.from(counts.entries())
        .sort((a, b) => b[1] - a[1])
        .slice(0, limit)
        .map(([file]) => file); // Just return filenames
}

async function getProjectExampleFiles() {
    try {
        const cwd = process.cwd();
        // Get frequently modified files by current user
        const { stdout: userEmail } = await runGitCommand(["config", "user.email"], cwd);
        let files: string[] = [];

        if (userEmail.trim()) {
            const { stdout } = await runGitCommand([
                "log", "-n", "1000", "--pretty=format:", "--name-only", "--diff-filter=M", `--author=${userEmail.trim()}`
            ], cwd);
            files = stdout.split("\n").filter(f => f.trim());
        }

        // If not enough, get general frequently modified files
        if (files.length < 50) {
            const { stdout } = await runGitCommand([
                "log", "-n", "1000", "--pretty=format:", "--name-only", "--diff-filter=M"
            ], cwd);
            const moreFiles = stdout.split("\n").filter(f => f.trim());
            files = [...files, ...moreFiles];
        }

        // Simple frequency analysis instead of LLM for now
        const frequentFiles = formatFileFrequency(files, 10);

        // Filter for code files (simple heuristic)
        return frequentFiles.filter(f => /\.(ts|tsx|js|jsx|py|java|c|cpp|h|rs|go)$/.test(f)).slice(0, 5);
    } catch {
        // console.error(e);
        return [];
    }
}

const EXAMPLE_FILES_TTL_MS = 7 * 24 * 60 * 60 * 1000;

let promptHintCache = { exampleFiles: [] as string[], exampleFilesGeneratedAt: 0 };

function getPromptHintCache() {
    return promptHintCache;
}

function setPromptHintCache(next: typeof promptHintCache) {
    promptHintCache = next;
}

function chooseExamplePrompt(): string {
    const cache = getPromptHintCache();
    // Pick a random file from examples
    const exampleFile = cache.exampleFiles?.length
        ? cache.exampleFiles[Math.floor(Math.random() * cache.exampleFiles.length)]
        : "<filepath>";

    const options = [
        "fix lint errors",
        "fix typecheck errors",
        `how does ${exampleFile} work?`,
        `refactor ${exampleFile}`,
        "how do I log an error?",
        `edit ${exampleFile} to...`,
        `write a test for ${exampleFile}`,
        "create a util logging.py that..."
    ];
    return `Try "${options[Math.floor(Math.random() * options.length)]}"`;
}

async function refreshExampleFiles() {
    const cache = getPromptHintCache();
    const now = Date.now();

    // Check TTL
    if (now - cache.exampleFilesGeneratedAt > EXAMPLE_FILES_TTL_MS) {
        cache.exampleFiles = [];
    }

    if (!cache.exampleFiles?.length) {
        getProjectExampleFiles().then((files) => {
            if (files.length) {
                setPromptHintCache({
                    exampleFiles: files,
                    exampleFilesGeneratedAt: Date.now()
                });
            }
        });
    }
}

export function getPromptHint() {
    refreshExampleFiles();
    return chooseExamplePrompt();
}

const PROMPT_HINT_LIMIT = 3;

export function PromptHintView({ input, submitCount }: { input: string; submitCount: number }) {
    const [state] = useAppState();
    const { queuedCommands, promptSuggestionEnabled } = state;
    return useMemo(() => {
        if (input !== "") return;
        if (queuedCommands.length > 0 && (getSettings().queuedCommandUpHintCount || 0) < PROMPT_HINT_LIMIT) {
            return "Press up to edit queued messages";
        }
        if (submitCount < 1 && promptSuggestionEnabled) return getPromptHint();
    }, [input, queuedCommands, submitCount, promptSuggestionEnabled]);
}

function getSettings() {
    return { queuedCommandUpHintCount: 0 };
}

export function MessageTimestamp({ message, isTranscriptMode }: { message: MessageItem; isTranscriptMode: boolean }) {
    void isTranscriptMode;
    return <Text dimColor>{message?.timestamp || ""}</Text>;
}

export function MessageModelLabel({ message, isTranscriptMode }: { message: MessageItem; isTranscriptMode: boolean }) {
    void isTranscriptMode;
    return <Text dimColor>{message?.model || ""}</Text>;
}

const _unusedAnimations = TerminalFocusContext;
void _unusedAnimations;
