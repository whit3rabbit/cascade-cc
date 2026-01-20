// Logic from chunk_508.ts (Agent/Session Views and Security)

import React, { useCallback, useMemo, useState, useSyncExternalStore } from "react";
import { Box, useInput } from "ink";
import { Text } from "../../vendor/inkText.js";
import { Shortcut } from "../shared/Shortcut.js";
import { MessageViewAdapter } from "./MessageAdapter.js";
import { useCtrlExit } from "../../hooks/useCtrlExit.js";
import { useAppState, type AppState } from "../../contexts/AppStateContext.js";
import { useTheme } from "../../services/terminal/themeManager.js";
import {
    formatCompactNumber,
    formatDuration
} from "../../utils/shared/lodashLikeRuntimeAndEnv.js";
import { normalizeMessages } from "../../services/terminal/MessageFactory.js";
import { resumeSession } from "../../services/teleport/TeleportService.js";
import { TodoList } from "../../services/telemetry/CliActivityTracker.js";
import { figures } from "../../vendor/terminalFigures.js";
import { jsonParse } from "../../utils/shared/jsonParser.js";
import { estimateTokens } from "../../utils/shared/tokenUtils.js";
import {
    getConfig as getCachedConfig,
    getProjectConfig,
    updateProjectConfig
} from "../../services/terminal/ConfigService.js";
import { join as joinPath } from "path";
import { readFile as readFileAsync } from "fs/promises";

type ToolDefinition = {
    name: string;
    inputSchema: { safeParse: (input: any) => { success: boolean; data: any } };
    userFacingName: (input: any) => string | null | undefined;
    renderToolUseMessage?: (input: any, options: { theme: any; verbose: boolean }) => any;
    renderGroupedToolUse?: boolean;
};

type ToolUseContext = {
    options: {
        tools: ToolDefinition[];
        commands: any[];
        verbose: boolean;
    };
};

type MessageContent = {
    type: string;
    [key: string]: any;
};

type AssistantMessage = {
    type: "assistant";
    message: {
        id: string;
        content: MessageContent[];
    };
    uuid?: string;
    timestamp?: number;
};

type UserMessage = {
    type: "user";
    message: {
        content: MessageContent[];
    };
};

type SessionMessage = AssistantMessage | UserMessage | { type: string;[key: string]: any };

type RemoteSession = {
    id: string;
    title: string;
    status: string;
    startTime: number;
    log: SessionMessage[];
    progress?: { percentage?: number; percent?: number };
};

type AgentProgressActivity = {
    toolName: string;
    input: any;
};

type AgentSession = {
    agentId: string;
    status: string;
    startTime: number;
    prompt: string;
    selectedAgent?: { agentType?: string };
    description?: string;
    result?: { totalTokens?: number; totalToolUseCount?: number };
    progress?: {
        tokenCount?: number;
        toolUseCount?: number;
        recentActivities?: AgentProgressActivity[];
    };
    error?: string;
};

type VulnerabilityDetection = {
    detected: boolean;
    package: string | null;
    packageName: string | null;
    version: string | null;
    packageManager: string | null;
    lockFilePath: string | null;
};

type ShortcutGroupProps = {
    children: React.ReactNode;
};

const ShortcutGroup: React.FC<ShortcutGroupProps> = ({ children }) => {
    const items = React.Children.toArray(children).filter(Boolean);
    return (
        <>
            {items.map((child, index) => (
                <React.Fragment key={index}>
                    {child}
                    {index < items.length - 1 ? " / " : ""}
                </React.Fragment>
            ))}
        </>
    );
};

const serializeMessages = (messages: SessionMessage[]) => messages;

const getToolDefinitions = (appState: AppState, toolUseContext?: ToolUseContext) => {
    if (toolUseContext?.options?.tools) return toolUseContext.options.tools;
    return appState?.mcp?.tools ?? [];
};

const formatElapsedTime = (startTimeMs: number) => {
    const elapsedSeconds = Math.floor((Date.now() - startTimeMs) / 1000);
    const hours = Math.floor(elapsedSeconds / 3600);
    const minutes = Math.floor((elapsedSeconds - hours * 3600) / 60);
    const seconds = elapsedSeconds - hours * 3600 - minutes * 60;

    return `${hours > 0 ? `${hours}h ` : ""}${minutes > 0 || hours > 0 ? `${minutes}m ` : ""}${seconds}s`;
};

export function RemoteSessionProgress({ session }: { session: RemoteSession }) {
    const percentage = session?.progress?.percentage ?? session?.progress?.percent;
    if (percentage === undefined || percentage === null) {
        return <Text dimColor>{session?.status ?? ""}</Text>;
    }

    return <TerminalProgressBar state={session?.status} percentage={percentage} />;
}

export function RemoteSessionView({
    session,
    toolUseContext,
    onDone,
    onBack
}: {
    session: RemoteSession;
    toolUseContext: ToolUseContext;
    onDone: (message: string, meta?: any) => void;
    onBack?: () => void;
}) {
    const [isTeleporting, setIsTeleporting] = useState(false);
    const [teleportError, setTeleportError] = useState<string | null>(null);

    useInput((input, key) => {
        if (key.escape || key.return || input === " ") {
            onDone("Remote session details dismissed", { display: "system" });
            return;
        }
        if (key.leftArrow && onBack) {
            onBack();
            return;
        }
        if (input === "t" && !isTeleporting) {
            void teleportToSession();
        }
    });

    const exitState = useCtrlExit();

    const teleportToSession = async () => {
        setIsTeleporting(true);
        setTeleportError(null);
        try {
            await resumeSession(session.id);
        } catch (error) {
            setTeleportError(error instanceof Error ? error.message : String(error));
            setIsTeleporting(false);
        }
    };

    const recentMessages = useMemo(() => {
        return normalizeMessages(serializeMessages(session.log.slice(-3))).filter(
            (message: SessionMessage) => message.type !== "progress"
        );
    }, [session]);

    const truncatedTitle =
        session.title.length > 50 ? `${session.title.substring(0, 47)}...` : session.title;
    const statusLabel = session.status === "pending" ? "starting" : session.status;

    return (
        <Box width="100%" flexDirection="column">
            <Box width="100%">
                <Box
                    borderStyle="round"
                    borderColor="background"
                    flexDirection="column"
                    marginTop={1}
                    paddingLeft={1}
                    paddingRight={1}
                    width="100%"
                >
                    <Box>
                        <Text color="background" bold>
                            Remote session details
                        </Text>
                    </Box>
                    <Box flexDirection="column" marginTop={1}>
                        <Text>
                            <Text bold>Status</Text>: {" "}
                            {statusLabel === "running" || statusLabel === "starting" ? (
                                <Text color="background">{statusLabel}</Text>
                            ) : statusLabel === "completed" ? (
                                <Text color="success">{statusLabel}</Text>
                            ) : (
                                <Text color="error">{statusLabel}</Text>
                            )}
                        </Text>
                        <Text>
                            <Text bold>Runtime</Text>: {formatElapsedTime(session.startTime)}
                        </Text>
                        <Text wrap="truncate-end">
                            <Text bold>Title</Text>: {truncatedTitle}
                        </Text>
                        <Text>
                            <Text bold>Progress</Text>: {" "}
                            <RemoteSessionProgress session={session} />
                        </Text>
                        <Text>
                            <Text bold>Session URL</Text>: {" "}
                            <Text dimColor>https://claude.ai/code/{session.id}</Text>
                        </Text>
                    </Box>
                    {session.log.length > 0 && (
                        <Box flexDirection="column" marginTop={1}>
                            <Text>
                                <Text bold>Recent messages</Text>:
                            </Text>
                            <Box flexDirection="column" height={10} overflowY="hidden">
                                {recentMessages.map((message: SessionMessage, index: number) => (
                                    <MessageViewAdapter
                                        key={index}
                                        message={message}
                                        messages={recentMessages}
                                        addMargin={index > 0}
                                        tools={toolUseContext.options.tools}
                                        commands={toolUseContext.options.commands}
                                        verbose={toolUseContext.options.verbose}
                                        erroredToolUseIDs={new Set()}
                                        inProgressToolUseIDs={new Set()}
                                        resolvedToolUseIDs={new Set()}
                                        progressMessagesForMessage={[]}
                                        shouldAnimate={false}
                                        shouldShowDot={false}
                                        style="condensed"
                                        isTranscriptMode={false}
                                        isStatic
                                    />
                                ))}
                            </Box>
                            <Box marginTop={1}>
                                <Text dimColor italic>
                                    Showing last {Math.min(3, session.log.length)} of {" "}
                                    {session.log.length} messages
                                </Text>
                            </Box>
                        </Box>
                    )}
                    {teleportError && (
                        <Box marginTop={1}>
                            <Text color="error">Teleport failed: {teleportError}</Text>
                        </Box>
                    )}
                    {isTeleporting && (
                        <Box marginTop={1}>
                            <Text color="background">Teleporting to session...</Text>
                        </Box>
                    )}
                </Box>
            </Box>
            <Box marginLeft={2}>
                {exitState.pending ? (
                    <Text dimColor>
                        Press {exitState.keyName} again to exit
                    </Text>
                ) : (
                    <Text dimColor>
                        <ShortcutGroup>
                            {onBack && <Shortcut shortcut="<-" action="go back" />}
                            <Shortcut shortcut="Esc/Enter/Space" action="close" />
                            {!isTeleporting && <Shortcut shortcut="t" action="teleport" />}
                        </ShortcutGroup>
                    </Text>
                )}
            </Box>
        </Box>
    );
}

export function useDuration(startTimeMs: number, isActive: boolean, intervalMs = 1000) {
    const getSnapshot = useCallback(() => formatDuration(Date.now() - startTimeMs), [startTimeMs]);
    const subscribe = useCallback(
        (onChange: () => void) => {
            if (!isActive) return () => { };
            const interval = setInterval(onChange, intervalMs);
            return () => clearInterval(interval);
        },
        [isActive, intervalMs]
    );

    return useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
}

export function getStatusIcon(status: string) {
    switch (status) {
        case "running":
        case "pending":
            return figures.pointer;
        case "completed":
            return figures.tick;
        case "failed":
        case "killed":
            return figures.cross;
        default:
            return figures.bullet;
    }
}

export function getStatusColor(status: string) {
    switch (status) {
        case "running":
        case "pending":
            return "background";
        case "completed":
            return "success";
        case "failed":
        case "killed":
            return "error";
        default:
            return "background";
    }
}

export function getToolLabel(toolUse: AgentProgressActivity, tools: ToolDefinition[], theme: any) {
    const tool = tools.find((candidate) => candidate.name === toolUse.toolName);
    if (!tool) return toolUse.toolName;

    try {
        const parsedInput = tool.inputSchema.safeParse(toolUse.input);
        const input = parsedInput.success ? parsedInput.data : {};
        const userFacingName = tool.userFacingName(input);
        if (!userFacingName) return toolUse.toolName;

        const renderedMessage = tool.renderToolUseMessage?.(input, {
            theme,
            verbose: false
        });
        if (renderedMessage) {
            return (
                <Text>
                    {userFacingName}({renderedMessage})
                </Text>
            );
        }

        return userFacingName;
    } catch {
        return toolUse.toolName;
    }
}

export function AsyncAgentView({
    agent,
    onDone,
    onKillAgent,
    onBack
}: {
    agent: AgentSession;
    onDone: () => void;
    onKillAgent?: () => void;
    onBack?: () => void;
}) {
    const [appState] = useAppState();
    const agentTodos = appState.todos?.[agent.agentId] ?? [];
    const completedTodos = agentTodos.filter((item: any) => item.status === "completed").length;
    const [theme] = useTheme();
    const tools = useMemo(() => getToolDefinitions(appState), [appState]);
    const duration = useDuration(agent.startTime, agent.status === "running");

    useInput((input, key) => {
        if (key.escape || key.return || input === " ") onDone();
        else if (key.leftArrow && onBack) onBack();
        else if (input === "k" && agent.status === "running" && onKillAgent) onKillAgent();
    });

    const exitState = useCtrlExit();
    const promptPreview =
        agent.prompt.length > 300 ? `${agent.prompt.substring(0, 297)}...` : agent.prompt;
    const totalTokens = agent.result?.totalTokens ?? agent.progress?.tokenCount;
    const totalToolUses = agent.result?.totalToolUseCount ?? agent.progress?.toolUseCount;

    return (
        <Box width="100%" flexDirection="column">
            <Box width="100%">
                <Box
                    borderStyle="round"
                    borderColor="background"
                    flexDirection="column"
                    marginTop={1}
                    paddingLeft={1}
                    paddingRight={1}
                    width="100%"
                >
                    <Box>
                        <Text color="background" bold>
                            {agent.selectedAgent?.agentType ?? "agent"} &gt; {" "}
                            {agent.description || "Async agent"}
                        </Text>
                    </Box>
                    <Box>
                        {agent.status !== "running" && (
                            <Text color={getStatusColor(agent.status)}>
                                {getStatusIcon(agent.status)}{" "}
                                {agent.status === "completed"
                                    ? "Completed"
                                    : agent.status === "failed"
                                        ? "Failed"
                                        : "Killed"}{" "}
                                ·{" "}
                            </Text>
                        )}
                        <Text dimColor>
                            {duration}
                            {totalTokens !== undefined && totalTokens > 0 && (
                                <>
                                    {" "}· {formatCompactNumber(totalTokens)} tokens
                                </>
                            )}
                            {totalToolUses !== undefined && totalToolUses > 0 && (
                                <>
                                    {" "}· {totalToolUses} tools
                                </>
                            )}
                        </Text>
                    </Box>
                    <Box flexDirection="column">
                        {agent.status === "running" &&
                            agent.progress?.recentActivities &&
                            agent.progress.recentActivities.length > 0 && (
                                <Box flexDirection="column" marginTop={1}>
                                    <Text bold dimColor>
                                        Progress
                                    </Text>
                                    {agent.progress.recentActivities.map((activity: AgentProgressActivity, index: number) => {
                                        const activities = agent.progress?.recentActivities || [];
                                        return (
                                            <Text
                                                key={index}
                                                dimColor={index < activities.length - 1}
                                                wrap="truncate-end"
                                            >
                                                {index === activities.length - 1
                                                    ? "> "
                                                    : "  "}
                                                {getToolLabel(activity, tools, theme)}
                                            </Text>
                                        );
                                    })}
                                </Box>
                            )}
                        {agentTodos.length > 0 && (
                            <Box flexDirection="column" marginTop={1}>
                                <Text bold dimColor>
                                    Tasks ({completedTodos}/{agentTodos.length})
                                </Text>
                                <TodoList todos={agentTodos} />
                            </Box>
                        )}
                        <Box flexDirection="column" marginTop={1}>
                            <Text bold dimColor>
                                Prompt
                            </Text>
                            <Text wrap="wrap">{promptPreview}</Text>
                        </Box>
                        {agent.status === "failed" && agent.error && (
                            <Box flexDirection="column" marginTop={1}>
                                <Text bold color="error">
                                    Error
                                </Text>
                                <Text color="error" wrap="wrap">
                                    {agent.error}
                                </Text>
                            </Box>
                        )}
                    </Box>
                </Box>
            </Box>
            <Box marginLeft={2}>
                {exitState.pending ? (
                    <Text dimColor>
                        Press {exitState.keyName} again to exit
                    </Text>
                ) : (
                    <Text dimColor>
                        <ShortcutGroup>
                            {onBack && <Shortcut shortcut="<-" action="go back" />}
                            <Shortcut shortcut="Esc/Enter/Space" action="close" />
                            {agent.status === "running" && onKillAgent && (
                                <Shortcut shortcut="k" action="kill" />
                            )}
                        </ShortcutGroup>
                    </Text>
                )}
            </Box>
        </Box>
    );
}

export function TerminalProgressBar({
    state,
    percentage
}: {
    state: string;
    percentage: number;
}) {
    if (!getCachedConfig().terminalProgressBarEnabled) return null;
    const clamped = Math.max(0, Math.min(percentage ?? 0, 100));
    const ratio = clamped > 1 ? clamped / 100 : clamped;
    const width = 16;
    const filled = Math.floor(ratio * width);
    const bar = `${"#".repeat(filled)}${"-".repeat(Math.max(width - filled, 0))}`;

    return <Text dimColor>[{bar}] {state}</Text>;
}

export function getToolUseDescriptor(message: SessionMessage) {
    if (message.type === "assistant" && message.message.content[0]?.type === "tool_use") {
        const toolUse = message.message.content[0];
        return {
            messageId: message.message.id,
            toolUseId: toolUse.id,
            toolName: toolUse.name
        };
    }

    return null;
}

export function groupMessagesByTool(
    messages: SessionMessage[],
    tools: ToolDefinition[],
    skipGrouping = false
) {
    if (skipGrouping) return { messages };

    const groupableTools = new Set(
        tools.filter((tool) => tool.renderGroupedToolUse).map((tool) => tool.name)
    );
    const toolUseGroups = new Map<string, any[]>();

    for (const message of messages) {
        const toolUse = getToolUseDescriptor(message);
        if (toolUse && groupableTools.has(toolUse.toolName)) {
            const groupKey = `${toolUse.messageId}:${toolUse.toolName}`;
            const currentGroup = toolUseGroups.get(groupKey) ?? [];
            currentGroup.push(message);
            toolUseGroups.set(groupKey, currentGroup);
        }
    }

    const groupedToolUses = new Map<string, any[]>();
    const groupedToolUseIds = new Set<string>();
    for (const [groupKey, groupMessages] of toolUseGroups) {
        if (groupMessages.length >= 2) {
            groupedToolUses.set(groupKey, groupMessages);
            for (const groupMessage of groupMessages) {
                const toolUse = getToolUseDescriptor(groupMessage);
                if (toolUse) groupedToolUseIds.add(toolUse.toolUseId);
            }
        }
    }

    const resultsByToolUseId = new Map<string, any>();
    for (const message of messages) {
        if (message.type === "user") {
            for (const content of message.message.content) {
                if (content.type === "tool_result" && groupedToolUseIds.has(content.tool_use_id)) {
                    resultsByToolUseId.set(content.tool_use_id, message);
                }
            }
        }
    }

    const groupedMessages: SessionMessage[] = [];
    const includedGroups = new Set<string>();

    for (const message of messages) {
        const toolUse = getToolUseDescriptor(message);
        if (toolUse) {
            const groupKey = `${toolUse.messageId}:${toolUse.toolName}`;
            const groupMessages = groupedToolUses.get(groupKey);
            if (groupMessages) {
                if (!includedGroups.has(groupKey)) {
                    includedGroups.add(groupKey);
                    const displayMessage = groupMessages[0];
                    const results: any[] = [];
                    for (const groupedMessage of groupMessages) {
                        const toolUseId = groupedMessage.message.content[0].id;
                        const resultMessage = resultsByToolUseId.get(toolUseId);
                        if (resultMessage) results.push(resultMessage);
                    }

                    groupedMessages.push({
                        type: "grouped_tool_use",
                        toolName: toolUse.toolName,
                        messages: groupMessages,
                        results,
                        displayMessage,
                        uuid: `grouped-${displayMessage.uuid}`,
                        timestamp: displayMessage.timestamp,
                        messageId: toolUse.messageId
                    });
                }
                continue;
            }
        }

        if (message.type === "user") {
            const toolResults = message.message.content.filter(
                (content: any) => content.type === "tool_result"
            );
            if (toolResults.length > 0) {
                if (toolResults.every((result: any) => groupedToolUseIds.has(result.tool_use_id))) {
                    continue;
                }
            }
        }

        groupedMessages.push(message);
    }

    return { messages: groupedMessages };
}

export function countCustomAgentTokens(appState: AppState) {
    if (!appState) return 0;
    return appState.agentDefinitions.activeAgents
        .filter((agent: any) => agent.source !== "built-in")
        .reduce((sum: number, agent: any) => {
            const descriptor = `${agent.agentType}: ${agent.whenToUse}`;
            return sum + estimateTokens(descriptor);
        }, 0);
}

const VULNERABILITY_CACHE_TTL_MS = 15000;

export function isNextJsVulnerable(version: string) {
    const match = version.match(/^(\d+)\.(\d+)\.(\d+)(?:-canary\.(\d+))?/);
    if (!match?.[1] || !match[2] || !match[3]) return false;

    const major = parseInt(match[1], 10);
    const minor = parseInt(match[2], 10);
    const patch = parseInt(match[3], 10);
    const canary = match[4] ? parseInt(match[4], 10) : null;

    if (major <= 13) return false;
    if (major === 14) {
        if (canary !== null && minor === 3 && patch === 0) return canary >= 77;
        return false;
    }
    if (major === 15 && canary !== null) {
        if (minor === 6 && patch === 0) return canary < 58;
        return true;
    }
    if (major === 16 && canary !== null) {
        if (minor === 1 && patch === 0) return canary < 12;
        return minor === 0;
    }
    if (major >= 17) return false;

    const minorKey = `${major}.${minor}`;
    const maxPatch = NEXTJS_PATCH_VULN_MAX_PATCH[minorKey];
    if (maxPatch === undefined) {
        const sameMajor = Object.keys(NEXTJS_PATCH_VULN_MAX_PATCH)
            .filter((key) => key.startsWith(`${major}.`))
            .map((key) => parseInt(key.split(".")[1], 10));
        const maxMinor = Math.max(...sameMajor, 0);
        return minor <= maxMinor;
    }

    return patch < maxPatch;
}

export async function detectLockfileType(): Promise<VulnerabilityDetection> {
    const projectRoot = process.cwd();
    const packageLockPath = joinPath(projectRoot, "package-lock.json");

    try {
        const content = await readFileAsync(packageLockPath, "utf-8");
        const parsed = jsonParse(content);
        if (parsed) {
            const result = parseNpmLockfile(parsed, packageLockPath);
            if (result) return result;
        }
    } catch { }

    const yarnLockPath = joinPath(projectRoot, "yarn.lock");
    try {
        const content = await readFileAsync(yarnLockPath, "utf-8");
        const result = parseYarnLockfile(content, yarnLockPath);
        if (result) return result;
    } catch { }

    const pnpmLockPath = joinPath(projectRoot, "pnpm-lock.yaml");
    try {
        const content = await readFileAsync(pnpmLockPath, "utf-8");
        const result = parsePnpmLockfile(content, pnpmLockPath);
        if (result) return result;
    } catch { }

    const bunLockPath = joinPath(projectRoot, "bun.lock");
    try {
        const content = await readFileAsync(bunLockPath, "utf-8");
        const parsed = jsonParse(content);
        if (parsed) {
            const result = parseBunLockfile(parsed, bunLockPath);
            if (result) return result;
        }
    } catch { }

    return {
        detected: false,
        package: null,
        packageName: null,
        version: null,
        packageManager: null,
        lockFilePath: null
    };
}

function parseNpmLockfile(lockfile: any, lockFilePath: string): VulnerabilityDetection | null {
    const packageVersion = lockfile.packages?.["node_modules/next"]?.version;
    const dependencyVersion = lockfile.dependencies?.next?.version;
    const version = packageVersion || dependencyVersion;

    if (version) {
        if (isNextJsVulnerable(version)) {
            return {
                detected: true,
                package: "next",
                packageName: "next",
                version,
                packageManager: "npm",
                lockFilePath
            };
        }
        return null;
    }

    for (const packageName of REACT_SERVER_DOM_PACKAGES) {
        const packageVersion = lockfile.packages?.[`node_modules/${packageName}`]?.version;
        if (packageVersion && REACT_SERVER_DOM_VULNERABLE_VERSIONS.includes(packageVersion)) {
            return {
                detected: true,
                package: "react-server-dom",
                packageName,
                version: packageVersion,
                packageManager: "npm",
                lockFilePath
            };
        }
        const dependencyVersion = lockfile.dependencies?.[packageName]?.version;
        if (dependencyVersion && REACT_SERVER_DOM_VULNERABLE_VERSIONS.includes(dependencyVersion)) {
            return {
                detected: true,
                package: "react-server-dom",
                packageName,
                version: dependencyVersion,
                packageManager: "npm",
                lockFilePath
            };
        }
    }

    return null;
}

function parseYarnLockfile(content: string, lockFilePath: string): VulnerabilityDetection | null {
    const nextVersion = extractYarnVersion(content, "next");
    if (nextVersion) {
        if (isNextJsVulnerable(nextVersion)) {
            return {
                detected: true,
                package: "next",
                packageName: "next",
                version: nextVersion,
                packageManager: "yarn",
                lockFilePath
            };
        }
        return null;
    }

    for (const packageName of REACT_SERVER_DOM_PACKAGES) {
        const packageVersion = extractYarnVersion(content, packageName);
        if (packageVersion && REACT_SERVER_DOM_VULNERABLE_VERSIONS.includes(packageVersion)) {
            return {
                detected: true,
                package: "react-server-dom",
                packageName,
                version: packageVersion,
                packageManager: "yarn",
                lockFilePath
            };
        }
    }

    return null;
}

function parsePnpmLockfile(content: string, lockFilePath: string): VulnerabilityDetection | null {
    const nextVersion = extractPnpmVersion(content, "next");
    if (nextVersion) {
        if (isNextJsVulnerable(nextVersion)) {
            return {
                detected: true,
                package: "next",
                packageName: "next",
                version: nextVersion,
                packageManager: "pnpm",
                lockFilePath
            };
        }
        return null;
    }

    for (const packageName of REACT_SERVER_DOM_PACKAGES) {
        const packageVersion = extractPnpmVersion(content, packageName);
        if (packageVersion && REACT_SERVER_DOM_VULNERABLE_VERSIONS.includes(packageVersion)) {
            return {
                detected: true,
                package: "react-server-dom",
                packageName,
                version: packageVersion,
                packageManager: "pnpm",
                lockFilePath
            };
        }
    }

    return null;
}

function parseBunLockfile(lockfile: any, lockFilePath: string): VulnerabilityDetection | null {
    if (!lockfile.packages) return null;

    if ("next" in lockfile.packages) {
        const nextEntry = lockfile.packages.next;
        if (Array.isArray(nextEntry) && nextEntry[0]) {
            const match = nextEntry[0].match(/^next@(.+)$/);
            if (match?.[1]) {
                if (isNextJsVulnerable(match[1])) {
                    return {
                        detected: true,
                        package: "next",
                        packageName: "next",
                        version: match[1],
                        packageManager: "bun",
                        lockFilePath
                    };
                }
                return null;
            }
        }
    }

    for (const packageName of REACT_SERVER_DOM_PACKAGES) {
        if (packageName in lockfile.packages) {
            const packageEntry = lockfile.packages[packageName];
            if (Array.isArray(packageEntry) && packageEntry[0]) {
                const match = packageEntry[0].match(new RegExp(`^${packageName}@(.+)$`));
                if (match?.[1] && REACT_SERVER_DOM_VULNERABLE_VERSIONS.includes(match[1])) {
                    return {
                        detected: true,
                        package: "react-server-dom",
                        packageName,
                        version: match[1],
                        packageManager: "bun",
                        lockFilePath
                    };
                }
            }
        }
    }

    return null;
}

function extractYarnVersion(content: string, packageName: string) {
    const regex = new RegExp(`^"?${packageName}@[^:]+:\\s*\\n\\s+version\\s+\"([^\"]+)\"`, "m");
    return content.match(regex)?.[1] ?? null;
}

function extractPnpmVersion(content: string, packageName: string) {
    const regex = new RegExp(
        `^\\s+${packageName}:\\s*\\n\\s+specifier:[^\\n]*\\n\\s+version:\\s*([\\d.]+(?:-[\\w.]+)?)`,
        "m"
    );
    const match = content.match(regex);
    if (match?.[1]) return match[1];

    const fallback = new RegExp(`['\"]?${packageName}@([\\d.]+(?:-[\\w.]+)?)['\"]?:`);
    return content.match(fallback)?.[1] ?? null;
}

export function getPackageUpdateCommand(packageManager: string, packageName: string) {
    switch (packageManager) {
        case "npm":
            return `npm update ${packageName}`;
        case "yarn":
            return `yarn upgrade ${packageName}`;
        case "pnpm":
            return `pnpm update ${packageName}`;
        case "bun":
            return `bun update ${packageName}`;
        default:
            return "";
    }
}

export function getCachedVulnerabilityResult(): VulnerabilityDetection | null {
    const config = getProjectConfig(process.cwd());
    const cached = config.reactVulnerabilityCache;
    if (!cached) return null;

    return {
        detected: cached.detected,
        package: cached.package,
        packageName: cached.packageName,
        version: cached.version,
        packageManager: cached.packageManager,
        lockFilePath: null
    };
}

export async function scanForPackageVulnerabilities(): Promise<VulnerabilityDetection> {
    const detection = await detectLockfileType();
    updateProjectConfig(process.cwd(), (config) => ({
        ...config,
        reactVulnerabilityCache: {
            detected: detection.detected,
            package: detection.package,
            packageName: detection.packageName,
            version: detection.version,
            packageManager: detection.packageManager
        }
    }));

    return detection;
}

const REACT_SERVER_DOM_VULNERABLE_VERSIONS = ["19.0.0", "19.1.0", "19.1.1", "19.2.0"];
const REACT_SERVER_DOM_PACKAGES = [
    "react-server-dom-webpack",
    "react-server-dom-parcel",
    "react-server-dom-turbopack"
];
const NEXTJS_PATCH_VULN_MAX_PATCH: Record<string, number> = {
    "15.0": 5,
    "15.1": 9,
    "15.2": 6,
    "15.3": 6,
    "15.4": 8,
    "15.5": 7,
    "16.0": 7
};

export function getActiveVulnerabilityNotices(state: any) {
    return vulnerabilityNoticeDefinitions.filter((notice) => notice.isActive(state));
}

let warningIcon: any;
let warningTitle: any;
let warningDescription: any;
let warningAction: any;
let warningDetail: any;
let warningFooter: any;
let warningMetadata: any;
let warningStatus: any;
let warningSeverity: any;
let warningTimestamp: any;
const MAX_VULNERABILITY_NOTICES = 3;
const REACT_VULNERABILITY_NOTICE_ID = "tengu_react_vulnerability_warning";
let reactVulnerabilityNotice: any;
let vulnerabilityNoticeDefinitions: Array<{ isActive: (state: any) => boolean }> = [];
