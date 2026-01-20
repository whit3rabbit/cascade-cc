
import React, { useState, useMemo, useEffect } from 'react';
import { Box, Text, useInput } from 'ink';
import { useAppState } from '../../contexts/AppStateContext.js';
import { Shortcut } from '../shared/Shortcut.js';
import { useTerminalSize } from '../../hooks/terminal/useTerminalControllerHooks.js';

// Symbols (G1)
const SYMBOLS = {
    pointer: '›',
    tick: '✔',
    cross: '✖',
    bullet: '•'
};

function formatRuntime(startTime: number) {
    const diff = Math.floor((Date.now() - startTime) / 1000);
    const hours = Math.floor(diff / 3600);
    const minutes = Math.floor((diff - hours * 3600) / 60);
    const seconds = diff - hours * 3600 - minutes * 60;
    return `${hours > 0 ? `${hours}h ` : ''}${minutes > 0 || hours > 0 ? `${minutes}m ` : ''}${seconds}s`;
}

// Mock kill functions until services are fully ready
const killBashTask = async (id: string, context: any) => {
    // In real implementation: import { BashService } from ...; await BashService.kill(id);
    console.log("Killing bash task", id);
};

const killAgentTask = async (id: string, context: any) => {
    // In real implementation: import { AgentService } from ...; await AgentService.kill(id);
    console.log("Killing agent task", id);
};

// --- Sub-components ---

function LocalBashDetail({ shell, onDone, onKillShell, onBack }: any) {
    const { columns } = useTerminalSize();
    const [outputLines, setOutputLines] = useState(0);
    const [stdout, setStdout] = useState("");

    useInput((input, key) => {
        if (key.escape || key.return || input === ' ') {
            onDone("Shell details dismissed", { display: 'system' });
        } else if (key.leftArrow && onBack) {
            onBack();
        } else if (input === 'k' && shell.status === 'running' && onKillShell) {
            onKillShell();
        }
    });

    // Mock streaming output update
    useEffect(() => {
        if (shell.status === 'running') {
            const timer = setInterval(() => {
                // Just mocking output for now since we don't have the real shell buffer subscription here yet
                setOutputLines(prev => prev + 1);
            }, 1000);
            return () => clearInterval(timer);
        }
    }, [shell.status]);

    const commandDisplay = shell.command.length > 280 ? shell.command.substring(0, 277) + '…' : shell.command;

    return (
        <Box flexDirection="column" width="100%">
            <Box borderStyle="round" borderColor="blue" flexDirection="column" marginTop={1} paddingX={1} width="100%">
                <Text bold color="blue">Shell details</Text>
                <Box flexDirection="column" marginTop={1}>
                    <Text><Text bold>Status:</Text> <Text color={shell.status === 'running' ? 'blue' : shell.status === 'completed' ? 'green' : 'red'}>{shell.status}</Text></Text>
                    <Text><Text bold>Runtime:</Text> {formatRuntime(shell.startTime)}</Text>
                    <Text><Text bold>Command:</Text> {commandDisplay}</Text>
                </Box>
                <Box flexDirection="column" marginTop={1}>
                    <Text bold>Output:</Text>
                    <Box borderStyle="round" borderDimColor paddingX={1} flexDirection="column" height={10}>
                        <Text dimColor>Output not connected in mock...</Text>
                    </Box>
                </Box>
            </Box>
            <Box marginLeft={2}>
                <Text dimColor>
                    {onBack && <Shortcut shortcut="←" action="go back" />}
                    <Shortcut shortcut="Esc/Enter/Space" action="close" />
                    {shell.status === 'running' && <Shortcut shortcut="k" action="kill" />}
                </Text>
            </Box>
        </Box>
    );
}

function LocalAgentDetail({ agent, onDone, onKillAgent, onBack }: any) {
    useInput((input, key) => {
        if (key.escape || key.return || input === ' ') {
            onDone();
        } else if (key.leftArrow && onBack) {
            onBack();
        } else if (input === 'k' && agent.status === 'running' && onKillAgent) {
            onKillAgent();
        }
    });

    return (
        <Box flexDirection="column" width="100%">
            <Box borderStyle="round" borderColor="blue" flexDirection="column" marginTop={1} paddingX={1} width="100%">
                <Text bold color="blue">{agent.selectedAgent?.agentType ?? "agent"} › {agent.description || "Async agent"}</Text>
                <Box flexDirection="column" marginTop={1}>
                    <Text><Text bold>Status:</Text> <Text color={agent.status === 'running' ? 'blue' : agent.status === 'completed' ? 'green' : 'red'}>{agent.status}</Text></Text>
                    <Text><Text bold>Runtime:</Text> {formatRuntime(agent.startTime)}</Text>
                </Box>
            </Box>
            <Box marginLeft={2}>
                <Text dimColor>
                    {onBack && <Shortcut shortcut="←" action="go back" />}
                    <Shortcut shortcut="Esc/Enter/Space" action="close" />
                    {agent.status === 'running' && <Shortcut shortcut="k" action="kill" />}
                </Text>
            </Box>
        </Box>
    );
}

function RemoteAgentDetail({ session, toolUseContext, onDone, onBack }: any) {
    useInput((input, key) => {
        if (key.escape || key.return || input === ' ') {
            onDone("Remote session details dismissed", { display: 'system' });
        } else if (key.leftArrow && onBack) {
            onBack();
        }
    });

    return (
        <Box flexDirection="column" width="100%">
            <Box borderStyle="round" borderColor="blue" flexDirection="column" marginTop={1} paddingX={1} width="100%">
                <Text bold color="blue">Remote session details</Text>
                <Text>Details for session {session.id}</Text>
            </Box>
            <Box marginLeft={2}>
                <Text dimColor>
                    {onBack && <Shortcut shortcut="←" action="go back" />}
                    <Shortcut shortcut="Esc/Enter/Space" action="close" />
                </Text>
            </Box>
        </Box>
    )
}

function TaskItem({ item, isSelected }: any) {
    const statusColor = item.status === 'running' ? 'blue' : item.status === 'completed' ? 'green' : 'red';
    return (
        <Box flexDirection="row" gap={1}>
            <Text color={isSelected ? 'blue' : undefined}>
                {isSelected ? SYMBOLS.pointer : ' '} <Text color={statusColor}>{item.label} ({item.status})</Text>
            </Text>
        </Box>
    );
}

// --- Main Component ---

export function TasksDialog({ onDone, toolUseContext }: any) {
    const [appState, setAppState] = useAppState();
    const [viewState, setViewState] = useState<{ mode: 'list' | 'detail'; itemId?: string }>({ mode: 'list' });
    const [selectedIndex, setSelectedIndex] = useState(0);

    const tasks = appState.tasks || {};

    const allTasks = useMemo(() => {
        return Object.values(tasks).map((t: any) => {
            if (t.type === 'local_bash') return { id: t.id, type: 'local_bash', label: t.command, status: t.status, task: t, startTime: t.startTime };
            if (t.type === 'remote_agent') return { id: t.id, type: 'remote_agent', label: t.title, status: t.status, task: t, startTime: t.startTime };
            if (t.type === 'local_agent') return { id: t.id, type: 'local_agent', label: t.description, status: t.status, task: t, startTime: t.startTime };
            return null;
        }).filter(Boolean).sort((a: any, b: any) => {
            if (a.status === 'running' && b.status !== 'running') return -1;
            if (a.status !== 'running' && b.status === 'running') return 1;
            return b.startTime - a.startTime;
        });
    }, [tasks]);

    const selectedTask = allTasks[selectedIndex] || null;

    useInput((input, key) => {
        if (viewState.mode !== 'list') return;

        if (key.escape) {
            onDone("Background tasks dialog dismissed", { display: 'system' });
            return;
        }
        if (key.upArrow) {
            setSelectedIndex(prev => Math.max(0, prev - 1));
            return;
        }
        if (key.downArrow) {
            setSelectedIndex(prev => Math.min(allTasks.length - 1, prev + 1));
            return;
        }
        if (!selectedTask) return;

        if (key.return) {
            setViewState({ mode: 'detail', itemId: selectedTask.id });
            return;
        }

        if (input === 'k') {
            if (selectedTask.status === 'running') {
                if (selectedTask.type === 'local_bash') killBashTask(selectedTask.id, toolUseContext);
                else if (selectedTask.type === 'local_agent') killAgentTask(selectedTask.id, toolUseContext);
            }
        }
    });

    useEffect(() => {
        // Bounds check if tasks list changes
        if (selectedIndex >= allTasks.length && allTasks.length > 0) {
            setSelectedIndex(allTasks.length - 1);
        }
    }, [allTasks.length]);

    if (viewState.mode !== 'list' && viewState.itemId) {
        const taskItem = allTasks.find((t: any) => t.id === viewState.itemId);
        if (!taskItem) {
            setViewState({ mode: 'list' });
            return null;
        }

        if (taskItem.type === 'local_bash') {
            return <LocalBashDetail shell={taskItem.task} onDone={onDone} onKillShell={() => killBashTask(taskItem.id, toolUseContext)} onBack={() => setViewState({ mode: 'list' })} />;
        }
        if (taskItem.type === 'local_agent') {
            return <LocalAgentDetail agent={taskItem.task} onDone={onDone} onKillAgent={() => killAgentTask(taskItem.id, toolUseContext)} onBack={() => setViewState({ mode: 'list' })} />;
        }
        if (taskItem.type === 'remote_agent') {
            return <RemoteAgentDetail session={taskItem.task} toolUseContext={toolUseContext} onDone={onDone} onBack={() => setViewState({ mode: 'list' })} />;
        }
    }

    const runningCount = allTasks.filter((t: any) => t.status === 'running').length;

    return (
        <Box flexDirection="column" width="100%">
            <Box borderStyle="round" borderColor="blue" flexDirection="column" marginTop={1} paddingX={1} width="100%">
                <Text bold color="blue">Background tasks</Text>
                {runningCount > 0 && <Text dimColor>{runningCount} active tasks</Text>}

                {allTasks.length === 0 ? (
                    <Text dimColor>No tasks currently running</Text>
                ) : (
                    <Box flexDirection="column" marginTop={1}>
                        {allTasks.map((item: any, index: number) => (
                            <TaskItem key={item.id} item={item} isSelected={index === selectedIndex} />
                        ))}
                    </Box>
                )}
            </Box>

            <Box marginLeft={2}>
                <Text dimColor>
                    <Shortcut shortcut="↑/↓" action="select" />
                    <Shortcut shortcut="Enter" action="view" />
                    {selectedTask?.status === 'running' && <Shortcut shortcut="k" action="kill" />}
                    <Shortcut shortcut="Esc" action="close" />
                </Text>
            </Box>
        </Box>
    );
}
