import React from "react";
import { Box, Text } from "ink";
import { updateAppState, AppState } from "../../contexts/AppStateContext.js";
import {
    generateTaskId,
    notifyBackgroundTask,
    InputQueueManager
} from "../input/InputQueueManager.js";
import {
    initTaskOutput,
    appendTaskOutput,
    getTaskOutputPath
} from "../persistence/persistenceUtils.js";
import { log, logError } from "../logger/loggerService.js";

// Common task initialization (fm)
function createBaseTask(id: string, type: string, description?: string) {
    return {
        id,
        type,
        description,
        startTime: Date.now(),
        outputFile: getTaskOutputPath(id)
    };
}

// Update task in AppState (UW)
function updateTaskInState(taskId: string, setAppState: (updater: (s: AppState) => AppState) => void, updater: (task: any) => any) {
    setAppState((state) => ({
        ...state,
        tasks: {
            ...state.tasks,
            [taskId]: updater(state.tasks[taskId])
        }
    }));
}

// Add task to AppState (gm)
function addTaskToState(task: any, setAppState: (updater: (s: AppState) => AppState) => void) {
    setAppState((state) => ({
        ...state,
        tasks: {
            ...state.tasks,
            [task.id]: task
        }
    }));
}

// Unregister cleanup helper (B3)
function createUnregisterCleanup(taskId: string, setAppState: (updater: (s: AppState) => AppState) => void, onKill: (taskId: string, setAppState: any) => void) {
    let cleaned = false;
    return () => {
        if (cleaned) return;
        cleaned = true;
        onKill(taskId, setAppState);
    };
}

// Kill Bash Task (AJ2)
function killBashTask(taskId: string, setAppState: (updater: (s: AppState) => AppState) => void) {
    updateTaskInState(taskId, setAppState, (task) => {
        if (!task || task.status !== "running") return task;
        if (task.shellCommand?.kill) task.shellCommand.kill();
        return {
            ...task,
            status: "killed",
            shellCommand: null,
            unregisterCleanup: undefined,
            endTime: Date.now()
        };
    });
}

// Kill Agent Task (lKA)
function killAgentTask(taskId: string, setAppState: (updater: (s: AppState) => AppState) => void) {
    updateTaskInState(taskId, setAppState, (task) => {
        if (!task || task.status !== "running") return task;
        task.abortController?.abort();
        task.unregisterCleanup?.();
        return {
            ...task,
            status: "killed",
            endTime: Date.now()
        };
    });
}

export interface BashTask {
    id: string;
    type: "local_bash";
    status: "pending" | "running" | "completed" | "failed" | "killed";
    description?: string;
    command: string;
    startTime: number;
    endTime?: number;
    outputFile: string;
    stdoutLineCount: number;
    stderrLineCount: number;
    lastReportedStdoutLines: number;
    lastReportedStderrLines: number;
    shellCommand: any;
    unregisterCleanup: () => void;
    result?: {
        code: number;
        interrupted: boolean;
    };
}

export const LocalBashTask = {
    name: "LocalBashTask",
    type: "local_bash",

    async spawn(taskDef: { command: string, description?: string, shellCommand: any }, context: { setAppState: any }) {
        const { command, description, shellCommand } = taskDef;
        const { setAppState } = context;
        const taskId = generateTaskId("local_bash");

        initTaskOutput(taskId);

        const unregisterCleanup = createUnregisterCleanup(taskId, setAppState, killBashTask);

        const task: BashTask = {
            ...createBaseTask(taskId, "local_bash", description),
            type: "local_bash",
            status: "running",
            command,
            shellCommand,
            unregisterCleanup,
            stdoutLineCount: 0,
            stderrLineCount: 0,
            lastReportedStdoutLines: 0,
            lastReportedStderrLines: 0
        };

        addTaskToState(task, setAppState);

        const backgroundHandle = shellCommand.background(taskId);
        if (!backgroundHandle) {
            updateTaskInState(taskId, setAppState, (t) => ({
                ...t,
                status: "failed",
                result: { code: 1, interrupted: false },
                endTime: Date.now()
            }));
            notifyBackgroundTask(taskId, description || command, "failed", 1, setAppState);
            return { taskId };
        }

        backgroundHandle.stdoutStream.on("data", (data: Buffer) => {
            const str = data.toString();
            appendTaskOutput(taskId, str);
            const lines = str.split("\n").filter(l => l.length > 0).length;
            updateTaskInState(taskId, setAppState, (t) => ({
                ...t,
                stdoutLineCount: t.stdoutLineCount + lines
            }));
        });

        backgroundHandle.stderrStream.on("data", (data: Buffer) => {
            const str = data.toString();
            appendTaskOutput(taskId, `[stderr] ${str}`);
            const lines = str.split("\n").filter(l => l.length > 0).length;
            updateTaskInState(taskId, setAppState, (t) => ({
                ...t,
                stderrLineCount: t.stderrLineCount + lines
            }));
        });

        shellCommand.result.then((res: { code: number, interrupted: boolean }) => {
            let wasKilled = false;
            updateTaskInState(taskId, setAppState, (t) => {
                if (t.status === "killed") {
                    wasKilled = true;
                    return t;
                }
                return {
                    ...t,
                    status: res.code === 0 ? "completed" : "failed",
                    result: res,
                    shellCommand: null,
                    unregisterCleanup: undefined,
                    endTime: Date.now()
                };
            });

            const status = wasKilled ? "killed" : (res.code === 0 ? "completed" : "failed");
            notifyBackgroundTask(taskId, description || command, status, res.code, setAppState);
        });

        return {
            taskId,
            cleanup: unregisterCleanup
        };
    },

    async kill(taskId: string, context: { setAppState: any }) {
        killBashTask(taskId, context.setAppState);
    },

    renderStatus(task: BashTask) {
        if (!task) return null;
        const { status, command } = task;
        const color = status === "running" ? "yellow" : status === "completed" ? "green" : status === "failed" ? "red" : "gray";
        return (
            <Box>
                <Text color={color}>[{status}] {command}</Text>
            </Box>
        );
    },

    renderOutput(output: string) {
        return (
            <Box>
                <Text>{output}</Text>
            </Box>
        );
    },

    getProgressMessage(task: BashTask) {
        if (!task) return null;
        const newStdout = task.stdoutLineCount - task.lastReportedStdoutLines;
        const newStderr = task.stderrLineCount - task.lastReportedStderrLines;
        if (newStdout === 0 && newStderr === 0) return null;

        const parts = [];
        if (newStdout > 0) parts.push(`${newStdout} line${newStdout > 1 ? "s" : ""} of stdout`);
        if (newStderr > 0) parts.push(`${newStderr} line${newStderr > 1 ? "s" : ""} of stderr`);

        return `Background bash ${task.id} has new output: ${parts.join(", ")}. Read ${task.outputFile} to see output.`;
    }
};

export interface AgentTask {
    id: string;
    type: "local_agent";
    status: "pending" | "running" | "completed" | "failed" | "killed";
    description?: string;
    agentId: string;
    prompt: string;
    selectedAgent: any;
    agentType: string;
    abortController: AbortController;
    outputFile: string;
    retrieved: boolean;
    lastReportedToolCount: number;
    lastReportedTokenCount: number;
    unregisterCleanup: () => void;
    progress?: {
        toolUseCount: number;
        tokenCount: number;
        lastActivity?: { toolName: string, input: any };
    };
    result?: any;
    endTime?: number;
    error?: string;
}

export const LocalAgentTask = {
    name: "LocalAgentTask",
    type: "local_agent",

    async spawn(taskDef: { prompt: string, description?: string, agentType?: string, model?: string, selectedAgent: any, agentId?: string }, context: { setAppState: any }) {
        const { prompt, description, agentType, model, selectedAgent, agentId: providedId } = taskDef;
        const { setAppState } = context;
        const taskId = providedId || generateTaskId("local_agent");

        initTaskOutput(taskId);

        const abortController = new AbortController();
        const unregisterCleanup = createUnregisterCleanup(taskId, setAppState, killAgentTask);

        const task: AgentTask = {
            ...createBaseTask(taskId, "local_agent", description),
            type: "local_agent",
            status: "running",
            agentId: taskId,
            prompt,
            selectedAgent,
            agentType: agentType || selectedAgent.agentType || "general-purpose",
            abortController,
            retrieved: false,
            lastReportedToolCount: 0,
            lastReportedTokenCount: 0,
            unregisterCleanup
        };

        addTaskToState(task, setAppState);

        return {
            taskId,
            cleanup: () => {
                unregisterCleanup();
                abortController.abort();
            }
        };
    },

    async kill(taskId: string, context: { setAppState: any }) {
        killAgentTask(taskId, context.setAppState);
    },

    renderStatus(task: AgentTask) {
        if (!task) return null;
        const { status, description, progress } = task;
        const color = status === "running" ? "yellow" : status === "completed" ? "green" : status === "failed" ? "red" : "gray";
        const progressStr = progress ? ` (${progress.toolUseCount} tools, ${progress.tokenCount} tokens)` : "";
        return (
            <Box>
                <Text color={color}>[{status}] {description}{progressStr}</Text>
            </Box>
        );
    },

    renderOutput(output: string) {
        return (
            <Box>
                <Text>{output}</Text>
            </Box>
        );
    },

    getProgressMessage(task: AgentTask) {
        if (!task || !task.progress) return null;
        const newTools = task.progress.toolUseCount - task.lastReportedToolCount;
        const newTokens = task.progress.tokenCount - task.lastReportedTokenCount;
        if (newTools === 0 && newTokens === 0) return null;

        const parts = [];
        if (newTools > 0) parts.push(`${newTools} new tool${newTools > 1 ? "s" : ""} used`);
        if (newTokens > 0) parts.push(`${newTokens} new tokens`);

        return `Agent ${task.id} progress: ${parts.join(", ")}. Read ${task.outputFile} to see full output.`;
    }
};

// Agent task completed (CJ0)
export function onAgentTaskCompleted(result: any, setAppState: any) {
    const agentId = result.agentId;
    updateTaskInState(agentId, setAppState, (t) => {
        if (!t || t.status !== "running") return t;
        t.unregisterCleanup?.();

        // Append result to output file
        if (result.content && result.content.length > 0) {
            const text = result.content.filter((c: any) => c.type === "text").map((c: any) => c.text).join("\n");
            appendTaskOutput(agentId, `\n--- RESULT ---\n${text}\n`);
        }

        return {
            ...t,
            status: "completed",
            result,
            endTime: Date.now()
        };
    });
}

// Agent task failed ($J0)
export function onAgentTaskFailed(agentId: string, error: string, setAppState: any) {
    updateTaskInState(agentId, setAppState, (t) => {
        if (!t || t.status !== "running") return t;
        t.unregisterCleanup?.();
        appendTaskOutput(agentId, `\n--- ERROR ---\n${error}\n`);
        return {
            ...t,
            status: "failed",
            error,
            endTime: Date.now()
        };
    });
}

// Notify agent status (p2A)
export function notifyAgentStatus(agentId: string, agentName: string, status: "completed" | "failed" | "killed", error: string | undefined, setAppState: any) {
    const summaryMsg = status === "completed" ? `Agent "${agentName}" completed.` : status === "failed" ? `Agent "${agentName}" failed: ${error || "Unknown error"}` : `Agent "${agentName}" was stopped.`;
    const notification = `<agent-notification>
<agent-id>${agentId}</agent-id>
<output-file>${getTaskOutputPath(agentId)}</output-file>
<status>${status}</status>
<summary>${summaryMsg}</summary>
Read the output file to retrieve the full result.
</agent-notification>`;

    InputQueueManager.enqueue({ value: notification, mode: "agent-notification" }, setAppState);
    updateTaskInState(agentId, setAppState, (t) => ({ ...t, notified: true }));
}

// Update task progress (zJ0)
export function updateTaskProgress(taskId: string, progress: any, setAppState: any) {
    updateTaskInState(taskId, setAppState, (t) => {
        if (!t || t.status !== "running") return t;
        if (progress.lastActivity) {
            appendTaskOutput(taskId, `[Tool: ${progress.lastActivity.toolName}] ${JSON.stringify(progress.lastActivity.input)}\n`);
        }
        return {
            ...t,
            progress
        };
    });
}

// Placeholder for hook execution logic (Vo, UJ0, etc)
async function executeHooks(hookName: string, context: any): Promise<any[]> {
    // This will be implemented when plugin system is ready
    return [];
}

// Logic for KL (Session Hooks)
export async function runSessionHooks(hookName: string, context: any) {
    const messages: any[] = [];
    const contexts: any[] = [];

    // try {
    //     await loadPluginHooks();
    // } catch (err) {
    //     logError("plugins", err, `Failed to load plugin hooks during ${hookName}`);
    // }

    const hookResults = await executeHooks(hookName, context);
    for (const res of hookResults) {
        if (res.message) messages.push(res.message);
        if (res.additionalContexts) contexts.push(...res.additionalContexts);
    }

    if (contexts.length > 0) {
        const item = {
            type: "hook_additional_context",
            content: contexts,
            hookName,
            toolUseID: hookName,
            hookEvent: hookName
        };
        // Enqueue additional context block if needed
        // This is simplified from Q4 logic
        messages.push(JSON.stringify(item));
    }

    return messages;
}
