/**
 * File: src/services/terminal/TaskManager.ts
 * Role: Manages life cycle and polling of background tasks (bash, agents, etc.).
 */

export interface Task {
    id: string;
    type: string;
    status: string;
    description?: string;
    result?: { code: number };
    prompt?: string;
    error?: any;
    command?: string;
}

export interface TaskSnapshot {
    tasks?: Record<string, Task>;
}

export interface NormalizedTask {
    id: string;
    type: string;
    status: string;
    description?: string;
    exitCode?: number | null;
    prompt?: string;
    error?: any;
}

/**
 * Normalizes task data for storage or display.
 */
export function normalizeTaskState(task: Task): NormalizedTask {
    const base = {
        id: task.id,
        type: task.type,
        status: task.status,
        description: task.description
    };

    switch (task.type) {
        case "local_bash":
            return {
                ...base,
                exitCode: task.result?.code ?? null
            };
        case "local_agent":
            return {
                ...base,
                prompt: task.prompt,
                error: task.error
            };
        case "remote_agent":
            return {
                ...base,
                prompt: task.command
            };
        default:
            return base;
    }
}

/**
 * Polls for task completion.
 */
export async function waitForTask(
    taskId: string,
    getTaskFn: () => Promise<TaskSnapshot>,
    timeoutMs: number = 30000,
    signal: AbortSignal | null = null
): Promise<Task | null> {
    const start = Date.now();

    while (Date.now() - start < timeoutMs) {
        if (signal?.aborted) throw new Error("AbortError");

        const snapshot = await getTaskFn();
        const task = snapshot.tasks?.[taskId];

        if (!task) return null;
        if (task.status !== "running" && task.status !== "pending") return task;

        await new Promise(r => setTimeout(r, 100));
    }

    const finalSnapshot = await getTaskFn();
    return finalSnapshot.tasks?.[taskId] || null;
}
