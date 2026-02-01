/**
 * File: src/services/terminal/TaskManager.ts
 * Role: Manages life cycle and polling of background tasks (bash, agents, etc.).
 */

export interface Task {
    id: string;
    type: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    description?: string;
    progress?: number; // 0 to 100
    result?: { code: number };
    prompt?: string;
    error?: any;
    command?: string;
}

export interface TaskSnapshot {
    tasks?: Record<string, Task>;
}

export type TaskListener = (tasks: Task[]) => void;

class GlobalTaskManager {
    private tasks: Map<string, Task> = new Map();
    private listeners: Set<TaskListener> = new Set();

    addTask(task: Omit<Task, 'status'> & { status?: Task['status'] }): void {
        const newTask: Task = {
            status: 'pending',
            ...task
        };
        this.tasks.set(newTask.id, newTask);
        this.notify();
    }

    updateTask(id: string, updates: Partial<Task>): void {
        const task = this.tasks.get(id);
        if (task) {
            this.tasks.set(id, { ...task, ...updates });
            this.notify();
        }
    }

    removeTask(id: string): void {
        this.tasks.delete(id);
        this.notify();
    }

    cancelTask(id: string): void {
        const task = this.tasks.get(id);
        if (task && (task.status === 'pending' || task.status === 'running')) {
            this.updateTask(id, { status: 'failed', error: 'User cancelled' });
        }
    }


    getTasks(): Task[] {
        return Array.from(this.tasks.values());
    }

    subscribe(listener: TaskListener): () => void {
        this.listeners.add(listener);
        listener(this.getTasks());
        return () => this.listeners.delete(listener);
    }

    private notify(): void {
        const tasks = this.getTasks();
        for (const listener of this.listeners) {
            listener(tasks);
        }
    }
}

export const taskManager = new GlobalTaskManager();

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
