/**
 * File: src/services/tasks/TaskManager.ts
 */

import { Task, TaskStatus, TaskStore, TaskMessageQueue, TaskType } from './TaskTypes.js';
import { PersistentTaskStore } from './PersistentTaskStore.js';
import { InMemoryTaskMessageQueue } from './InMemoryTaskMessageQueue.js';
import { randomUUID } from 'node:crypto';

function isDependencySatisfied(status: TaskStatus | undefined): boolean {
    return status === 'completed';
}

function computeBlockedBy(task: Task, tasksById: Map<string, Task>): string[] {
    const dependencies = task.dependsOn ?? [];
    if (dependencies.length === 0) return [];

    const blockedBy: string[] = [];
    for (const depId of dependencies) {
        const dependency = tasksById.get(depId);
        if (!isDependencySatisfied(dependency?.status)) {
            blockedBy.push(depId);
        }
    }
    return blockedBy;
}

export class TaskManager {
    private store: TaskStore;
    private messageQueue: TaskMessageQueue;

    constructor(store?: TaskStore, messageQueue?: TaskMessageQueue) {
        this.store = store || new PersistentTaskStore();
        this.messageQueue = messageQueue || new InMemoryTaskMessageQueue();
    }

    async createTask(
        type: TaskType,
        description: string,
        shellCommand?: any,
        options?: { dependsOn?: string[] }
    ): Promise<Task> {
        const id = randomUUID();
        const dependsOn = options?.dependsOn ? Array.from(new Set(options.dependsOn.filter(Boolean))) : [];
        const task: Task = {
            id,
            type,
            status: 'pending',
            description,
            shellCommand,
            startTime: Date.now(),
            ...(dependsOn.length > 0 ? { dependsOn } : {})
        };

        await this.store.addTask(task);

        return task;
    }

    async getTask(taskId: string): Promise<Task | undefined> {
        const task = await this.store.getTask(taskId);
        if (!task) return undefined;
        const { tasks } = await this.store.listTasks();
        const tasksById = new Map(tasks.map(t => [t.id, t]));
        const blockedBy = computeBlockedBy(task, tasksById);
        return {
            ...task,
            ...(blockedBy.length > 0 ? { blockedBy } : { blockedBy: [] })
        };
    }

    async listTasks(): Promise<Task[]> {
        const result = await this.store.listTasks();
        const tasksById = new Map(result.tasks.map(task => [task.id, task]));
        return result.tasks.map(task => {
            const blockedBy = computeBlockedBy(task, tasksById);
            return {
                ...task,
                ...(blockedBy.length > 0 ? { blockedBy } : { blockedBy: [] })
            };
        });
    }

    async updateTaskStatus(taskId: string, status: TaskStatus, message?: string): Promise<void> {
        await this.store.updateTaskStatus(taskId, status, message);
    }

    async cancelTask(taskId: string): Promise<void> {
        const task = await this.store.getTask(taskId);
        if (!task) throw new Error(`Task not found: ${taskId}`);
        if (task.status === 'completed' || task.status === 'failed' || task.status === 'killed' || task.status === 'cancelled') {
            throw new Error(`Cannot cancel task in terminal status: ${task.status}`);
        }

        await this.store.updateTaskStatus(taskId, 'cancelled', 'Client cancelled task execution.');
        if (task.unregisterCleanup) {
            task.unregisterCleanup();
        }
    }

    async backgroundTask(taskId: string): Promise<boolean> {
        const task = await this.store.getTask(taskId);
        if (!task || task.isBackgrounded || !task.shellCommand || typeof task.shellCommand.background !== 'function') {
            return false;
        }

        const success = task.shellCommand.background(taskId);
        if (!success) {
            return false;
        }

        task.isBackgrounded = true;
        // In a real app, we'd update the store
        await this.updateTaskStatus(taskId, task.status);

        return true;
    }

    async enqueueMessage(taskId: string, message: any): Promise<void> {
        await this.messageQueue.enqueue(taskId, {
            type: 'notification',
            message,
            timestamp: Date.now()
        });
    }

    async dequeueMessage(taskId: string): Promise<any> {
        return this.messageQueue.dequeue(taskId);
    }
}

export const taskManager = new TaskManager();
