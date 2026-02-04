/**
 * File: src/services/tasks/TaskManager.ts
 */

import { Task, TaskStatus, TaskStore, TaskMessageQueue, TaskType, TaskResult } from './TaskTypes.js';
import { InMemoryTaskStore } from './InMemoryTaskStore.js';
import { InMemoryTaskMessageQueue } from './InMemoryTaskMessageQueue.js';
import { randomUUID } from 'node:crypto';

export class TaskManager {
    private store: TaskStore;
    private messageQueue: TaskMessageQueue;

    constructor(store?: TaskStore, messageQueue?: TaskMessageQueue) {
        this.store = store || new InMemoryTaskStore();
        this.messageQueue = messageQueue || new InMemoryTaskMessageQueue();
    }

    async createTask(type: TaskType, description: string, shellCommand?: any): Promise<Task> {
        const id = randomUUID();
        const task: Task = {
            id,
            type,
            status: 'pending',
            description,
            shellCommand,
            startTime: Date.now()
        };

        if (this.store instanceof InMemoryTaskStore) {
            this.store.addTask(task);
        } else {
            // This would normally be handled by the store itself if it had an add method
            // or we'd use a different interface. For simplicity, we assume InMemory for now.
        }

        return task;
    }

    async getTask(taskId: string): Promise<Task | undefined> {
        return this.store.getTask(taskId);
    }

    async listTasks(): Promise<Task[]> {
        const result = await this.store.listTasks();
        return result.tasks;
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
