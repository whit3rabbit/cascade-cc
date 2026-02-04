/**
 * File: src/services/tasks/InMemoryTaskStore.ts
 */

import { Task, TaskStatus, TaskStore } from './TaskTypes.js';

export class InMemoryTaskStore implements TaskStore {
    private tasks: Map<string, Task> = new Map();

    async getTask(taskId: string, _sessionId?: string): Promise<Task | undefined> {
        return this.tasks.get(taskId);
    }

    async listTasks(cursor?: string, _sessionId?: string): Promise<{ tasks: Task[], nextCursor?: string }> {
        const allTasks = Array.from(this.tasks.values());
        // Simple pagination can be added if needed
        return { tasks: allTasks };
    }

    async updateTaskStatus(taskId: string, status: TaskStatus, _message?: string, _sessionId?: string): Promise<void> {
        const task = this.tasks.get(taskId);
        if (task) {
            task.status = status;
            if (status === 'completed' || status === 'failed' || status === 'killed' || status === 'cancelled') {
                task.endTime = Date.now();
            }
            this.tasks.set(taskId, task);
        }
    }

    async getTaskResult(taskId: string, _sessionId?: string): Promise<any> {
        const task = this.tasks.get(taskId);
        return task?.result;
    }

    // Additional method for internal use
    async addTask(task: Task): Promise<void> {
        this.tasks.set(task.id, task);
    }
}
