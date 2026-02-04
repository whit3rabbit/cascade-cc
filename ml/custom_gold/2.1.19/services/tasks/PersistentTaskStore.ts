/**
 * File: src/services/tasks/PersistentTaskStore.ts
 */

import { Task, TaskStatus, TaskStore } from './TaskTypes.js';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { getBaseConfigDir } from '../../utils/shared/runtimeAndEnv.js';

export class PersistentTaskStore implements TaskStore {
    private tasks: Map<string, Task> = new Map();
    private filePath: string;

    constructor(filePath?: string) {
        if (filePath) {
            this.filePath = filePath;
        } else {
            const configDir = getBaseConfigDir();
            this.filePath = join(configDir, 'tasks.json');
        }
        this.load();
    }

    private load(): void {
        try {
            if (existsSync(this.filePath)) {
                const data = readFileSync(this.filePath, 'utf8');
                const tasks = JSON.parse(data) as Task[];
                // Verify it's an array
                if (Array.isArray(tasks)) {
                    this.tasks.clear();
                    tasks.forEach(task => this.tasks.set(task.id, task));
                }
            }
        } catch (error) {
            console.error('Failed to load tasks:', error);
        }
    }

    private save(): void {
        try {
            const dir = dirname(this.filePath);
            if (!existsSync(dir)) {
                mkdirSync(dir, { recursive: true });
            }
            const tasks = Array.from(this.tasks.values());
            writeFileSync(this.filePath, JSON.stringify(tasks, null, 2), 'utf8');
        } catch (error) {
            console.error('Failed to save tasks:', error);
        }
    }

    async getTask(taskId: string, _sessionId?: string): Promise<Task | undefined> {
        return this.tasks.get(taskId);
    }

    async listTasks(cursor?: string, _sessionId?: string): Promise<{ tasks: Task[], nextCursor?: string }> {
        const allTasks = Array.from(this.tasks.values());
        return { tasks: allTasks };
    }

    async updateTaskStatus(taskId: string, status: TaskStatus, _message?: string, _sessionId?: string): Promise<void> {
        const task = this.tasks.get(taskId);
        if (task) {
            task.status = status;
            if (status === 'completed' || status === 'failed' || status === 'killed' || status === 'cancelled') {
                if (!task.endTime) {
                    task.endTime = Date.now();
                }
            }
            this.tasks.set(taskId, task);
            this.save();
        }
    }

    async getTaskResult(taskId: string, _sessionId?: string): Promise<any> {
        const task = this.tasks.get(taskId);
        return task?.result;
    }

    async addTask(task: Task): Promise<void> {
        this.tasks.set(task.id, task);
        this.save();
    }
}
