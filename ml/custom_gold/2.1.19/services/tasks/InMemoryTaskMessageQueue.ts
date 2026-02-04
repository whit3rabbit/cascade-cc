/**
 * File: src/services/tasks/InMemoryTaskMessageQueue.ts
 */

import { TaskMessage, TaskMessageQueue } from './TaskTypes.js';

export class InMemoryTaskMessageQueue implements TaskMessageQueue {
    private queues: Map<string, TaskMessage[]> = new Map();

    async enqueue(taskId: string, message: TaskMessage, _sessionId?: string): Promise<void> {
        if (!this.queues.has(taskId)) {
            this.queues.set(taskId, []);
        }
        this.queues.get(taskId)!.push(message);
    }

    async dequeue(taskId: string, _sessionId?: string): Promise<TaskMessage | undefined> {
        const queue = this.queues.get(taskId);
        if (!queue || queue.length === 0) {
            return undefined;
        }
        return queue.shift();
    }
}
