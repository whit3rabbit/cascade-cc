/**
 * File: src/services/tasks/TaskTypes.ts
 */

export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed' | 'killed' | 'cancelled' | 'input_required';

export type TaskType = 'local_bash' | 'remote_agent' | 'local_agent' | 'in_process_teammate';

export interface TaskResult {
    code: number;
    interrupted?: boolean;
}

export interface Task {
    id: string;
    type: TaskType;
    status: TaskStatus;
    description: string;
    shellCommand?: any; // Should be a BashCommand object
    isBackgrounded?: boolean;
    result?: TaskResult;
    startTime: number;
    endTime?: number;
    unregisterCleanup?: () => void;
    pollInterval?: number;
    _meta?: any;
}

export interface TaskStore {
    getTask(taskId: string, sessionId?: string): Promise<Task | undefined>;
    listTasks(cursor?: string, sessionId?: string): Promise<{ tasks: Task[], nextCursor?: string }>;
    updateTaskStatus(taskId: string, status: TaskStatus, message?: string, sessionId?: string): Promise<void>;
    getTaskResult(taskId: string, sessionId?: string): Promise<any>;
    addTask(task: Task): Promise<void>;
}

export interface TaskMessage {
    type: 'request' | 'response' | 'error' | 'notification';
    message: any;
    timestamp: number;
}

export interface TaskMessageQueue {
    enqueue(taskId: string, message: TaskMessage, sessionId?: string): Promise<void>;
    dequeue(taskId: string, sessionId?: string): Promise<TaskMessage | undefined>;
}
