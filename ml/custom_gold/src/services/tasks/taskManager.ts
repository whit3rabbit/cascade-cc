import { join } from "node:path";
import {
    existsSync,
    mkdirSync,
    readdirSync,
    readFileSync,
    unlinkSync,
    writeFileSync
} from "node:fs";
import { z } from "zod";
import { getConfigDir } from "../../utils/shared/pathUtils.js";
import { logError } from "../logger/loggerService.js";

// Logic from chunk_228.ts (Task Management Service)

const taskStatusSchema = z.enum(["open", "resolved"]);

const commentSchema = z.object({
    author: z.string(),
    content: z.string()
});

export const taskSchema = z.object({
    id: z.string(),
    subject: z.string(),
    description: z.string(),
    owner: z.string().optional(),
    status: taskStatusSchema,
    references: z.array(z.string()),
    blocks: z.array(z.string()),
    blockedBy: z.array(z.string()),
    comments: z.array(commentSchema)
});

export type Task = z.infer<typeof taskSchema>;
export type Comment = z.infer<typeof commentSchema>;
export type TaskUpdate = Partial<Omit<Task, "id">>;

const taskCache = new Map<string, number>();
const changeListeners = new Set<() => void>();

function notifyListeners() {
    for (const listener of changeListeners) {
        try {
            listener();
        } catch { }
    }
}

export function onTaskChange(callback: () => void): () => void {
    changeListeners.add(callback);
    return () => changeListeners.delete(callback);
}

export function isDeprecated(): boolean {
    return false;
}

export function getTeamName(): string {
    return process.env.CLAUDE_CODE_TEAM_NAME || process.env.USER || "default";
}

function getTasksDir(teamName: string): string {
    return join(getConfigDir(), "tasks", teamName);
}

function getTaskFilePath(teamName: string, taskId: string): string {
    return join(getTasksDir(teamName), `${taskId}.json`);
}

function ensureTasksDir(teamName: string) {
    const dir = getTasksDir(teamName);
    if (!existsSync(dir)) {
        mkdirSync(dir, { recursive: true });
    }
}

function getMaxTaskId(teamName: string): number {
    const dir = getTasksDir(teamName);
    if (!existsSync(dir)) return 0;

    const files = readdirSync(dir);
    let max = 0;
    for (const file of files) {
        if (!file.endsWith(".json")) continue;
        const id = parseInt(file.replace(".json", ""), 10);
        if (!isNaN(id) && id > max) max = id;
    }
    return max;
}

function getNextTaskId(teamName: string): string {
    ensureTasksDir(teamName);
    let next = taskCache.get(teamName);
    if (next === undefined) {
        next = getMaxTaskId(teamName);
    }
    next++;
    taskCache.set(teamName, next);
    return String(next);
}

export function createTask(teamName: string, taskData: Omit<Task, "id">): string {
    const id = getNextTaskId(teamName);
    const task: Task = {
        id,
        ...taskData
    };
    const filePath = getTaskFilePath(teamName, id);
    writeFileSync(filePath, JSON.stringify(task, null, 2));
    notifyListeners();
    return id;
}

export function getTask(teamName: string, taskId: string): Task | null {
    const filePath = getTaskFilePath(teamName, taskId);
    if (!existsSync(filePath)) return null;

    try {
        const content = readFileSync(filePath, "utf-8");
        const result = taskSchema.safeParse(JSON.parse(content));
        if (!result.success) {
            logError("tasks", `Task ${taskId} failed schema validation: ${result.error.message}`);
            return null;
        }
        return result.data;
    } catch (err) {
        logError("tasks", `Failed to read task ${taskId}: ${err instanceof Error ? err.message : String(err)}`);
        // In original this would also call telemetry/error tracking
        return null;
    }
}

export function updateTask(teamName: string, taskId: string, updates: TaskUpdate): Task | null {
    const existing = getTask(teamName, taskId);
    if (!existing) return null;

    const updated: Task = {
        ...existing,
        ...updates,
        id: taskId
    };

    const filePath = getTaskFilePath(teamName, taskId);
    writeFileSync(filePath, JSON.stringify(updated, null, 2));
    notifyListeners();
    return updated;
}

export function listTasks(teamName: string): Task[] {
    const dir = getTasksDir(teamName);
    if (!existsSync(dir)) return [];

    const files = readdirSync(dir);
    const tasks: Task[] = [];
    for (const file of files) {
        if (!file.endsWith(".json")) continue;
        const taskId = file.replace(".json", "");
        const task = getTask(teamName, taskId);
        if (task) tasks.push(task);
    }
    return tasks;
}

export function addTaskComment(teamName: string, taskId: string, comment: Comment): Task | null {
    const task = getTask(teamName, taskId);
    if (!task) return null;

    return updateTask(teamName, taskId, {
        comments: [...task.comments, comment]
    });
}

export function setTaskDependency(teamName: string, sourceId: string, targetId: string, type: "references" | "blocks"): boolean {
    const source = getTask(teamName, sourceId);
    const target = getTask(teamName, targetId);
    if (!source || !target) return false;

    const currentDeps = source[type];
    if (!currentDeps.includes(targetId)) {
        updateTask(teamName, sourceId, {
            [type]: [...currentDeps, targetId]
        });
    }

    if (type === "references") {
        if (!target.references.includes(sourceId)) {
            updateTask(teamName, targetId, {
                references: [...target.references, sourceId]
            });
        }
    } else {
        if (!target.blockedBy.includes(sourceId)) {
            updateTask(teamName, targetId, {
                blockedBy: [...target.blockedBy, sourceId]
            });
        }
    }

    return true;
}

