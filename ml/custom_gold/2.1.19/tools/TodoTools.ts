
import { readFile, writeFile, mkdir } from 'fs/promises';
import { join } from 'path';
import { existsSync } from 'fs';

const TODO_FILE = '.claude/todos.json';

interface Todo {
    id: string;
    subject: string;
    description: string;
    status: 'pending' | 'in_progress' | 'completed' | 'cancelled';
    created_at: number;
}

async function getTodosPath(): Promise<string> {
    const cwd = process.cwd();
    const claudeDir = join(cwd, '.claude');
    if (!existsSync(claudeDir)) {
        await mkdir(claudeDir, { recursive: true });
    }
    return join(cwd, TODO_FILE);
}

async function readTodos(): Promise<Todo[]> {
    const path = await getTodosPath();
    if (!existsSync(path)) return [];
    try {
        const content = await readFile(path, 'utf-8');
        return JSON.parse(content);
    } catch {
        return [];
    }
}

async function writeTodos(todos: Todo[]): Promise<void> {
    const path = await getTodosPath();
    await writeFile(path, JSON.stringify(todos, null, 2), 'utf-8');
}

export const TaskCreateTool = {
    name: "TaskCreate",
    description: "Create a new task/todo.",
    async call(input: { subject: string; description?: string }) {
        const todos = await readTodos();
        const newTodo: Todo = {
            id: Math.random().toString(36).substring(2, 8),
            subject: input.subject,
            description: input.description || "",
            status: "pending",
            created_at: Date.now()
        };
        todos.push(newTodo);
        await writeTodos(todos);
        return { is_error: false, content: `Task created with ID ${newTodo.id}` };
    }
};

export const TaskUpdateTool = {
    name: "TaskUpdate",
    description: "Update an existing task.",
    async call(input: { id: string; status?: string; subject?: string; description?: string }) {
        const todos = await readTodos();
        const todo = todos.find(t => t.id === input.id);
        if (!todo) {
            return { is_error: true, content: `Task ${input.id} not found.` };
        }
        if (input.status) todo.status = input.status as any;
        if (input.subject) todo.subject = input.subject;
        if (input.description) todo.description = input.description;
        await writeTodos(todos);
        return { is_error: false, content: `Task ${input.id} updated.` };
    }
};

export const TaskListTool = {
    name: "TaskList",
    description: "List tasks.",
    async call(input: { status?: string }) {
        const todos = await readTodos();
        const filtered = input.status && input.status !== 'all'
            ? todos.filter(t => t.status === input.status)
            : todos;

        if (filtered.length === 0) return { is_error: false, content: "No tasks found." };

        const content = filtered.map(t => `[${t.status}] ${t.id}: ${t.subject}`).join('\n');
        return { is_error: false, content };
    }
};
