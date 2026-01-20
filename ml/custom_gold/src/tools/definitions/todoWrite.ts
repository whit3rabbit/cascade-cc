import { z } from 'zod';

const TODO_WRITER_DESCRIPTION = `Use this tool to create and manage a structured task list for your current coding session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.
It also helps the user understand the progress of the task and overall progress of their requests.

## When to Use This Tool
Use this tool proactively in these scenarios:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests todo list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
5. After receiving new instructions - Immediately capture user requirements as todos
6. When you start working on a task - Mark it as in_progress BEFORE beginning work. Ideally you should only have one todo as in_progress at a time
7. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation

## When NOT to Use This Tool

Skip using this tool when:
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no organizational benefit
3. The task can be completed in less than 3 trivial steps
4. The task is purely conversational or informational

NOTE that you should not use this tool if there is only one trivial task to do. In this case you are better off just doing the task directly.

## Task States and Management

1. **Task States**: Use these states to track progress:
   - pending: Task not yet started
   - in_progress: Currently working on (limit to ONE task at a time)
   - completed: Task finished successfully

   **IMPORTANT**: Task descriptions must have two forms:
   - content: The imperative form describing what needs to be done (e.g., "Run tests", "Build the project")
   - activeForm: The present continuous form shown during execution (e.g., "Running tests", "Building the project")

2. **Task Management**:
   - Update task status in real-time as you work
   - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
   - Exactly ONE task must be in_progress at any time (not less, not more)
   - Complete current tasks before starting new ones
   - Remove tasks that are no longer relevant from the list entirely

3. **Task Completion Requirements**:
   - ONLY mark a task as completed when you have FULLY accomplished it
   - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
   - When blocked, create a new task describing what needs to be resolved
   - Never mark a task as completed if:
     - Tests are failing
     - Implementation is partial
     - You encountered unresolved errors
     - You couldn't find necessary files or dependencies

4. **Task Breakdown**:
   - Create specific, actionable items
   - Break complex tasks into smaller, manageable steps
   - Use clear, descriptive task names
   - Always provide both forms:
     - content: "Fix authentication bug"
     - activeForm: "Fixing authentication bug"

When in doubt, use this tool. Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully.`;

const DEPRECATION_NOTICE = `**DEPRECATED**: This tool is deprecated. Prefer using the new task management tools instead:
- TaskCreate: Create new tasks
- TaskGet: Retrieve task details by ID
- TaskUpdate: Update task status, add comments, or set dependencies
- TaskList: List all tasks

The new tools support team collaboration, task dependencies, and persistent task storage across sessions.

---

`;

const taskStatusSchema = z.enum(["pending", "in_progress", "completed"]);

const todoItemSchema = z.object({
    content: z.string().min(1, "Content cannot be empty"),
    status: taskStatusSchema,
    activeForm: z.string().min(1, "Active form cannot be empty")
});

const todoListSchema = z.array(todoItemSchema);

const inputSchema = z.object({
    todos: todoListSchema.describe("The updated todo list")
});

const outputSchema = z.object({
    oldTodos: todoListSchema.describe("The todo list before the update"),
    newTodos: todoListSchema.describe("The todo list after the update")
});

export const todoWriteTool = {
    name: "TodoWrite",
    strict: true,
    input_examples: [{
        todos: [{
            content: "Implement user authentication",
            status: "in_progress",
            activeForm: "Implementing user authentication"
        }, {
            content: "Write unit tests",
            status: "pending",
            activeForm: "Writing unit tests"
        }]
    }],
    async description() {
        return "Update the todo list for the current session. To be used proactively and often to track progress and pending tasks. Make sure that at least one task is in_progress at all times. Always provide both content (imperative) and activeForm (present continuous) for each task.";
    },
    async prompt() {
        // Assuming sU() check for deprecation
        // if (isNewTaskToolsEnabled()) return DEPRECATION_NOTICE + TODO_WRITER_DESCRIPTION;
        return TODO_WRITER_DESCRIPTION;
    },
    inputSchema,
    outputSchema,
    userFacingName() {
        return "";
    },
    isEnabled() {
        return true;
    },
    isConcurrencySafe() {
        return false;
    },
    isReadOnly() {
        return false;
    },
    async checkPermissions(input: any) {
        return {
            behavior: "allow",
            updatedInput: input
        };
    },
    async call({ todos }: z.infer<typeof inputSchema>, context: any) {
        const appState = await context.getAppState();
        const agentId = context.agentId ?? "default";
        const oldTodos = appState.todos?.[agentId] ?? [];

        // Auto-clear completed todos if all are done
        const updatedTodos = todos.every(t => t.status === "completed") ? [] : todos;

        await context.setAppState((state: any) => ({
            ...state,
            todos: {
                ...(state.todos || {}),
                [agentId]: updatedTodos
            }
        }));

        return {
            data: {
                oldTodos,
                newTodos: todos
            }
        };
    }
};
