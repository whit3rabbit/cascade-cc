import React from 'react';
import { Box, Text } from 'ink';
import { z } from 'zod';
import {
    createTask,
    getTask,
    updateTask,
    listTasks,
    addTaskComment,
    setTaskDependency,
    getTeamName
} from '../../services/tasks/taskManager.js';
import { figures } from '../../vendor/terminalFigures.js';
import { log, logError } from '../../services/logger/loggerService.js';

// --- LSP Tool (fq0) ---
export const LspTool = {
    name: "LSP",
    description: "Language Server Protocol integration for code analysis.",
    inputSchema: z.object({
        operation: z.enum(["goToDefinition", "findReferences", "hover", "documentSymbol", "workspaceSymbol", "goToImplementation", "prepareCallHierarchy", "incomingCalls", "outgoingCalls"]).describe("The LSP operation to perform"),
        filePath: z.string().describe("The absolute or relative path to the file"),
        line: z.number().int().positive().describe("The line number (1-based)"),
        character: z.number().int().positive().describe("The character offset (1-based)")
    }),
    async call(input: any, context: any) {
        const { operation, filePath } = input;
        log("lsp")(`LSP Operation: ${operation} on ${filePath}`);
        // Integration with LspServerManager would go here
        return { data: { operation, result: "LSP operation successful (simulated)", filePath } };
    }
};

// --- MCP Search Tool (i49) ---
export const MCPSearchTool = {
    name: "MCPSearch",
    description: "Searches for and loads MCP tools.",
    inputSchema: z.object({
        query: z.string().describe('Query to find MCP tools. Use "select:<tool_name>" for direct selection, or keywords to search.'),
        max_results: z.number().optional().default(5).describe("Maximum number of results to return (default: 5)")
    }),
    async call(input: any, context: any) {
        const { query } = input;
        log("mcp")(`Searching for MCP tools: ${query}`);
        // MCP tool discovery logic would go here
        return { data: { matches: [], total_mcp_tools: 0, query } };
    }
};

// --- Task Management Tools (B69 / pV1 / cV1 / lV1) ---

export const TaskCreateTool = {
    name: "TaskCreate",
    description: "Create a new task in the task list",
    inputSchema: z.object({
        subject: z.string().describe("A brief title for the task"),
        description: z.string().describe("A detailed description of what needs to be done")
    }),
    async call({ subject, description }: { subject: string, description: string }) {
        const teamName = getTeamName();
        const taskId = createTask(teamName, {
            subject,
            description,
            status: "open",
            references: [],
            blocks: [],
            blockedBy: [],
            comments: []
        });
        return { data: { task: { id: taskId, subject } } };
    },
    renderToolResultMessage(result: any) {
        const { task } = result;
        return (
            <Box>
                <Text color="green">{figures.tick} </Text>
                <Text>Task </Text>
                <Text bold>#{task.id}</Text>
                <Text> created: </Text>
                <Text>{task.subject}</Text>
            </Box>
        );
    }
};

export const TaskGetTool = {
    name: "TaskGet",
    description: "Get a task by ID from the task list",
    inputSchema: z.object({
        taskId: z.string().describe("The ID of the task to retrieve")
    }),
    async call({ taskId }: { taskId: string }) {
        const teamName = getTeamName();
        const task = getTask(teamName, taskId);
        return { data: { task } };
    },
    renderToolResultMessage(result: any) {
        const { task } = result;
        if (!task) {
            return (
                <Box>
                    <Text color="red">{figures.cross} </Text>
                    <Text>Task not found</Text>
                </Box>
            );
        }
        return (
            <Box flexDirection="column">
                <Text bold>Task #{task.id}: {task.subject}</Text>
                <Text>Status: {task.status}</Text>
                <Text>Description: {task.description}</Text>
                {task.blockedBy.length > 0 && <Text color="yellow">Blocked by: {task.blockedBy.join(", ")}</Text>}
                {task.comments.length > 0 && (
                    <Box flexDirection="column">
                        <Text>Comments:</Text>
                        {task.comments.map((c: any, i: number) => (
                            <Text key={i}>  [{c.author}]: {c.content}</Text>
                        ))}
                    </Box>
                )}
            </Box>
        );
    }
};

export const TaskUpdateTool = {
    name: "TaskUpdate",
    description: "Update a task in the task list",
    inputSchema: z.object({
        taskId: z.string().describe("The ID of the task to update"),
        subject: z.string().optional().describe("New subject for the task"),
        description: z.string().optional().describe("New description for the task"),
        status: z.enum(["open", "resolved"]).optional().describe("New status for the task"),
        addComment: z.object({
            author: z.string().describe("Author of the comment"),
            content: z.string().describe("Content of the comment")
        }).optional().describe("Add a comment to the task"),
        addReferences: z.array(z.string()).optional().describe("Task IDs to add as references"),
        addBlocks: z.array(z.string()).optional().describe("Task IDs that this task blocks"),
        addBlockedBy: z.array(z.string()).optional().describe("Task IDs that block this task")
    }),
    async call(input: any) {
        const teamName = getTeamName();
        const { taskId, subject, description, status, addComment, addReferences, addBlocks, addBlockedBy } = input;
        const existing = getTask(teamName, taskId);
        if (!existing) {
            return { data: { success: false, taskId, updatedFields: [], error: "Task not found" } };
        }

        const updatedFields: string[] = [];
        const updates: any = {};
        if (subject !== undefined) { updates.subject = subject; updatedFields.push("subject"); }
        if (description !== undefined) { updates.description = description; updatedFields.push("description"); }
        if (status !== undefined) { updates.status = status; updatedFields.push("status"); }

        if (Object.keys(updates).length > 0) {
            updateTask(teamName, taskId, updates);
        }

        if (addComment) {
            addTaskComment(teamName, taskId, { content: addComment.content, author: addComment.author });
            updatedFields.push("comments");
        }

        if (addReferences) {
            for (const refId of addReferences) setTaskDependency(teamName, taskId, refId, "references");
            updatedFields.push("references");
        }

        if (addBlocks) {
            for (const blockId of addBlocks) setTaskDependency(teamName, taskId, blockId, "blocks");
            updatedFields.push("blocks");
        }

        if (addBlockedBy) {
            for (const blockerId of addBlockedBy) setTaskDependency(teamName, blockerId, taskId, "blocks");
            updatedFields.push("blockedBy");
        }

        return { data: { success: true, taskId, updatedFields, wasResolved: status === "resolved" } };
    },
    renderToolResultMessage(result: any) {
        const { success, taskId, updatedFields, error } = result;
        if (!success) {
            return (
                <Box>
                    <Text color="red">{figures.cross} </Text>
                    <Text>Task </Text>
                    <Text bold>#{taskId}</Text>
                    <Text>: {error}</Text>
                </Box>
            );
        }
        return (
            <Box>
                <Text color="green">{figures.tick} </Text>
                <Text>Task </Text>
                <Text bold>#{taskId}</Text>
                <Text> updated: </Text>
                <Text dimColor>{updatedFields.join(", ")}</Text>
            </Box>
        );
    }
};

export const TaskListTool = {
    name: "TaskList",
    description: "List all tasks in the task list",
    inputSchema: z.object({}),
    async call() {
        const teamName = getTeamName();
        const tasks = listTasks(teamName);
        const resolvedIds = new Set(tasks.filter(t => t.status === "resolved").map(t => t.id));

        return {
            data: {
                tasks: tasks.map(t => ({
                    id: t.id,
                    subject: t.subject,
                    status: t.status,
                    owner: t.owner,
                    blockedBy: t.blockedBy.filter(id => !resolvedIds.has(id))
                }))
            }
        };
    },
    renderToolResultMessage(result: any) {
        const { tasks } = result;
        if (tasks.length === 0) {
            return <Box><Text dimColor>No tasks found</Text></Box>;
        }

        const openCount = tasks.filter((t: any) => t.status === "open").length;
        const resolvedCount = tasks.filter((t: any) => t.status === "resolved").length;

        return (
            <Box flexDirection="column">
                <Box marginBottom={1}>
                    <Text dimColor>{tasks.length} task{tasks.length !== 1 ? "s" : ""} ({resolvedCount} done, {openCount} open)</Text>
                </Box>
                {tasks.map((task: any) => {
                    const isResolved = task.status === "resolved";
                    const isBlocked = task.blockedBy.length > 0;
                    const icon = isResolved ? figures.tick : figures.squareSmallFilled;
                    return (
                        <Box key={task.id}>
                            <Text color={isResolved ? "green" : isBlocked ? "yellow" : undefined}>{icon} </Text>
                            <Text dimColor>#{task.id} </Text>
                            <Text strikethrough={isResolved} dimColor={isResolved}>{task.subject}</Text>
                            {task.owner && <Text dimColor> ({task.owner})</Text>}
                            {isBlocked && (
                                <Text color="yellow"> {figures.warning} blocked by {task.blockedBy.join(", ")}</Text>
                            )}
                        </Box>
                    );
                })}
            </Box>
        );
    }
};
