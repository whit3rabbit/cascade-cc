
// Logic from chunk_532.ts (Tool Registry & Permissions)

import React from 'react';
import { Box, Text } from 'ink';

// --- Tool Registry (iV1) ---
export const ALL_TOOLS = [
    "Bash", "WriteFile", "ReadURL", "Subagent",
    "LSP", "MCPSearch", "TaskCreate", "TaskUpdate", "TaskList"
];

export function getAvailableTools() {
    return ALL_TOOLS;
}

// --- Permission Initialization (h69) ---
export function initializeToolPermissionContext(options: any) {
    const { mode = "default", allowedTools = [], disallowedTools = [] } = options;

    return {
        mode,
        alwaysAllowRules: { tools: allowedTools },
        alwaysDenyRules: { tools: disallowedTools },
        isBypassPermissionsModeAvailable: mode === "bypassPermissions"
    };
}

// --- Task Update Tool (q69) ---
export const TaskUpdateTool = {
    name: "TaskUpdate",
    description: "Updates an existing task.",
    async call(input: any) {
        const { taskId, status, addComment } = input;
        console.log(`Updating task ${taskId} (status: ${status})`);
        return { data: { success: true, taskId, updatedFields: Object.keys(input) } };
    }
};

// --- Task List Tool View (P69) ---
export function TaskListView({ tasks }: any) {
    if (!tasks || tasks.length === 0) return <Text dimColor>No tasks found</Text>;
    return (
        <Box flexDirection="column">
            <Text dimColor>{tasks.length} tasks recorded</Text>
            {tasks.map((t: any) => (
                <Text key={t.id}>#{t.id} [{t.status}] {t.subject}</Text>
            ))}
        </Box>
    );
}
