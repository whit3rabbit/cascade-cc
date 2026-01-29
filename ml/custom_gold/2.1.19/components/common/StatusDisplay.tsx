/**
 * File: src/components/common/StatusDisplay.tsx
 * Role: Ink components for displaying task and session status.
 */

import React from 'react';
import { Text } from 'ink';

export interface Task {
    status: 'pending' | 'in_progress' | 'completed' | 'failed';
    [key: string]: any;
}

export interface SessionData {
    status: 'pending' | 'in_progress' | 'completed' | 'failed' | string;
    todoList: Task[];
}

/**
 * Checks if a task is completed.
 */
function isTaskCompleted(task: Task): boolean {
    return task.status === 'completed';
}

/**
 * Renders a compact status indicator for a session or task group.
 */
export const StatusIndicator: React.FC<{ sessionData: SessionData }> = ({ sessionData }) => {
    const { status, todoList } = sessionData;

    if (status === "completed") {
        return <Text bold color="green" dimColor>done</Text>;
    }

    if (status === "failed") {
        return <Text bold color="red" dimColor>error</Text>;
    }

    if (!todoList || todoList.length === 0) {
        return <Text dimColor>{status}…</Text>;
    }

    const completedCount = todoList.filter(isTaskCompleted).length;
    const totalCount = todoList.length;

    return (
        <Text dimColor>
            {completedCount}/{totalCount}
        </Text>
    );
};
