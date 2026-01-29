/**
 * File: src/components/TaskList.tsx
 * Role: Display current tasks/TODOs.
 */

import React from 'react';
import { Box, Text } from 'ink';

export interface TaskListProps {
    tasks: string[];
}

export const TaskList: React.FC<TaskListProps> = ({ tasks }) => {
    if (tasks.length === 0) {
        return null;
    }

    return (
        <Box flexDirection="column" borderStyle="single" borderColor="gray" paddingX={1} marginBottom={1}>
            <Text bold underline>Tasks</Text>
            {tasks.map((task, i) => (
                <Text key={i}>- {task}</Text>
            ))}
        </Box>
    );
};
