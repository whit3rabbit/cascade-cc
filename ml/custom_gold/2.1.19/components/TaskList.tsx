import React from 'react';
import { Box, Text } from 'ink';
import Spinner from 'ink-spinner';
import { Task } from '../services/terminal/TaskManager.js';

export interface TaskListProps {
    tasks: Task[];
}

export const TaskList: React.FC<TaskListProps> = ({ tasks }) => {
    if (tasks.length === 0) {
        return null;
    }

    return (
        <Box
            flexDirection="column"
            borderStyle="single"
            borderColor="gray"
            paddingX={1}
            marginBottom={1}
            width={40}
        >
            <Text bold underline>Active Tasks</Text>
            {tasks.map((task) => {
                const isRunning = task.status === 'running';
                const isFailed = task.status === 'failed';
                const isCompleted = task.status === 'completed';

                let statusColor = 'gray';
                if (isRunning) statusColor = 'blue';
                if (isFailed) statusColor = 'red';
                if (isCompleted) statusColor = 'green';

                return (
                    <Box key={task.id} flexDirection="column" marginTop={1}>
                        <Box justifyContent="space-between">
                            <Text color={statusColor}>
                                {isRunning && <Spinner type="dots" />} {task.description || task.type}
                            </Text>
                            <Text dimColor>[{task.status}]</Text>
                        </Box>
                        {task.progress !== undefined && (
                            <Box>
                                <Text dimColor>[</Text>
                                <Text color="green">{'█'.repeat(Math.floor(task.progress / 5))}</Text>
                                <Text color="gray">{'░'.repeat(20 - Math.floor(task.progress / 5))}</Text>
                                <Text dimColor>] {task.progress}%</Text>
                            </Box>
                        )}
                    </Box>
                );
            })}
        </Box>
    );
};
