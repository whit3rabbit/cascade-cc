/**
 * File: src/components/terminal/TasksDialog.tsx
 */

import React, { useState, useEffect } from 'react';
import { Box, Text, useInput } from 'ink';
import { taskManager } from '../../services/tasks/TaskManager.js';
import { Task } from '../../services/tasks/TaskTypes.js';

interface TasksDialogProps {
    onClose: () => void;
}

export const TasksDialog: React.FC<TasksDialogProps> = ({ onClose }) => {
    const [tasks, setTasks] = useState<Task[]>([]);
    const [selectedIndex, setSelectedIndex] = useState(0);

    useEffect(() => {
        const fetchTasks = async () => {
            const allTasks = await taskManager.listTasks();
            setTasks(allTasks);
        };
        fetchTasks();

        const interval = setInterval(fetchTasks, 1000);
        return () => clearInterval(interval);
    }, []);

    useInput((input, key) => {
        if (key.escape || input === 'q') {
            onClose();
        } else if (key.upArrow) {
            setSelectedIndex(Math.max(0, selectedIndex - 1));
        } else if (key.downArrow) {
            setSelectedIndex(Math.min(tasks.length - 1, selectedIndex + 1));
        } else if (input === 'c') {
            // Cancel selected task
            const selectedTask = tasks[selectedIndex];
            if (selectedTask) {
                taskManager.cancelTask(selectedTask.id).catch(console.error);
            }
        }
    });

    return (
        <Box flexDirection="column" borderStyle="round" padding={1}>
            <Text bold color="cyan">Active Tasks</Text>
            <Box flexDirection="column" marginTop={1}>
                {tasks.length === 0 ? (
                    <Text dimColor>No active tasks</Text>
                ) : (
                    tasks.map((task, index) => {
                        const blockedCount = task.blockedBy?.length ?? 0;
                        const statusLabel = blockedCount > 0 ? 'blocked' : task.status;
                        const dependencySuffix = blockedCount > 0 ? ` deps:${blockedCount}` : '';

                        return (
                            <Box key={task.id}>
                                <Text color={index === selectedIndex ? 'blue' : undefined}>
                                    {index === selectedIndex ? '> ' : '  '}
                                    [{statusLabel.padEnd(10)}] {task.type.padEnd(12)} - {task.description}{dependencySuffix}
                                </Text>
                            </Box>
                        );
                    })
                )}
            </Box>
            <Box marginTop={1}>
                <Text dimColor>↑/↓: Navigate | c: Cancel | q/Esc: Close</Text>
            </Box>
        </Box>
    );
};
