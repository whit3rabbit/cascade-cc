/**
 * File: src/components/menus/TaskMenu.tsx
 * Role: Interactive menu for managing background tasks.
 */

import React, { useState } from 'react';
import { Box, Text, useInput } from 'ink';
import SelectInput from 'ink-select-input';
import { taskManager, Task } from '../../services/terminal/TaskManager.js';
import { useTheme } from '../../services/terminal/ThemeService.js';

interface TaskMenuProps {
    onExit: () => void;
}

export const TaskMenu: React.FC<TaskMenuProps> = ({ onExit }) => {
    const theme = useTheme();
    const [selectedTaskId, setSelectedTaskId] = useState<string | null>(null);
    const tasks = taskManager.getTasks();

    useInput((input, key) => {
        if (key.escape) {
            if (selectedTaskId) setSelectedTaskId(null);
            else onExit();
        }
    });

    const handleTaskSelect = (item: { value: string }) => {
        setSelectedTaskId(item.value);
    };

    const handleActionSelect = (item: { value: string }) => {
        if (!selectedTaskId) return;

        if (item.value === 'cancel') {
            taskManager.cancelTask(selectedTaskId);
        } else if (item.value === 'remove') {
            taskManager.removeTask(selectedTaskId);
        }
        setSelectedTaskId(null);
    };

    if (tasks.length === 0) {
        return (
            <Box flexDirection="column" padding={1} borderStyle="round" borderColor={theme.subtle}>
                <Text italic dimColor>No active tasks found.</Text>
                <Box marginTop={1}>
                    <Text>Press Esc to exit.</Text>
                </Box>
            </Box>

        );
    }

    if (selectedTaskId) {
        const selectedTask = tasks.find(t => t.id === selectedTaskId);
        return (
            <Box flexDirection="column" padding={1} borderStyle="round" borderColor={theme.claudeBlue_FOR_SYSTEM_SPINNER}>
                <Text bold color={theme.claudeBlue_FOR_SYSTEM_SPINNER}>
                    Managing Task: {selectedTask?.description || selectedTask?.id}
                </Text>
                <Box marginTop={1}>
                    <SelectInput
                        items={[
                            { label: 'Cancel Task', value: 'cancel' },
                            { label: 'Remove from List', value: 'remove' },
                            { label: 'Back', value: 'back' }
                        ]}
                        onSelect={handleActionSelect}
                    />
                </Box>
            </Box>
        );
    }

    const items = tasks.map(t => ({
        label: `[${t.status.toUpperCase()}] ${t.description || t.type} (${t.id.slice(0, 8)})`,
        value: t.id
    }));

    return (
        <Box flexDirection="column" padding={1} borderStyle="round" borderColor={theme.claudeBlue_FOR_SYSTEM_SPINNER}>
            <Text bold color={theme.claudeBlue_FOR_SYSTEM_SPINNER}>Background Tasks</Text>
            <Box marginTop={1}>
                <SelectInput
                    items={items}
                    onSelect={handleTaskSelect}
                />
            </Box>
            <Box marginTop={1}>
                <Text dimColor>Esc to exit • Enter to manage task</Text>
            </Box>
        </Box>
    );
};
