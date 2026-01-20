
// Logic from chunk_433.ts (User Activity, Todo Viewer)
import React from 'react';
import { Text, Box } from 'ink';

export class CliActivityTracker {
    private static instance: CliActivityTracker;
    private activeOperations = new Set<string>();
    private lastUserActivityTime = 0;
    private lastCLIRecordedTime = Date.now();
    private isCLIActive = false;
    private readonly USER_ACTIVITY_TIMEOUT_MS = 5000;

    static getInstance(): CliActivityTracker {
        if (!CliActivityTracker.instance) {
            CliActivityTracker.instance = new CliActivityTracker();
        }
        return CliActivityTracker.instance;
    }

    recordUserActivity() {
        if (!this.isCLIActive && this.lastUserActivityTime !== 0) {
            // Metric recording logic stub
        }
        this.lastUserActivityTime = Date.now();
    }

    startCLIActivity(opId: string) {
        if (this.activeOperations.has(opId)) this.endCLIActivity(opId);
        const wasEmpty = this.activeOperations.size === 0;
        this.activeOperations.add(opId);
        if (wasEmpty) {
            this.isCLIActive = true;
            this.lastCLIRecordedTime = Date.now();
        }
    }

    endCLIActivity(opId: string) {
        this.activeOperations.delete(opId);
        if (this.activeOperations.size === 0) {
            this.isCLIActive = false;
            // Metric recording logic stub
            this.lastCLIRecordedTime = Date.now();
        }
    }

    async trackOperation<T>(opId: string, operation: () => Promise<T>): Promise<T> {
        this.startCLIActivity(opId);
        try {
            return await operation();
        } finally {
            this.endCLIActivity(opId);
        }
    }

    getActivityStates() {
        return {
            isUserActive: (Date.now() - this.lastUserActivityTime) < this.USER_ACTIVITY_TIMEOUT_MS,
            isCLIActive: this.isCLIActive,
            activeOperationCount: this.activeOperations.size
        };
    }
}

// Logic for Kr (Todo List Viewer)
export interface TodoItem {
    status: 'in_progress' | 'completed' | 'pending';
    content: string;
}

export const TodoList: React.FC<{ todos: TodoItem[], isStandalone?: boolean }> = ({ todos, isStandalone = false }) => {
    if (todos.length === 0) return null;

    const list = todos.map((todo, index) => (
        <Box key={index}>
            <Text dimColor={todo.status === 'completed'}>
                {todo.status === 'completed' ? '[x] ' : '[ ] '}
            </Text>
            <Text
                bold={todo.status === 'in_progress'}
                dimColor={todo.status === 'completed'}
                strikethrough={todo.status === 'completed'}
            >
                {todo.content}
            </Text>
        </Box>
    ));

    if (isStandalone) {
        return (
            <Box flexDirection="column" marginTop={1} marginLeft={2}>
                <Text bold dimColor>Todos</Text>
                {list}
            </Box>
        );
    }

    return <Box flexDirection="column">{list}</Box>;
};
