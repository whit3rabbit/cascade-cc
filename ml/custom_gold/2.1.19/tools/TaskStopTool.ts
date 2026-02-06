
import { taskManager } from '../services/terminal/TaskManager.js';

export interface TaskStopInput {
    task_id: string;
    shell_id?: string;
}

export const TaskStopTool = {
    name: "TaskStop",
    description: "Stop a background task/shell.",
    async call(input: TaskStopInput) {
        const id = input.task_id || input.shell_id;
        if (!id) {
            return {
                is_error: true,
                content: "task_id is required"
            };
        }

        try {
            // Check if task exists
            const tasks = taskManager.getTasks();
            const task = tasks.find(t => t.id === id);

            if (!task) {
                return {
                    is_error: true,
                    content: `Task ${id} not found.`
                };
            }

            // Logic to kill the task. 
            // In a real implementation, TaskManager should expose a kill method that eventually kills the process.
            // We now use killTask which sends SIGKILL to the PID if available.
            taskManager.killTask(id);

            return {
                is_error: false,
                content: `Task ${id} stopped.`
            };
        } catch (error: any) {
            return {
                is_error: true,
                content: `Failed to stop task: ${error.message}`
            };
        }
    }
};
