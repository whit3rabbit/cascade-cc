
import { getGlobalState } from "../session/globalState.js";

export function getTodoList(agentId: string = "default") {
    // In a real implementation, this might come from a more complex state management
    return (getGlobalState() as any).todos?.[agentId] ?? [];
}

export function getTodoReminders(agentId: string = "default") {
    const todos = getTodoList(agentId);
    if (todos.length === 0) return [];

    return todos.filter((t: any) => t.status === "pending" || t.status === "in_progress");
}
