import { randomUUID } from "node:crypto";
import { log, logError } from "../logger/loggerService.js";
import { getSessionId } from "../session/globalState.js";
import { logQueueOperationToSession } from "../session/SessionHistory.js";

// Queue operations (HSA, N$, sY2, tY2, eY2, YG1)

export interface QueueItem {
    value: any;
    mode: string;
}

export interface QueueState {
    queuedCommands: QueueItem[];
}

export type QueueUpdater = (updater: (state: QueueState) => QueueState) => void;
export type QueueGetter = () => Promise<QueueState>;

export class InputQueueManager {
    static logQueueOperation(operation: string, content?: any) {
        const sessionId = getSessionId();
        const event = {
            type: "queue-operation",
            operation,
            timestamp: new Date().toISOString(),
            sessionId,
            ...(content !== undefined && { content })
        };

        // Centralized debug log
        log("input")(`Queue Operation: ${operation}`, { sessionId, content });

        // Session history persistence (W12)
        logQueueOperationToSession(event).catch(err => {
            logError("input", err, "Failed to persist queue operation");
        });
    }

    static enqueue(item: QueueItem, updateState: QueueUpdater) {
        updateState((state) => ({
            ...state,
            queuedCommands: [...state.queuedCommands, item]
        }));
        this.logQueueOperation("enqueue", typeof item.value === "string" ? item.value : undefined);
    }

    static async dequeue(getState: QueueGetter, updateState: QueueUpdater): Promise<QueueItem | undefined> {
        const state = await getState();
        if (state.queuedCommands.length === 0) return undefined;

        const [first, ...rest] = state.queuedCommands;
        updateState((s) => ({ ...s, queuedCommands: rest }));
        this.logQueueOperation("dequeue");
        return first;
    }

    // ... popAll, popInteractive logic (YG1)
    static async popInteractive(currentInput: string, currentCursor: number, getState: QueueGetter, updateState: QueueUpdater) {
        const state = await getState();
        if (state.queuedCommands.length === 0) return undefined;

        const editable = state.queuedCommands.filter(item => item.mode !== "agent-notification" && item.mode !== "bash-notification");
        const nonEditable = state.queuedCommands.filter(item => item.mode === "agent-notification" || item.mode === "bash-notification");

        if (editable.length === 0) return undefined;

        const values = editable.map(item => item.value);
        const text = [...values, currentInput].filter(Boolean).join("\n");
        const cursorOffset = values.join("\n").length + 1 + currentCursor; // Fixed cursor offset logic

        // Notify pop
        editable.forEach(item => this.logQueueOperation("popAll", typeof item.value === "string" ? item.value : undefined));

        updateState(s => ({ ...s, queuedCommands: nonEditable }));

        return { text, cursorOffset };
    }
}

// Task ID Generation (JG1)
const ID_PREFIXES = {
    local_bash: "b",
    local_agent: "a",
    remote_agent: "r"
} as const;

export function generateTaskId(type: keyof typeof ID_PREFIXES): string {
    const prefix = ID_PREFIXES[type];
    const uuid = randomUUID().replace(/-/g, "").substring(0, 6);
    return `${prefix}${uuid}`;
}

// Background Task Notification (FJ0)
export function notifyBackgroundTask(taskId: string, command: string, status: "completed" | "failed" | "killed", exitCode: number | undefined, updateState: QueueUpdater) {
    let summary = `Background command "${command}" `;
    if (status === "completed") {
        summary += `completed${exitCode !== undefined ? ` (exit code ${exitCode})` : ""}`;
    } else if (status === "failed") {
        summary += `failed${exitCode !== undefined ? ` with exit code ${exitCode}` : ""}`;
    } else {
        summary += "was killed";
    }

    const notification = `<bash-notification>
<shell-id>${taskId}</shell-id>
<output-file>${taskId /* Placeholder for output file path */}</output-file>
<status>${status}</status>
<summary>${summary}.</summary>
Read the output file to retrieve the output.
</bash-notification>`;

    InputQueueManager.enqueue({ value: notification, mode: "bash-notification" }, updateState);
}
