
import { randomUUID } from "node:crypto";
import { log } from "../logger/loggerService.js";
import { getGlobalState, setGlobalState } from "../session/globalState.js";
import { createUserMessage, createAssistantMessage, createMetadataMessage, normalizeMessages } from "../terminal/MessageFactory.js";

import { getProjectRoot } from "../../utils/shared/pathUtils.js";
import { getTodoReminders, getTodoList } from "../terminal/TodoService.js";
import { getSessionId } from "../session/sessionStore.js";
import { logTelemetryEvent } from "../telemetry/telemetryInit.js";
import { getActiveModel, formatNumberCompact } from "../claude/claudeUtils.js";
import { readFileAttachment } from "../attachments/AttachmentManager.js";
import { executePreCompactHooks } from "../hooks/HookExecutor.js";

const logger = log("history");

export interface Attachment {
    type: string;
    uuid: string;
    timestamp: string;
    attachment: any;
}

export const CompactionConfig = {
    TURNS_SINCE_WRITE: 20,
    TURNS_BETWEEN_REMINDERS: 5,
    MINIMUM_MESSAGE_TOKENS_TO_INIT: 10000,
    MINIMUM_TOKENS_BETWEEN_UPDATE: 5000,
    TOOL_CALLS_BETWEEN_UPDATES: 3,
    MAX_FILE_RESTORE: 10,
    MAX_FILE_RESTORE_TOKENS: 10000,
    MAX_ATTACHMENT_TOKENS: 50000
};

/**
 * Creates a standard attachment object. (Q4)
 */
export function createAttachment(payload: any): Attachment {
    return {
        attachment: payload,
        type: "attachment",
        uuid: randomUUID(),
        timestamp: new Date().toISOString()
    };
}

/**
 * Calculates turns since last todo write or reminder. (ax5)
 */
export function getTurnsSinceLastTodo(messages: any[]) {
    let lastTodoWriteIndex = -1;
    let lastTodoReminderIndex = -1;
    let turnsSinceWrite = 0;
    let turnsSinceReminder = 0;

    for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i];
        if (msg.type === "assistant") {
            const isSummary = (msg as any).isCompactSummary;
            if (isSummary) continue;

            if (lastTodoWriteIndex === -1) turnsSinceWrite++;
            if (lastTodoReminderIndex === -1) turnsSinceReminder++;

            const content = msg.message?.content;
            if (Array.isArray(content) && content.some((c: any) => c.type === "tool_use" && c.name === "TodoWrite")) {
                lastTodoWriteIndex = i;
            }
        } else if (lastTodoReminderIndex === -1 && msg.type === "attachment" && msg.attachment.type === "todo_reminder") {
            lastTodoReminderIndex = i;
        }
        if (lastTodoWriteIndex !== -1 && lastTodoReminderIndex !== -1) break;
    }

    return {
        turnsSinceLastTodoWrite: turnsSinceWrite,
        turnsSinceLastReminder: turnsSinceReminder
    };
}

/**
 * Returns todo reminders as attachments if needed. (ox5)
 */
export async function getTodoAttachmentsForHistory(messages: any[], context: any) {
    if (!messages || messages.length === 0) return [];
    const { turnsSinceLastTodoWrite, turnsSinceLastReminder } = getTurnsSinceLastTodo(messages);

    if (turnsSinceLastTodoWrite >= CompactionConfig.TURNS_SINCE_WRITE &&
        turnsSinceLastReminder >= CompactionConfig.TURNS_BETWEEN_REMINDERS) {
        const agentId = context.agentId ?? "default";
        const reminders = await getTodoReminders(agentId);
        if (reminders && reminders.length > 0) {
            return [createAttachment({
                type: "todo_reminder",
                content: reminders,
                itemCount: reminders.length
            })];
        }
    }
    return [];
}

/**
 * Calculate task progress (rx5)
 */
export function getTaskProgressStats(messages: any[]) {
    const stats = new Map<string, { total: number; resolved: number }>();
    if (!messages || messages.length === 0) return stats;

    const seenTaskIds = new Set<string>();
    for (const msg of messages) {
        if (msg.type === "assistant" && msg.message?.content) {
            for (const content of msg.message.content) {
                if (content.type === "tool_use" && content.name === "TaskWrite") {
                    const taskId = content.input?.id;
                    if (taskId && !seenTaskIds.has(taskId)) {
                        seenTaskIds.add(taskId);
                        const current = stats.get(taskId) || { total: 0, resolved: 0 };
                        current.total++;
                        if (content.input?.status === "resolved") {
                            current.resolved++;
                        }
                        stats.set(taskId, current);
                    }
                }
            }
        }
    }
    return stats;
}

/**
 * Get token usage summary (ex5 / Ay5)
 */
export function getTokenUsageAttachment(stats: any) {
    return createAttachment({
        type: "token_usage",
        total: stats.total,
        user: stats.user,
        assistant: stats.assistant,
        tools: stats.tools
    });
}

/**
 * Calculate token stats (uj2)
 */
export function calculateTokenStats(messages: any[]) {
    const stats = {
        total: 0,
        user: 0,
        assistant: 0,
        tools: 0,
        images: 0,
        attachments: 0,
        toolCalls: new Map<string, number>()
    };

    for (const msg of messages) {
        const tokens = (msg as any).usage?.total_tokens || 0;
        if (msg.type === "user") {
            stats.user += tokens;
        } else if (msg.type === "assistant") {
            stats.assistant += tokens;
        } else if (msg.type === "tool_result" || msg.type === "tool_use") {
            stats.tools += tokens;
        } else if (msg.type === "attachment") {
            stats.attachments += tokens;
        }
        stats.total += tokens;

        if (msg.type === "assistant" && msg.message?.content) {
            for (const block of msg.message.content) {
                if (block.type === "tool_use") {
                    const name = block.name;
                    stats.toolCalls.set(name, (stats.toolCalls.get(name) || 0) + 1);
                }
            }
        }
    }
    return stats;
}

/**
 * Format token stats (mj2)
 */
export function formatTokenStats(stats: any): string {
    const parts = [
        `Total: ${formatNumberCompact(stats.total)}`,
        `User: ${formatNumberCompact(stats.user)}`,
        `Assistant: ${formatNumberCompact(stats.assistant)}`
    ];
    if (stats.tools > 0) parts.push(`Tools: ${formatNumberCompact(stats.tools)}`);
    return parts.join(", ");
}

/**
 * Restore files after compaction (Xy5)
 */
export async function getFileRestorationAttachments(readFileState: any, context: any, maxFiles: number) {
    const files = Object.entries(readFileState)
        .map(([filename, data]: [string, any]) => ({ filename, ...data }))
        .filter(f => !shouldSkipFileRestoration(f.filename, context.agentId))
        .sort((a, b) => b.timestamp - a.timestamp)
        .slice(0, maxFiles);

    const attachments = await Promise.all(files.map(async (f) => {
        const attachment = await readFileAttachment(
            f.filename,
            context,
            "tengu_post_compact_file_restore_success",
            "tengu_post_compact_file_restore_error",
            "compact",
            { fileReadingLimits: { maxTokens: CompactionConfig.MAX_FILE_RESTORE_TOKENS } }
        );
        return attachment ? createAttachment(attachment) : null;
    }));

    return attachments.filter(Boolean) as Attachment[];
}

function shouldSkipFileRestoration(filename: string, agentId?: string): boolean {
    // Logic from Vy5
    return false; // placeholder
}

/**
 * Main compaction logic. (NJ1)
 */
export async function compactConversation(
    messages: any[],
    toolUseContext: any,
    triggerSource: string = "manual"
) {
    if (messages.length === 0) throw new Error("Not enough messages to compact.");

    logger.info(`Compacting conversation triggered by ${triggerSource}`);

    const stats = calculateTokenStats(messages);
    const preTokens = stats.total;

    // Call LLM for summary
    const summaryText = "Summary generated by AI...";

    const summaryMessage = createMetadataMessage({
        content: summaryText,
        isCompactSummary: true,
        isVisibleInTranscriptOnly: true
    } as any);

    const attachments: Attachment[] = [];
    const state = getGlobalState();

    // executePreCompactHooks
    const hookResult = await executePreCompactHooks(
        triggerSource,
        undefined,
        toolUseContext.abortController?.signal
    );

    // File restoration (Xy5)
    const fileAttachments = await getFileRestorationAttachments(
        toolUseContext.readFileState || {},
        toolUseContext,
        CompactionConfig.MAX_FILE_RESTORE
    );
    attachments.push(...fileAttachments);

    // Todo restoration (Iy5)
    const todos = getTodoList(toolUseContext.agentId);
    if (todos.length > 0) {
        attachments.push(createAttachment({
            type: "todo",
            content: todos,
            itemCount: todos.length,
            context: "post-compact"
        }));
    }

    // Skill restoration (Wy5)
    if (state.invokedSkills.size > 0) {
        attachments.push(createAttachment({
            type: "invoked_skills",
            skills: Array.from(state.invokedSkills.keys())
        }));
    }

    // Plan restoration (RH0)
    if (state.needsPlanModeExitAttachment) {
        attachments.push(createAttachment({
            type: "plan_exit_reminder",
            content: "The agent previously exited plan mode."
        }));
    }

    logTelemetryEvent("tengu_compact", {
        trigger: triggerSource,
        preCompactTokenCount: preTokens,
        postCompactTokenCount: 0
    });

    const boundaryMarker = createMetadataMessage({
        content: `Conversation compacted (${triggerSource})`,
        isMeta: true
    });

    return {
        compactionResult: {
            boundaryMarker,
            summaryMessages: [summaryMessage],
            attachments,
            hookResults: [],
            preCompactTokenCount: preTokens,
            postCompactTokenCount: 0, // Placeholder
            messagesToKeep: messages.slice(-2)
        }
    };
}

/**
 * Micro-compacts messages by replaces long tool results with placeholders.
 * Based on chunk_446.ts (Vd)
 */
export async function microCompactMessages(
    messages: any[],
    threshold: number = 40000,
    context?: any
): Promise<{ messages: any[] }> {
    // Simplify implementation for now: only compact very long tool results
    const result = messages.map(msg => {
        if (msg.type !== "user" || !Array.isArray(msg.message.content)) return msg;

        const newContent = msg.message.content.map((block: any) => {
            if (block.type === "tool_result" && block.content && typeof block.content === "string") {
                if (block.content.length > threshold) {
                    return {
                        ...block,
                        content: `Tool result too long (${block.content.length} chars). Use Read tool to view full content for tool_use_id: ${block.tool_use_id}`
                    };
                }
            }
            return block;
        });

        return {
            ...msg,
            message: {
                ...msg.message,
                content: newContent
            }
        };
    });

    return { messages: result };
}
