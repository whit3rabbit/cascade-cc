// Logic from chunk_444.ts and chunk_445.ts

import * as fs from "node:fs";
import * as path from "node:path";
import { randomUUID } from "node:crypto";
import { log } from "../logger/loggerService.js";
import { normalizeMessages, createMetadataMessage, createUserMessage } from "../terminal/MessageFactory.js";
import { wrapInReminder } from "../terminal/StreamingHandler.js";
import { resolvePath } from "../../utils/shared/pathUtils.js";
import { DiagnosticsManager } from "../diagnostics/DiagnosticsManager.js";
import { LspDiagnosticsManager } from "../lsp/LspDiagnosticsManager.js";
import { getOriginalCwd, getSessionId } from "../session/sessionStore.js";
import { logTelemetryEvent } from "../telemetry/telemetryInit.js";
import { getConnectedIdeClient } from "../mcp/McpClientManager.js";
import { getTodoReminders } from "../terminal/TodoService.js";

const logger = log("attachments");

// Hook Registry Logic
interface HookRegistryEntry {
    processId: string;
    hookName: string;
    hookEvent: string;
    toolName?: string;
    command: string;
    startTime: number;
    timeout: number;
    stdout: string;
    stderr: string;
    responseAttachmentSent: boolean;
    shellCommand?: any;
}

const hookRegistry = new Map<string, HookRegistryEntry>();

export function registerHook(params: {
    processId: string;
    asyncResponse: { asyncTimeout?: number };
    hookName: string;
    hookEvent: string;
    command: string;
    shellCommand: any;
    toolName?: string;
}) {
    const timeout = params.asyncResponse.asyncTimeout || 15000;
    logger.info(`Hooks: Registering async hook ${params.processId} (${params.hookName}) with timeout ${timeout}ms`);
    hookRegistry.set(params.processId, {
        processId: params.processId,
        hookName: params.hookName,
        hookEvent: params.hookEvent,
        toolName: params.toolName,
        command: params.command,
        startTime: Date.now(),
        timeout,
        stdout: "",
        stderr: "",
        responseAttachmentSent: false,
        shellCommand: params.shellCommand
    });
}

export function addHookStdout(processId: string, output: string) {
    const hook = hookRegistry.get(processId);
    if (hook) {
        hook.stdout += output;
    }
}

export function addHookStderr(processId: string, output: string) {
    const hook = hookRegistry.get(processId);
    if (hook) {
        hook.stderr += output;
    }
}

export async function checkForNewResponses() {
    const responses: any[] = [];
    const hooksToRemove: string[] = [];

    for (const hook of Array.from(hookRegistry.values())) {
        if (!hook.shellCommand) {
            hooksToRemove.push(hook.processId);
            continue;
        }

        if (hook.shellCommand.status === "killed") {
            hooksToRemove.push(hook.processId);
            continue;
        }

        if (hook.shellCommand.status !== "completed") continue;

        if (hook.responseAttachmentSent || !hook.stdout.trim()) {
            hooksToRemove.push(hook.processId);
            continue;
        }

        const lines = hook.stdout.split('\n');
        const exitCode = (await hook.shellCommand.result).code;
        let responseJson = {};

        for (const line of lines) {
            if (line.trim().startsWith("{")) {
                try {
                    const parsed = JSON.parse(line.trim());
                    if (!("async" in parsed)) {
                        responseJson = parsed;
                        break;
                    }
                } catch { }
            }
        }

        responses.push({
            processId: hook.processId,
            response: responseJson,
            hookName: hook.hookName,
            hookEvent: hook.hookEvent,
            toolName: hook.toolName,
            stdout: hook.stdout,
            stderr: hook.stderr,
            exitCode
        });

        hook.responseAttachmentSent = true;
    }

    for (const pid of hooksToRemove) hookRegistry.delete(pid);
    return responses;
}

export function acknowledgeResponses(processIds: string[]) {
    for (const pid of processIds) {
        const hook = hookRegistry.get(pid);
        if (hook && hook.responseAttachmentSent) {
            hookRegistry.delete(pid);
        }
    }
}

export function createAttachment(data: any) {
    return {
        attachment: data,
        type: "attachment",
        uuid: randomUUID(),
        timestamp: new Date().toISOString()
    };
}

async function measureAttachmentCompute(label: string, computeFn: () => Promise<any[]>) {
    const startTime = Date.now();
    try {
        const result = await computeFn();
        const duration = Date.now() - startTime;
        if (Math.random() < 0.05) {
            logTelemetryEvent("tengu_attachment_compute_duration", { label, duration, count: result.length });
        }
        return result;
    } catch (err) {
        logger.error(`Attachment error in ${label}:`, err);
        return [];
    }
}

export async function getAttachments(context: any, items: any[]) {
    if (process.env.CLAUDE_CODE_DISABLE_ATTACHMENTS) return [];

    const isSubAgent = !!context.agentId;

    const mentions = items.length > 0 ? [
        measureAttachmentCompute("at_mentioned_files", () => getAtMentionAttachments(items, context)),
    ] : [];

    const sharedContext = [
        measureAttachmentCompute("todo_reminders", () => getTodoAttachments(items, context))
    ];

    const rootOnlyContext = !isSubAgent ? [
        measureAttachmentCompute("ide_selection", () => getIdeSelectionAttachment(context)),
        measureAttachmentCompute("diagnostics", () => getDiagnosticAttachments()),
        measureAttachmentCompute("async_hook_responses", () => getAsyncHookResponseAttachments())
    ] : [];

    const results = await Promise.all([
        ...mentions,
        ...sharedContext,
        ...rootOnlyContext
    ]);

    return results.flat();
}

async function getAtMentionAttachments(items: any[], context: any) {
    const mentions = items.filter(item => typeof item === 'string' && item.startsWith('@'));
    const results: any[] = [];

    for (const mention of mentions) {
        const filePath = mention.slice(1);
        try {
            const fullPath = path.resolve(getOriginalCwd(), filePath);
            if (fs.existsSync(fullPath)) {
                if (fs.statSync(fullPath).isDirectory()) {
                    results.push({
                        type: "directory",
                        path: filePath,
                        content: fs.readdirSync(fullPath).join('\n')
                    });
                } else {
                    results.push({
                        type: "file",
                        filename: filePath,
                        content: fs.readFileSync(fullPath, 'utf-8')
                    });
                }
            }
        } catch (err) {
            logger.error(`Error resolving at-mention ${mention}:`, err);
        }
    }
    return results;
}

async function getIdeSelectionAttachment(context: any) {
    try {
        const ideClient = await getConnectedIdeClient();
        if (!ideClient) return [];
        // Placeholder for real selection retrieval
        return [];
    } catch {
        return [];
    }
}

async function getTodoAttachments(items: any[], context: any) {
    if (!context.agentId) return [];

    // Logic from chunk_445.ts (jx5)
    // Only check if explicitly mentioned OR randomly with low probability? 
    // Actually typically on every turn if needed, but ConversationHistoryManager handles the periodic checks.
    // Here we might be checking for explicit mentions or relevance.

    // For now, let's rely on explicit checks or always returning pending if called
    // The caller `getAttachments` calls this every time.

    const reminders = await getTodoReminders(context.agentId);
    if (!reminders || reminders.length === 0) return [];

    return [{
        type: "todo_reminder",
        content: reminders,
        itemCount: reminders.length
    }];
}

async function getDiagnosticAttachments() {
    try {
        const diagManager = DiagnosticsManager.getInstance();
        const lspDiagManager = LspDiagnosticsManager.getInstance();

        const diagnostics = await diagManager.getNewDiagnostics();
        const lspDiagnostics = lspDiagManager.processPendingDiagnostics();

        const results: any[] = [];
        if (diagnostics && diagnostics.length > 0) {
            results.push({
                type: "diagnostics",
                files: diagnostics,
                isNew: true
            });
        }
        if (lspDiagnostics && lspDiagnostics.length > 0) {
            results.push(...lspDiagnostics.map(d => ({
                type: "lsp_diagnostics",
                ...d
            })));
        }
        return results;
    } catch {
        return [];
    }
}

async function getAsyncHookResponseAttachments() {
    const responses = await checkForNewResponses();
    if (responses.length === 0) return [];

    const attachments = responses.map(resp => ({
        type: "async_hook_response",
        ...resp
    }));

    acknowledgeResponses(responses.map(r => r.processId));
    return attachments;
}

export async function* attachmentStream(context: any, items: any[]) {
    const attachments = await getAttachments(context, items);
    for (const attachment of attachments) {
        yield createAttachment(attachment);
    }
}

export async function readFileAttachment(
    filename: string,
    context: any,
    successMetric: string,
    errorMetric: string,
    trigger: string,
    options: any = {}
) {
    try {
        const fullPath = resolvePath(filename, getOriginalCwd());
        if (!fs.existsSync(fullPath)) {
            logTelemetryEvent(errorMetric, { filename, trigger, reason: "not_found" });
            return null;
        }

        const stats = fs.statSync(fullPath);
        if (stats.isDirectory()) {
            return null;
        }

        const content = fs.readFileSync(fullPath, "utf-8");
        let finalContent = content;
        if (options.fileReadingLimits?.maxTokens) {
            if (content.length > options.fileReadingLimits.maxTokens * 4) {
                finalContent = content.substring(0, options.fileReadingLimits.maxTokens * 4) + "\n... [content truncated]";
            }
        }

        logTelemetryEvent(successMetric, { filename, trigger });
        return {
            type: "file",
            filename,
            content: finalContent
        };
    } catch (err: any) {
        logTelemetryEvent(errorMetric, { filename, trigger, error: err.message });
        return null;
    }
}
