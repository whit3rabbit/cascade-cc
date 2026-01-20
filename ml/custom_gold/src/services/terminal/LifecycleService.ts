
// Logic from chunk_587.ts (Lifecycle & Auto-Update)

import { execa } from "execa";
import { existsSync, writeFileSync, unlinkSync, statSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { runHookCommands } from "../hooks/HookDispatcher.js";
import { getGlobalState } from "../session/globalState.js";

/**
 * Gracefully shuts down the application.
 */
export async function gracefulExit(exitCode = 0, reason = "other") {
    console.log(`\nShutting down (reason: ${reason})...`);

    // 1. Run session end hooks
    try {
        // await executeSessionEndHooks(reason);
    } catch (err) {
        // Ignore errors during hook execution
    }

    // 2. Perform generic cleanup
    try {
        // await performCleanup();
    } catch (err) {
        // Ignore cleanup errors
    }

    process.exit(exitCode);
}

// --- Update Service ---

const UPDATE_LOCK_TIMEOUT = 300000; // 5 minutes

function getUpdateLockPath() {
    return join(process.cwd(), ".update.lock");
}

/**
 * Checks for the latest version on npm.
 */
export async function fetchLatestVersion(packageName: string) {
    try {
        const { stdout } = await execa("npm", ["view", `${packageName}@latest`, "version"]);
        return stdout.trim();
    } catch {
        return null;
    }
}

/**
 * Orchestrates the global update process.
 */
export async function installUpdate(packageName: string, version = "latest") {
    const lockPath = getUpdateLockPath();

    // 1. Check/Set lock
    if (existsSync(lockPath)) {
        const stats = statSync(lockPath);
        if (Date.now() - stats.mtimeMs < UPDATE_LOCK_TIMEOUT) {
            throw new Error("Another update is in progress.");
        }
    }
    writeFileSync(lockPath, String(process.pid));

    try {
        // 2. Install via npm or bun
        const installer = process.env.USE_BUN ? "bun" : "npm";
        const args = installer === "bun" ? ["pm", "install", "-g", `${packageName}@${version}`] : ["install", "-g", `${packageName}@${version}`];

        await execa(installer, args);
        console.log(`Successfully updated to ${version}`);
        return "success";
    } catch (err) {
        console.error("Update failed:", err);
        return "install_failed";
    } finally {
        // 3. Cleanup lock
        if (existsSync(lockPath)) unlinkSync(lockPath);
    }
}

/**
 * Registers signal handlers for graceful shutdown.
 */
export function registerSignalHandlers() {
    process.on("SIGINT", () => gracefulExit(0, "SIGINT"));
    process.on("SIGTERM", () => gracefulExit(143, "SIGTERM"));
}

/**
 * Creates base hook input for all hook events. (aF)
 */
function createBaseHookInput(mode: string) {
    const state = getGlobalState() as any;
    return {
        session_id: state.sessionId || "default",
        transcript_path: state.transcriptPath || "",
        cwd: process.cwd(),
        permission_mode: mode
    };
}

/**
 * Dispatches Stop hooks. (lF0)
 */
export async function* executeStopHooks(
    mode: string,
    signal: AbortSignal,
    timeoutMs?: number,
    isSubagent = false,
    agentId?: string,
    context?: any,
    messages?: any[]
) {
    const hookInput = {
        ...createBaseHookInput(mode),
        hook_event_name: isSubagent ? "SubagentStop" : "Stop",
        ...(agentId ? { agent_id: agentId } : {})
    };
    yield* runHookCommands({
        hookInput,
        toolUseID: agentId || "",
        matchQuery: mode,
        signal,
        timeoutMs,
        toolUseContext: context,
        messages
    });
}

export function getStopHookMessage(err: any): string {
    return `Stop hook feedback:\n${err.blockingError || err.message}`;
}

export async function* executePreToolHooks(
    toolName: string,
    toolUseId: string,
    toolInput: any,
    context: any,
    mode: string,
    signal?: AbortSignal
) {
    const hookInput = {
        ...createBaseHookInput(mode),
        hook_event_name: "PreToolUse",
        tool_name: toolName,
        tool_input: toolInput,
        tool_use_id: toolUseId
    };
    yield* runHookCommands({
        hookInput,
        toolUseID: toolUseId,
        matchQuery: toolName,
        signal,
        toolUseContext: context
    });
}

export async function* executePostToolHooks(
    toolName: string,
    toolUseId: string,
    toolOutput: any,
    context: any,
    mode: string,
    signal?: AbortSignal
) {
    const hookInput = {
        ...createBaseHookInput(mode),
        hook_event_name: "PostToolUse",
        tool_name: toolName,
        tool_output: toolOutput,
        tool_use_id: toolUseId
    };
    yield* runHookCommands({
        hookInput,
        toolUseID: toolUseId,
        matchQuery: toolName,
        signal,
        toolUseContext: context
    });
}

export async function* executePostToolFailureHooks(
    toolName: string,
    toolUseId: string,
    toolInput: any,
    error: any,
    context: any,
    mode: string,
    signal?: AbortSignal,
    isInterrupt = false
) {
    const hookInput = {
        ...createBaseHookInput(mode),
        hook_event_name: "PostToolUseFailure",
        tool_name: toolName,
        tool_input: toolInput,
        error,
        tool_use_id: toolUseId,
        is_interrupt: isInterrupt
    };
    yield* runHookCommands({
        hookInput,
        toolUseID: toolUseId,
        matchQuery: toolName,
        signal,
        toolUseContext: context
    });
}

export async function executeSessionStartHooks(source: string, mode: string) {
    const hookInput = {
        ...createBaseHookInput(mode),
        hook_event_name: "SessionStart",
        source
    };
    // runHookCommands is a generator, so we just run it to completion if we don't need yields
    for await (const _ of runHookCommands({ hookInput, toolUseID: "", matchQuery: source, toolUseContext: {} })) { }
}

export async function executeSessionEndHooks(reason: string, mode: string) {
    const hookInput = {
        ...createBaseHookInput(mode),
        hook_event_name: "SessionEnd",
        reason
    };
    for await (const _ of runHookCommands({ hookInput, toolUseID: "", matchQuery: reason, toolUseContext: {} })) { }
}
