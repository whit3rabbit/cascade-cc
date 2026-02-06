/**
 * File: src/utils/cleanup.ts
 * Role: Centralized cleanup registry for graceful shutdowns and resource management.
 */

import { terminalLog } from "./shared/runtime.js";
import { clearIterm2Progress } from "./terminal/iterm2ProgressBar.js";

type CleanupTask = () => void | Promise<void>;
const cleanupTasks: CleanupTask[] = [];
let isCleaningUp = false;
let handlersInitialized = false;

/**
 * Registers a function to be called during application cleanup.
 */
export function onCleanup(task: CleanupTask): void {
    cleanupTasks.push(task);
}

/**
 * Executes all registered cleanup tasks.
 */
export async function runCleanup(): Promise<void> {
    if (isCleaningUp) return;
    isCleaningUp = true;

    terminalLog("\nCleaning up resources...");

    // Run tasks in reverse order (LIFO)
    for (let i = cleanupTasks.length - 1; i >= 0; i--) {
        try {
            const task = cleanupTasks[i];
            const result = task();
            if (result instanceof Promise) {
                await result;
            }
        } catch (error) {
            console.error("[Cleanup] Task failed:", error);
        }
    }
}

/**
 * Sets up listeners for common signals to ensure cleanup runs.
 */
export function setupCleanupHandlers(): void {
    if (handlersInitialized) return;
    handlersInitialized = true;

    const signals: NodeJS.Signals[] = ['SIGINT', 'SIGTERM', 'SIGHUP'];

    // Ensure iterm2 progress bar is cleared on exit
    onCleanup(() => clearIterm2Progress());

    signals.forEach(signal => {
        process.on(signal, async () => {
            await runCleanup();
            process.exit(0);
        });
    });

    process.on('beforeExit', async () => {
        await runCleanup();
    });

    process.on('exit', () => {
        try {
            clearIterm2Progress();
        } catch {
            // ignore
        }
    });

    // Handle unexpected errors if possible
    process.on('uncaughtException', async (error) => {
        console.error("[Cleanup] Uncaught exception:", error);
        await runCleanup();
        process.exit(1);
    });
}

// Re-exporting existing utilities as per the original file's role
export * from "./terminal/terminalSupport.js";
export * from "./terminal/tmuxUtils.js";
export * from "../services/teams/TeamManager.js";
export * from "./shared/loggingUtils.js";
export * from "./shared/timeUtils.js";
export * from "../services/telemetry/TracingUtils.js";
