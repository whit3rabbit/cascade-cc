
// Logic from chunk_586.ts (Hook Event Orchestration)

import { parseHookOutput } from "./HookExecutionService.js";

/**
 * Result of a single hook execution.
 */
export interface HookResult {
    outcome: "success" | "blocking" | "non_blocking_error" | "cancelled";
    message?: any;
    blockingError?: any;
    systemMessage?: string;
    additionalContext?: string;
    permissionBehavior?: "allow" | "deny" | "ask" | "passthrough";
    updatedInput?: any;
    [key: string]: any;
}

/**
 * Orchestrates the execution of all hooks matched for a given event.
 */
export async function* orchestrateHooks({
    event,
    hooks,
    input,
    context
}: any) {
    let finalPermissionBehavior: string | undefined;
    let stats = { success: 0, blocking: 0, error: 0, cancelled: 0 };

    for (const hook of hooks) {
        try {
            // 1. Execute the hook based on its type
            let result: HookResult;
            if (hook.type === "agent") {
                // result = await executeAgentHook(...)
                result = { outcome: "success" }; // stub
            } else {
                // result = await executeBashHook(...)
                result = { outcome: "success" }; // stub
            }

            // 2. Process the result
            if (result.outcome === "success") stats.success++;
            if (result.outcome === "blocking") stats.blocking++;
            if (result.outcome === "non_blocking_error") stats.error++;
            if (result.outcome === "cancelled") stats.cancelled++;

            // 3. Aggregate permission decisions
            if (result.permissionBehavior) {
                if (result.permissionBehavior === "deny") {
                    finalPermissionBehavior = "deny";
                } else if (result.permissionBehavior === "ask" && finalPermissionBehavior !== "deny") {
                    finalPermissionBehavior = "ask";
                } else if (result.permissionBehavior === "allow" && !finalPermissionBehavior) {
                    finalPermissionBehavior = "allow";
                }
            }

            // 4. Yield intermediate results
            yield result;

            if (result.outcome === "blocking") break;

        } catch (err) {
            stats.error++;
            yield { outcome: "non_blocking_error", error: err };
        }
    }

    // Final summary event
    yield {
        type: "orchestration_complete",
        event,
        stats,
        finalPermissionBehavior
    };
}
