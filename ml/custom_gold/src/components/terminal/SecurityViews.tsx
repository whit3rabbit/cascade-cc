
// Logic from chunk_597.ts (Security Views & Datadog Analytics)

import React from "react";
import { Box, Text } from "ink";
import { Select } from "../shared/Select.js";

// --- Security Views ---

/**
 * WARNING: Shown when running with --dangerously-bypass-permissions.
 */
export function BypassPermissionsWarning({ onAccept }: { onAccept: () => void }) {
    return (
        <Box flexDirection="column" gap={1} padding={1} borderStyle="round" borderColor="error">
            <Text bold color="error">WARNING: Running in Bypass Permissions mode</Text>
            <Box flexDirection="column" gap={1}>
                <Text>Claude Code will NOT ask for approval before running potentially dangerous commands.</Text>
                <Text>Use this only in a sandboxed container/VM with restricted network access.</Text>
            </Box>
            <Select
                options={[
                    { label: "No, exit", value: "decline" },
                    { label: "Yes, I accept", value: "accept" }
                ]}
                onChange={(val: string) => val === "accept" ? onAccept() : process.exit(1)}
            />
        </Box>
    );
}

/**
 * Prompts user to trust a new MCP server found in the project.
 */
export function McpTrustDialog({ serverName, onDone }: { serverName: string, onDone: (trust: boolean) => void }) {
    return (
        <Box flexDirection="column" gap={1} borderStyle="round" borderColor="suggestion">
            <Text bold>Trust MCP Server: {serverName}?</Text>
            <Text>This project wants to use a new MCP server. Only enable if you trust the source.</Text>
            <Select
                options={[
                    { label: "Trust and use for this project", value: "yes" },
                    { label: "Skip for now", value: "no" }
                ]}
                onChange={(val: string) => onDone(val === "yes")}
            />
        </Box>
    );
}

// --- Datadog Logger ---

export async function logToDatadog(event: string, payload: any) {
    console.log(`[Datadog] ${event}: ${JSON.stringify(payload)}`);
}

/**
 * Unified Analytics Dispatcher
 */
export function dispatchEvent(event: string, payload: any) {
    logToDatadog(event, payload);
}
