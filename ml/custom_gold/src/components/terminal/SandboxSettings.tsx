
// Logic from chunk_573.ts (Session Memory & Sandbox UI)

import React from "react";
import { Box, Text } from "ink";
import { PermissionSelect } from "../permissions/PermissionComponents.js";
import { Link } from "../shared/Link.js";

// --- Session Memory Config (tj2) ---
let sessionMemoryConfig = {
    minimumMessageTokensToInit: 1000,
    minimumTokensBetweenUpdate: 500,
    toolCallsBetweenUpdates: 5
};

export function initSessionMemoryConfig(config: any) {
    sessionMemoryConfig = { ...sessionMemoryConfig, ...config };
}

// --- Memory Path Filter (eG7) ---
/**
 * Creates a permission filter that only allows tool j3 (readFile) on a specific path.
 */
export function createMemoryPathFilter(allowedPath: string, toolName: string) {
    return async (name: string, input: any) => {
        if (name === toolName && typeof input === "object" && input?.file_path === allowedPath) {
            return { behavior: "allow", updatedInput: input };
        }
        return {
            behavior: "deny",
            message: `Only ${toolName} on ${allowedPath} is allowed`,
            decisionReason: {
                type: "other",
                reason: `Restricted to session memory file: ${allowedPath}`
            }
        };
    };
}

// --- Sandbox Restrictions View (ZI9) ---
export function SandboxRestrictionsView({ sandboxConfig }: any) {
    const fsRead = sandboxConfig.fsRead || { denyOnly: [] };
    const fsWrite = sandboxConfig.fsWrite || { allowOnly: [], denyWithinAllow: [] };
    const network = sandboxConfig.network || { allowedHosts: [], deniedHosts: [] };
    const excludedCommands = sandboxConfig.excludedCommands || [];
    const unixSockets = sandboxConfig.unixSockets || [];

    return (
        <Box flexDirection="column" paddingY={1}>
            <Box flexDirection="column">
                <Text bold color="permission">Excluded Commands:</Text>
                <Text dimColor>{excludedCommands.length > 0 ? excludedCommands.join(", ") : "None"}</Text>
            </Box>

            {fsRead.denyOnly.length > 0 && (
                <Box marginTop={1} flexDirection="column">
                    <Text bold color="permission">Filesystem Read Restrictions:</Text>
                    <Text dimColor>Denied: {fsRead.denyOnly.join(", ")}</Text>
                </Box>
            )}

            {fsWrite.allowOnly.length > 0 && (
                <Box marginTop={1} flexDirection="column">
                    <Text bold color="permission">Filesystem Write Restrictions:</Text>
                    <Text dimColor>Allowed: {fsWrite.allowOnly.join(", ")}</Text>
                    {fsWrite.denyWithinAllow.length > 0 && (
                        <Text dimColor>Denied within allowed: {fsWrite.denyWithinAllow.join(", ")}</Text>
                    )}
                </Box>
            )}

            {(network.allowedHosts?.length > 0 || network.deniedHosts?.length > 0) && (
                <Box marginTop={1} flexDirection="column">
                    <Text bold color="permission">Network Restrictions:</Text>
                    {network.allowedHosts?.length > 0 && (
                        <Text dimColor>Allowed: {network.allowedHosts.join(", ")}</Text>
                    )}
                    {network.deniedHosts?.length > 0 && (
                        <Text dimColor>Denied: {network.deniedHosts.join(", ")}</Text>
                    )}
                </Box>
            )}

            {unixSockets.length > 0 && (
                <Box marginTop={1} flexDirection="column">
                    <Text bold color="permission">Allowed Unix Sockets:</Text>
                    <Text dimColor>{unixSockets.join(", ")}</Text>
                </Box>
            )}
        </Box>
    );
}

// --- Sandbox Overrides Form (JI9) ---
export function SandboxOverridesForm({ currentSettings, onSave, onCancel, isLocked }: any) {
    const currentMode = currentSettings.allowUnsandboxedCommands ? "open" : "strict";

    const options = [
        {
            label: currentMode === "open" ? "Allow unsandboxed fallback (current)" : "Allow unsandboxed fallback",
            value: "open"
        },
        {
            label: currentMode === "strict" ? "Strict sandbox mode (current)" : "Strict sandbox mode",
            value: "strict"
        }
    ];

    if (isLocked) {
        return (
            <Box flexDirection="column" paddingY={1}>
                <Text color="subtle">Override settings are managed by enterprise policy.</Text>
                <Box marginTop={1}>
                    <Text dimColor>Current setting: {currentMode === "strict" ? "Strict sandbox mode" : "Allow unsandboxed fallback"}</Text>
                </Box>
            </Box>
        );
    }

    return (
        <Box flexDirection="column" paddingY={1}>
            <Box marginBottom={1}>
                <Text bold>Configure Sandbox Overrides:</Text>
            </Box>

            <PermissionSelect
                options={options}
                onChange={(val: string) => onSave({ allowUnsandboxedCommands: val === "open" })}
            />

            <Box flexDirection="column" marginTop={1} gap={1}>
                <Text dimColor>
                    <Text bold dimColor>Allow unsandboxed fallback: </Text>
                    When a command fails due to sandbox restrictions, Claude can retry outside the sandbox.
                </Text>
                <Text dimColor>
                    <Text bold dimColor>Strict sandbox mode: </Text>
                    All bash commands must run in the sandbox unless explicitly excluded.
                </Text>
                <Box>
                    <Text dimColor>Learn more: </Text>
                    <Link url="https://code.claude.com/docs/en/sandboxing#configure-sandboxing">
                        code.claude.com/docs/en/sandboxing
                    </Link>
                </Box>
            </Box>
        </Box>
    );
}

