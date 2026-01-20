
// Logic from chunk_571.ts (Plugin Error Dashboard)

import React, { useState } from "react";
import { Box, Text, useInput } from "ink";
import { figures } from "../../vendor/terminalFigures.js";
import { usePagination } from "./MarketplaceManager.js";

// --- Error Message Formatter (hG7 / bG7) ---
export function formatPluginErrorMessage(error: any): string {
    switch (error.type) {
        case "path-not-found":
            return `${error.component} path not found: ${error.path}`;
        case "git-auth-failed":
            return `Git ${error.authType.toUpperCase()} authentication failed for ${error.gitUrl}`;
        case "git-timeout":
            return `Git ${error.operation} timed out for ${error.gitUrl}`;
        case "network-error":
            return `Network error accessing ${error.url}${error.details ? `: ${error.details}` : ""}`;
        case "manifest-parse-error":
            return `Failed to parse manifest at ${error.manifestPath}: ${error.parseError}`;
        case "manifest-validation-error":
            return `Invalid manifest at ${error.manifestPath}: ${error.validationErrors.join(", ")}`;
        case "plugin-not-found":
            return `Plugin '${error.pluginId}' not found in marketplace '${error.marketplace}'`;
        case "marketplace-not-found":
            return `Marketplace '${error.marketplace}' not found`;
        case "marketplace-load-failed":
            return `Failed to load marketplace '${error.marketplace}': ${error.reason}`;
        case "repository-scan-failed":
            return `Failed to scan repository at ${error.repositoryPath}: ${error.reason}`;
        case "mcp-config-invalid":
            return `Invalid MCP server config for '${error.serverName}': ${error.validationError}`;
        case "hook-load-failed":
            return `Failed to load hooks from ${error.hookPath}: ${error.reason}`;
        case "component-load-failed":
            return `Failed to load ${error.component} from ${error.path}: ${error.reason}`;
        case "mcpb-download-failed":
            return `Failed to download MCPB from ${error.url}: ${error.reason}`;
        case "mcpb-extract-failed":
            return `Failed to extract MCPB ${error.mcpbPath}: ${error.reason}`;
        case "mcpb-invalid-manifest":
            return `MCPB manifest invalid at ${error.mcpbPath}: ${error.validationError}`;
        case "marketplace-blocked-by-policy":
            return error.blockedByBlocklist
                ? `Marketplace '${error.marketplace}' is blocked by enterprise policy`
                : `Marketplace '${error.marketplace}' is not in the allowed marketplace list`;
        case "generic-error":
            return error.error;
        default:
            return "Unknown error";
    }
}

// --- Error Fix Suggestion (fG7 / kX9) ---
export function getPluginErrorSuggestion(error: any): string | null {
    switch (error.type) {
        case "path-not-found":
            return "Check that the path in your manifest or marketplace config is correct";
        case "git-auth-failed":
            return error.authType === "ssh"
                ? "Configure SSH keys or use HTTPS URL instead"
                : "Configure credentials or use SSH URL instead";
        case "git-timeout":
        case "network-error":
            return "Check your internet connection and try again";
        case "manifest-parse-error":
            return "Check manifest file syntax in the plugin directory";
        case "manifest-validation-error":
            return "Check manifest file follows the required schema";
        case "plugin-not-found":
            return `Plugin may not exist in marketplace '${error.marketplace}'`;
        case "marketplace-not-found":
            return error.availableMarketplaces?.length > 0
                ? `Available marketplaces: ${error.availableMarketplaces.join(", ")}`
                : "Add the marketplace first using /plugin marketplace add";
        case "mcp-config-invalid":
            return "Check MCP server configuration in .mcp.json or manifest";
        case "hook-load-failed":
            return "Check hooks.json file syntax and structure";
        case "component-load-failed":
            return `Check ${error.component} directory structure and file permissions`;
        case "mcpb-download-failed":
            return "Check your internet connection and URL accessibility";
        case "mcpb-extract-failed":
            return "Verify the MCPB file is valid and not corrupted";
        case "mcpb-invalid-manifest":
            return "Contact the plugin author about the invalid manifest";
        case "marketplace-blocked-by-policy":
            if (error.blockedByBlocklist) return "This marketplace source is explicitly blocked by your administrator";
            return error.allowedSources?.length > 0
                ? `Allowed sources: ${error.allowedSources.join(", ")}`
                : "Contact your administrator to configure allowed marketplace sources";
        default:
            return null;
    }
}

// --- Plugin Errors View (yX9) ---
export function PluginErrorsView({ errors, onExit }: { errors: any[], onExit: () => void }) {
    const [selectedIndex, setSelectedIndex] = useState(0);
    const { getVisibleItems, scrollPosition, needsPagination, toActualIndex } = usePagination({
        totalItems: errors.length,
        selectedIndex
    });

    useInput((input, key) => {
        if (key.escape) onExit();
        if (key.upArrow && selectedIndex > 0) setSelectedIndex(selectedIndex - 1);
        if (key.downArrow && selectedIndex < errors.length - 1) setSelectedIndex(selectedIndex + 1);
    });

    const visibleErrors = getVisibleItems(errors);

    return (
        <Box flexDirection="column" paddingX={1} borderStyle="round">
            <Box marginBottom={1}>
                <Text bold>Plugin Errors</Text>
                {needsPagination && (
                    <Text dimColor> ({scrollPosition.current}/{scrollPosition.total})</Text>
                )}
            </Box>

            {scrollPosition.canScrollUp && (
                <Box marginLeft={2}>
                    <Text dimColor>{figures.arrowUp} more above</Text>
                </Box>
            )}

            {errors.length === 0 ? (
                <Box marginLeft={2}>
                    <Text dimColor>No plugin errors</Text>
                </Box>
            ) : (
                visibleErrors.map((error: any, i: number) => {
                    const absIndex = toActualIndex(i);
                    const isSelected = absIndex === selectedIndex;
                    const pluginName = error.plugin;
                    const message = formatPluginErrorMessage(error);
                    const suggestion = getPluginErrorSuggestion(error);

                    return (
                        <Box key={absIndex} marginLeft={2} flexDirection="column" marginBottom={1}>
                            <Box>
                                <Text color={isSelected ? "claude" : "error"}>
                                    {isSelected ? figures.pointer : figures.cross}
                                </Text>
                                <Text bold={isSelected}> {pluginName || "Global Error"}</Text>
                                <Text dimColor> from {error.source}</Text>
                            </Box>
                            <Box marginLeft={3}>
                                <Text color="error" dimColor>{message}</Text>
                            </Box>
                            {suggestion && (
                                <Box marginLeft={3}>
                                    <Text dimColor italic>{figures.arrowRight} {suggestion}</Text>
                                </Box>
                            )}
                        </Box>
                    );
                })
            )}

            {scrollPosition.canScrollDown && (
                <Box marginLeft={2}>
                    <Text dimColor>{figures.arrowDown} more below</Text>
                </Box>
            )}

            <Box paddingLeft={3}>
                <Text dimColor italic>Esc to close</Text>
            </Box>
        </Box>
    );
}

