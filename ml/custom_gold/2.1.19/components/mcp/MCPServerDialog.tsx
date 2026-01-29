/**
 * File: src/components/mcp/MCPServerDialog.tsx
 * Role: UI Components for MCP Server and Plugin Management
 */

import React from 'react';
import { Box, Text } from 'ink';
import { ActionHint } from '../common/ActionHint.js';


interface MCPServerMultiselectDialogProps {
    hasSelection: boolean;
}

/**
 * UI Component for displaying plugin installation/toggle/detail actions.
 */
export function MCPServerMultiselectDialog({ hasSelection }: MCPServerMultiselectDialogProps) {
    return (
        <Box marginTop={1} flexDirection="row" flexWrap="wrap">
            {hasSelection && (
                <ActionHint shortcut="i" action="install" bold color="green" />
            )}
            <Box marginRight={2}>
                <Text dimColor italic>type to search</Text>
            </Box>
            <ActionHint shortcut="Space" action="toggle" />
            <ActionHint shortcut="Enter" action="details" />
            <ActionHint shortcut="Esc" action="back" />
        </Box>
    );
}

const ERROR_MESSAGES: Record<string, React.ReactNode> = {
    "git-not-installed": (
        <Box flexDirection="column">
            <Text color="red">Git is required to install marketplaces.</Text>
            <Text dimColor>Please install git and restart Claude Code.</Text>
        </Box>
    ),
    "all-blocked-by-policy": (
        <Box flexDirection="column">
            <Text color="red">Your organization policy does not allow any external marketplaces.</Text>
            <Text dimColor>Contact your administrator.</Text>
        </Box>
    ),
    "policy-restricts-sources": (
        <Box flexDirection="column">
            <Text color="yellow">Your organization restricts which marketplaces can be added.</Text>
            <Text dimColor>Switch to the Marketplaces tab to view allowed sources.</Text>
        </Box>
    ),
    "all-marketplaces-failed": (
        <Box flexDirection="column">
            <Text color="red">Failed to load marketplace data.</Text>
            <Text dimColor>Check your network connection.</Text>
        </Box>
    ),
    "all-plugins-installed": (
        <Box flexDirection="column">
            <Text color="green">All available plugins are already installed.</Text>
            <Text dimColor>Check for new plugins later or add more marketplaces.</Text>
        </Box>
    ),
    "no-marketplaces-configured": (
        <Box flexDirection="column">
            <Text color="yellow">No plugins available.</Text>
            <Text dimColor>Add a marketplace first using the Marketplaces tab.</Text>
        </Box>
    )
};

/**
 * Helper function to render different error messages based on the marketplace error reason.
 */
export function renderMarketplaceError(reason: string) {
    return ERROR_MESSAGES[reason] || ERROR_MESSAGES["no-marketplaces-configured"];
}
