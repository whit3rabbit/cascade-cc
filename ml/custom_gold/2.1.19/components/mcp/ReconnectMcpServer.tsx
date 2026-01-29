/**
 * File: src/components/mcp/ReconnectMcpServer.tsx
 * Role: UI component for handling the reconnection process of an MCP server.
 */

import React, { useState, useEffect } from 'react';
import { Box, Text } from 'ink';
import Spinner from 'ink-spinner';

interface ReconnectMcpServerProps {
    serverName: string;
    onComplete: (statusMessage: string) => void;
}

/**
 * Handles the visual state and logic for reconnecting to a specific MCP server.
 */
export function ReconnectMcpServer({ serverName, onComplete }: ReconnectMcpServerProps) {
    const [error, setError] = useState<string | null>(null);
    const [isReconnecting, setIsReconnecting] = useState(true);

    useEffect(() => {
        let mounted = true;

        const attemptReconnect = async () => {
            try {
                // In a real implementation, this would call the actual McpManager/State
                // For now, we simulate a connection attempt
                await new Promise(resolve => setTimeout(resolve, 1500));

                if (!mounted) return;

                // Success mock
                onComplete(`Successfully reconnected to ${serverName}`);
            } catch (err) {
                if (mounted) {
                    const errorMsg = err instanceof Error ? err.message : String(err);
                    setError(errorMsg);
                    setIsReconnecting(false);
                    onComplete(`Error: ${errorMsg}`);
                }
            }
        };

        attemptReconnect();

        return () => { mounted = false; };
    }, [serverName, onComplete]);

    if (isReconnecting) {
        return (
            <Box flexDirection="column" padding={1}>
                <Text>Reconnecting to <Text bold color="cyan">{serverName}</Text></Text>
                <Box marginTop={1}>
                    <Text color="yellow">
                        <Spinner type="dots" />
                    </Text>
                    <Text> Establishing connection to MCP server...</Text>
                </Box>
            </Box>
        );
    }

    if (error) {
        return (
            <Box flexDirection="column" padding={1} borderStyle="single" borderColor="red">
                <Box>
                    <Text color="red">✖ Failed to reconnect to {serverName}</Text>
                </Box>
                <Box marginTop={1}>
                    <Text dimColor>Error: {error}</Text>
                </Box>

            </Box>
        );
    }

    return null;
}

export default ReconnectMcpServer;