
// Logic from chunk_550.ts (MCP Tool Browsing UI)

import React, { useState, useEffect } from "react";
import { Box, Text } from "ink";
import { PermissionSelect } from "../permissions/PermissionComponents.js";

// --- MCP Server Tools View (hN0) ---
export function McpServerToolsView({ server, onSelectTool, onBack }: any) {
    const tools = server.tools || []; // Mocking tools access

    const options = tools.map((tool: any, index: number) => {
        const labels = [];
        if (tool.isReadOnly) labels.push("read-only");
        if (tool.isDestructive) labels.push("destructive");
        if (tool.isOpenWorld) labels.push("open-world");

        return {
            label: tool.userFacingName || tool.name,
            value: index.toString(),
            description: labels.length > 0 ? labels.join(", ") : undefined,
            descriptionColor: tool.isDestructive ? "error" : tool.isReadOnly ? "success" : undefined
        };
    });

    return (
        <Box flexDirection="column">
            <Box flexDirection="column" paddingX={1} borderStyle="round">
                <Box marginBottom={1}>
                    <Text bold>Tools for {server.name}</Text>
                    <Text dimColor> ({tools.length} tools)</Text>
                </Box>
                {tools.length === 0 ? (
                    <Text dimColor>No tools available</Text>
                ) : (
                    <PermissionSelect
                        options={options}
                        onChange={(val) => {
                            const idx = parseInt(val);
                            onSelectTool(tools[idx], idx);
                        }}
                        onCancel={onBack}
                    />
                )}
            </Box>
            <Box marginLeft={3}>
                <Text dimColor>Esc to go back</Text>
            </Box>
        </Box>
    );
}

// --- MCP Tool Detail View (uN0) ---
export function McpToolDetailView({ tool, server, onBack }: any) {
    const [description, setDescription] = useState("");

    useEffect(() => {
        // In actual implementation, this fetches the tool description using MCP protocol
        setDescription(tool.descriptionText || "Loading description...");
    }, [tool]);

    const properties = tool.inputJSONSchema?.properties || {};
    const required = tool.inputJSONSchema?.required || [];

    return (
        <Box flexDirection="column">
            <Box flexDirection="column" paddingX={1} borderStyle="round">
                <Box marginBottom={1}>
                    <Text bold>{tool.userFacingName || tool.name}</Text>
                    <Text dimColor> ({server.name})</Text>
                    {tool.isReadOnly && <Text color="success"> [read-only]</Text>}
                    {tool.isDestructive && <Text color="error"> [destructive]</Text>}
                </Box>

                <Box flexDirection="column">
                    <Box>
                        <Text bold>Full name: </Text>
                        <Text dimColor>{tool.name}</Text>
                    </Box>
                    {description && (
                        <Box flexDirection="column" marginTop={1}>
                            <Text bold>Description:</Text>
                            <Text wrap="wrap">{description}</Text>
                        </Box>
                    )}
                    {Object.keys(properties).length > 0 && (
                        <Box flexDirection="column" marginTop={1}>
                            <Text bold>Parameters:</Text>
                            <Box marginLeft={2} flexDirection="column">
                                {Object.entries(properties).map(([name, prop]: [string, any]) => (
                                    <Text key={name}>
                                        • {name}{required.includes(name) ? <Text dimColor> (required)</Text> : ""}:
                                        <Text dimColor> {prop.type || "unknown"}</Text>
                                        {prop.description && <Text dimColor> - {prop.description}</Text>}
                                    </Text>
                                ))}
                            </Box>
                        </Box>
                    )}
                </Box>
            </Box>
            <Box marginLeft={3}>
                <Text dimColor>Esc to go back</Text>
            </Box>
        </Box>
    );
}
