
// Logic from chunk_549.ts (MCP Management UI)

import React, { useState, useCallback } from "react";
import { Box, Text } from "ink";
import { PermissionSelect } from "../permissions/PermissionComponents.js";
import { figures } from "../../vendor/terminalFigures.js";
import { ModalHeader } from "./StatusView.js";
import { Tabs, Tab } from "./PermissionsManager.js";
import ExternalLink from "./ExternalLink.js";

// --- MCP Capabilities View ($H1) ---
export function McpCapabilitiesView({
    serverToolsCount = 0,
    serverPromptsCount = 0,
    serverResourcesCount = 0
}: any) {
    const caps = [];
    if (serverToolsCount > 0) caps.push("tools");
    if (serverResourcesCount > 0) caps.push("resources");
    if (serverPromptsCount > 0) caps.push("prompts");

    return (
        <Box>
            <Text bold>Capabilities: </Text>
            <Text>{caps.length > 0 ? caps.join(", ") : "none"}</Text>
        </Box>
    );
}

// --- MCP Server Details View (SN0) ---
export function McpServerDetailsView({
    server,
    serverToolsCount,
    onViewTools,
    onCancel,
    onComplete
}: any) {
    const [reconnecting, setReconnecting] = useState(false);

    const handleToggle = async () => {
        // Logic to enable/disable server
        onComplete(`MCP server '${server.name}' ${server.client.type === "disabled" ? "enabled" : "disabled"}.`);
    };

    const handleReconnect = async () => {
        setReconnecting(true);
        // Simulate reconnection
        setTimeout(() => {
            setReconnecting(false);
            onComplete(`Reconnected to ${server.name}.`);
        }, 1000);
    };

    const options = [];
    if (server.client.type !== "disabled" && serverToolsCount > 0) {
        options.push({ label: "View tools", value: "tools" });
    }
    if (server.client.type !== "disabled") {
        options.push({ label: "Reconnect", value: "reconnect" });
    }
    options.push({
        label: server.client.type !== "disabled" ? "Disable" : "Enable",
        value: "toggle"
    });
    options.push({ label: "Back", value: "back" });

    if (reconnecting) {
        return (
            <Box flexDirection="column" gap={1} padding={1}>
                <Text>Reconnecting to <Text bold>{server.name}</Text></Text>
                <Box>
                    <Text color="suggestion">{figures.info} </Text>
                    <Text> Restarting MCP server process</Text>
                </Box>
                <Text dimColor>This may take a few moments.</Text>
            </Box>
        );
    }

    return (
        <Box flexDirection="column" paddingX={1} borderStyle="round">
            <Box marginBottom={1}>
                <Text bold>{server.name.toUpperCase()} MCP Server</Text>
            </Box>

            <Box flexDirection="column">
                <Box>
                    <Text bold>Status: </Text>
                    {server.client.type === "disabled" ? <Text dimColor>{figures.circleEmpty} disabled</Text> :
                        server.client.type === "connected" ? <Text color="success">{figures.tick} connected</Text> :
                            <Text color="warning">connecting…</Text>}
                </Box>
                <Box>
                    <Text bold>Command: </Text>
                    <Text dimColor>{server.config.command}</Text>
                </Box>
                {server.config.args?.length > 0 && (
                    <Box>
                        <Text bold>Args: </Text>
                        <Text dimColor>{server.config.args.join(" ")}</Text>
                    </Box>
                )}
                {server.client.type === "connected" && (
                    <McpCapabilitiesView
                        serverToolsCount={serverToolsCount}
                        serverPromptsCount={0} // Mock
                        serverResourcesCount={0} // Mock
                    />
                )}
            </Box>

            <Box marginTop={1}>
                <PermissionSelect
                    options={options}
                    onChange={(val) => {
                        if (val === "tools") onViewTools();
                        else if (val === "reconnect") handleReconnect();
                        else if (val === "toggle") handleToggle();
                        else if (val === "back") onCancel();
                    }}
                    onCancel={onCancel}
                />
            </Box>
            <Box marginLeft={3}>
                <Text dimColor>Esc to go back</Text>
            </Box>
        </Box>
    );
}

// --- MCP Servers List View (TN0) ---
export function McpServersListView({ servers, onSelectServer, onCancel }: any) {
    const claudeCodeServers = servers.filter((s: any) => s.source === "claude-code");
    const claudeAiServers = servers.filter((s: any) => s.source === "claude-ai");

    const renderServerList = (list: any[]) => {
        const options = list.map(s => ({
            label: s.name,
            value: s.id,
            description: s.client.type === "connected" ? "Connected" : "Disconnected"
        }));

        return (
            <Box flexDirection="column" gap={1}>
                <PermissionSelect
                    options={options}
                    onChange={(id) => onSelectServer(list.find(s => s.id === id))}
                    onCancel={onCancel}
                />
                <Box flexDirection="column" gap={1}>
                    <Text dimColor>For help configuring MCP servers, see: <ExternalLink url="https://code.claude.com/docs/en/mcp" /></Text>
                </Box>
            </Box>
        );
    };

    return (
        <Box flexDirection="column">
            <ModalHeader
                title="Manage MCP servers"
                subtitle={`${servers.length} server${servers.length === 1 ? "" : "s"}`}
                onCancel={onCancel}
            />

            <Tabs>
                <Tab title="Claude Code">
                    {renderServerList(claudeCodeServers)}
                </Tab>
                <Tab title="claude.ai">
                    {renderServerList(claudeAiServers)}
                    <Box marginTop={1}>
                        <Text dimColor>Config: <ExternalLink url="https://claude.ai/settings/connectors" /></Text>
                    </Box>
                </Tab>
            </Tabs>
        </Box>
    );
}
