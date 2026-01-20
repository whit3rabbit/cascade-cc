
import React, { useMemo } from 'react';
import { Box, Text } from 'ink';
import { ToolUseConfirm } from './ToolUseConfirm.js'; // chunk_461
import { PermissionSelect } from './PermissionComponents.js';
import { logTelemetryEvent } from '../../services/telemetry/telemetryInit.js';
import { McpClient } from '../../services/mcp/McpClient.js';

// chunk_466: ck2 (McpToolPermissionRequest)
export function McpToolPermissionRequest({
    toolUseConfirm,
    onDone,
    onReject,
    serverName,
    toolName,
    args
}: {
    toolUseConfirm: any;
    onDone: () => void;
    onReject: () => void;
    serverName: string;
    toolName: string;
    args: any;
}) {
    const fullToolName = `${serverName} - ${toolName}`;
    const id = `mcp__${serverName}__${toolName}`;

    const context = useMemo(() => ({
        ...toolUseConfirm,
        tool: {
            ...toolUseConfirm.tool,
            name: id,
            isMcp: true
        }
    }), [toolUseConfirm, id]);

    // Handle choices
    const handleChange = (value: string) => {
        switch (value) {
            case 'yes':
                // Log and accept
                // logTelemetry...
                context.onAllow(context.input, []);
                onDone();
                break;
            case 'yes-dont-ask-again':
                // Log and accept always
                const suggestions = context.permissionResult.behavior === "ask" ? context.permissionResult.suggestions || [] : [];
                context.onAllow(context.input, suggestions);
                onDone();
                break;
            case 'no':
                // Log and reject
                context.onReject();
                onReject();
                onDone();
                break;
        }
    };

    const options = [
        { label: "Yes", value: "yes" },
        { label: `Yes, and don't ask again for ${fullToolName} commands`, value: "yes-dont-ask-again" },
        { label: "No", value: "no" }
    ];

    return (
        <Box flexDirection="column" paddingX={1} borderStyle="round" borderColor="gray">
            <Text bold>Tool use</Text>
            <Box flexDirection="column" paddingY={1}>
                <Text>{fullToolName} ({JSON.stringify(args || {})}) <Text dimColor>(MCP)</Text></Text>
                <Text dimColor>{context.description}</Text>
            </Box>

            <Text>Do you want to proceed?</Text>
            <PermissionSelect
                options={options}
                onChange={handleChange}
                onCancel={() => handleChange('no')}
            />
        </Box>
    );
}

// ik2 (BashConfirm? Or generic request?)
// Using a generic PermissionRequestView for now based on patterns
export function PermissionRequestView(props: any) {
    // Dispatch based on type
    if (props.type === 'mcp') {
        return <McpToolPermissionRequest {...props} />;
    }
    // Fallback to ToolUseConfirm
    return <ToolUseConfirm {...props} />;
}
