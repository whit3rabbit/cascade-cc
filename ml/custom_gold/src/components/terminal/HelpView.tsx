
// Logic from chunk_544.ts (Help Command UI)

import React, { useMemo } from "react";
import { Box, Text, useInput, useStdout } from "ink";
import { PermissionSelect } from "../permissions/PermissionComponents.js";
import { Tabs, Tab } from "./PermissionsManager.js";
import ExternalLink from "./ExternalLink.js";
import { useCtrlExit } from "../../hooks/useCtrlExit.js";

// --- Shortcuts View (X59) ---
export function HelpShortcutsView({ onCancel }: any) {
    useInput((_input, key) => {
        if (key.escape) onCancel();
    });

    return (
        <Box flexDirection="column" paddingY={1} gap={1}>
            <Box>
                <Text>Claude understands your codebase, makes edits with your permission, and executes commands — right from your terminal.</Text>
            </Box>
            <Box flexDirection="column">
                <Text bold>Shortcuts</Text>

                <Box flexDirection="column" marginTop={1}>
                    <Text bold dimColor>General</Text>
                    <Box marginLeft={2} flexDirection="column">
                        <Text>• <Text bold>Enter</Text>: Send message</Text>
                        <Text>• <Text bold>Esc</Text>: Cancel / Back</Text>
                        <Text>• <Text bold>Ctrl+C</Text>: Cancel response / Abort</Text>
                        <Text>• <Text bold>Ctrl+L</Text>: Clear screen</Text>
                        <Text>• <Text bold>Ctrl+D</Text>: Exit</Text>
                    </Box>
                </Box>

                <Box flexDirection="column" marginTop={1}>
                    <Text bold dimColor>Editing</Text>
                    <Box marginLeft={2} flexDirection="column">
                        <Text>• <Text bold>Alt+Enter</Text>: Insert newline</Text>
                        <Text>• <Text bold>Ctrl+A</Text>: Go to start of line</Text>
                        <Text>• <Text bold>Ctrl+E</Text>: Go to end of line</Text>
                        <Text>• <Text bold>Ctrl+K</Text>: Cut to end of line</Text>
                        <Text>• <Text bold>Ctrl+U</Text>: Cut to start of line</Text>
                    </Box>
                </Box>

                <Box flexDirection="column" marginTop={1}>
                    <Text bold dimColor>History</Text>
                    <Box marginLeft={2} flexDirection="column">
                        <Text>• <Text bold>↑ / ↓</Text>: Navigate history</Text>
                        <Text>• <Text bold>Ctrl+R</Text>: Search history</Text>
                    </Box>
                </Box>
            </Box>
        </Box>
    );
}

// --- Commands List View (RN0) ---
export function HelpCommandsView({ commands, maxHeight, title, onCancel, emptyMessage }: any) {
    const visibleCount = Math.max(1, Math.floor((maxHeight - 6) / 2));

    const options = useMemo(() => {
        return [...commands]
            .sort((a, b) => a.name.localeCompare(b.name))
            .map(cmd => ({
                label: `/${cmd.name}`,
                value: cmd.name,
                description: cmd.description
            }));
    }, [commands]);

    return (
        <Box flexDirection="column" paddingY={1}>
            {commands.length === 0 && emptyMessage ? (
                <Text dimColor>{emptyMessage}</Text>
            ) : (
                <>
                    <Text>{title}</Text>
                    <Box marginTop={1}>
                        <PermissionSelect
                            options={options}
                            onChange={() => { }}
                            onCancel={onCancel}
                            isDisabled={true}
                            defaultValue={options[0]?.value}
                        />
                    </Box>
                    <Box marginTop={1}>
                        <Text dimColor>{visibleCount} shown per page</Text>
                    </Box>
                </>
            )}
        </Box>
    );
}

// --- Main Help View (V59) ---
export function HelpView({ onClose, commands = [] }: any) {
    const { stdout } = useStdout();
    const rows = stdout?.rows || 24;
    const maxHeight = Math.floor(rows / 2);

    const handleClose = () => onClose("Help dialog dismissed", { display: "system" });
    const exitState = useCtrlExit(handleClose);

    // Separate default and custom commands
    const defaultCommandNames = new Set(["help", "compact", "config", "cost", "bug", "init", "logout", "review", "undo"]);
    const defaultCommands = commands.filter((c: any) => defaultCommandNames.has(c.name) && !c.isHidden);
    const customCommands = commands.filter((c: any) => !defaultCommandNames.has(c.name) && !c.isHidden);

    return (
        <Box flexDirection="column" height={maxHeight}>
            <Box>
                <Text>────────────────────────────────────────────</Text>
            </Box>
            <Box flexDirection="column" paddingX={1}>
                <Tabs title={`Claude Code v${"2.0.76"}`} color="professionalBlue">
                    <Tab title="general">
                        <HelpShortcutsView onCancel={handleClose} />
                    </Tab>
                    <Tab title="commands">
                        <HelpCommandsView
                            commands={defaultCommands}
                            maxHeight={maxHeight}
                            title="Browse default commands:"
                            onCancel={handleClose}
                        />
                    </Tab>
                    <Tab title="custom-commands">
                        <HelpCommandsView
                            commands={customCommands}
                            maxHeight={maxHeight}
                            title="Browse custom commands:"
                            emptyMessage="No custom commands found"
                            onCancel={handleClose}
                        />
                    </Tab>
                </Tabs>

                <Box marginTop={1}>
                    <Text>
                        For more help: <ExternalLink url="https://code.claude.com/docs/en/overview" />
                    </Text>
                </Box>

                <Box marginTop={1}>
                    <Text dimColor>
                        {exitState.pending ? `Press ${exitState.keyName} again to exit` : "Esc to cancel"}
                    </Text>
                </Box>
            </Box>
        </Box>
    );
}
