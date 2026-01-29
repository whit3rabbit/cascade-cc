/**
 * File: src/components/permissions/PermissionDialog.tsx
 * Role: Dialog for Tool Permissions
 * Deobfuscated from chunk1518 and chunk1524
 */

import React, { useState, useEffect } from 'react';
import { Box, Text, useInput } from 'ink';
import { usePermissionDialog, UsePermissionDialogProps, PermissionOptionValue } from './usePermissionDialog.js';
import { PermissionOption } from './PermissionOption.js';

// Simple Diff View Placeholder
const FileDiffView = ({ filePath, edits }: { filePath?: string, edits: any[] }) => {
    return (
        <Box flexDirection="column" borderStyle="single" borderColor="gray" paddingX={1}>
            <Text underline>{filePath || 'Unknown file'}</Text>
            {edits.map((edit, i) => (
                <Box key={i} flexDirection="column" marginTop={1}>
                    <Text color="red">- {edit.old_string?.slice(0, 200).replace(/\n/g, '↵') || ''}</Text>
                    <Text color="green">+ {edit.new_string?.slice(0, 200).replace(/\n/g, '↵') || ''}</Text>
                </Box>
            ))}
        </Box>
    )
}

export const PermissionDialog: React.FC<UsePermissionDialogProps> = (props) => {
    const {
        options,
        onChange
    } = usePermissionDialog(props);

    const [selectedIndex, setSelectedIndex] = useState(0);

    const isFileEdit = props.toolUseConfirm.tool.name === 'edit_file' || props.toolUseConfirm.tool.name === 'repl_replace';

    let edits: { old_string: string; new_string: string; replace_all: boolean }[] = [];
    if (isFileEdit && props.toolUseConfirm.input) {
        const { file_path, old_string, new_string, replace_all } = props.toolUseConfirm.input;
        edits = [{ old_string, new_string, replace_all }];
    }

    useInput((input, key) => {
        if (key.downArrow || input === 'j') {
            setSelectedIndex((prev) => (prev + 1) % options.length);
        }
        if (key.upArrow || input === 'k') {
            setSelectedIndex((prev) => (prev - 1 + options.length) % options.length);
        }
        if (key.return) {
            const selected = options[selectedIndex];
            if (selected) {
                onChange(selected.option, "allowed");
            }
        }
        if (key.escape) {
            props.onReject();
        }
    });

    return (
        <Box flexDirection="column" borderStyle="round" borderColor="yellow" padding={1}>
            <Text bold color="yellow">
                {isFileEdit ? "Confirm File Edit" : "Tool Permission Required"}
            </Text>

            <Text>
                Allow <Text bold color="cyan">{props.toolUseConfirm.tool.name}</Text> to execute?
            </Text>

            <Box marginY={1}>
                {isFileEdit ? (
                    <FileDiffView filePath={props.toolUseConfirm.input?.file_path} edits={edits} />
                ) : (
                    <Box borderStyle="single" borderColor="gray" paddingX={1}>
                        <Text dimColor>{JSON.stringify(props.toolUseConfirm.input, null, 2)}</Text>
                    </Box>
                )}
            </Box>

            <Box flexDirection="column" marginTop={1}>
                {options.map((opt: { option: PermissionOptionValue; label: string }, i: number) => (
                    <PermissionOption
                        key={i}
                        label={opt.label}
                        value={opt.option.type}
                        isFocused={i === selectedIndex}
                        onSelect={() => onChange(opt.option, "allowed")}
                        shortcut={i === 0 ? 'Enter' : undefined}
                    />
                ))}
            </Box>

            <Box marginTop={1}>
                <Text dimColor>Press Enter to confirm, Esc to reject</Text>
            </Box>
        </Box>
    );
};
