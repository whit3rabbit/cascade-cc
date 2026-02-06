/**
 * File: src/components/permissions/PermissionDialog.tsx
 * Role: Dialog for Tool Permissions
 * Deobfuscated from chunk1518 and chunk1524
 */

import React from 'react';
import { Box, Text, useInput } from 'ink';
import { usePermissionDialog, UsePermissionDialogProps } from './usePermissionDialog.js';
import { PermissionOption } from './PermissionOption.js';


import { StructuredDiff } from '../StructuredDiff.js';

export const PermissionDialog: React.FC<UsePermissionDialogProps> = (props) => {
    const {
        options,
        onChange,
        selectedIndex,
        setSelectedIndex
    } = usePermissionDialog(props);

    const isFileEdit = props.toolUseConfirm.tool.name === 'edit_file' ||
        props.toolUseConfirm.tool.name === 'repl_replace' ||
        props.toolUseConfirm.tool.name === 'replace_file_content';

    let edits: { old_string: string; new_string: string; replace_all: boolean }[] = [];
    if (isFileEdit && props.toolUseConfirm.input) {
        const { old_string, new_string, replace_all } = props.toolUseConfirm.input;
        edits = [{ old_string, new_string, replace_all }];
    }

    useInput((input, key) => {
        if (key.downArrow || input === 'j') {
            setSelectedIndex((prev: number) => (prev + 1) % options.length);
        }
        if (key.upArrow || input === 'k') {
            setSelectedIndex((prev: number) => (prev - 1 + options.length) % options.length);
        }
        if (key.return) {
            onChange(options[selectedIndex].option);
        }
        if (key.escape) {
            props.onReject();
        }
    });

    return (
        <Box flexDirection="column" borderStyle="round" borderColor="yellow" paddingX={1} paddingY={0}>
            <Box paddingX={1} marginTop={-1}>
                <Text bold color="yellow"> Tool Permission </Text>
            </Box>

            <Box flexDirection="column" padding={1}>
                <Text>
                    Allow <Text bold color="cyan">{props.toolUseConfirm.tool.name}</Text> to execute?
                </Text>

                <Box marginY={1}>
                    {isFileEdit ? (
                        <Box flexDirection="column">
                            {edits.map((edit, i) => (
                                <StructuredDiff
                                    key={i}
                                    filePath={props.toolUseConfirm.input?.file_path || props.toolUseConfirm.input?.TargetFile}
                                    oldContent={edit.old_string}
                                    newContent={edit.new_string}
                                />
                            ))}
                        </Box>
                    ) : (
                        <Box borderStyle="single" borderColor="gray" paddingX={1}>
                            <Text dimColor>{JSON.stringify(props.toolUseConfirm.input, null, 2)}</Text>
                        </Box>
                    )}
                </Box>

                <Box flexDirection="column" marginTop={0}>
                    {options.map((opt: any, i: number) => (
                        <PermissionOption
                            key={i}
                            label={opt.label}
                            value={opt.option.type}
                            isFocused={i === selectedIndex}
                            onSelect={() => onChange(opt.option)}
                            shortcut={i === 0 ? 'Enter' : undefined}
                            description={opt.option.description}
                        />
                    ))}
                </Box>
            </Box>

            <Box borderStyle="single" borderTop={false} borderBottom={false} borderLeft={false} borderRight={false} paddingX={1} marginBottom={-1}>
                <Text dimColor> Use ↑↓ to navigate · Enter to select · Esc to deny </Text>
            </Box>
        </Box>
    );
};
