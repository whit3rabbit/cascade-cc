/**
 * File: src/components/ModelPicker/ModelPicker.tsx
 * Role: UI component for selecting and switching Claude models.
 */

import React, { useState, useMemo } from 'react';
import { Box, Text } from 'ink';
import { Select } from '../common/Select.js';
import { Divider } from '../common/Divider.js';
import { ActionHint } from '../common/ActionHint.js';

interface ModelOption {
    label: string;
    value: string;
    description: string;
}

interface ModelPickerProps {
    initialModel: string | null;
    sessionModel?: string | null;
    onSelect: (model: string | null) => void;
    onCancel: () => void;
    isStandalone?: boolean;
}

const DEFAULT_MODELS: ModelOption[] = [
    {
        value: "claude-3-5-sonnet-20241022",
        label: "1. Default (recommended) ✔  Sonnet 4.5 · Best for everyday tasks",
        description: "Best for everyday tasks"
    },
    {
        value: "claude-3-opus-20240229",
        label: "2. Opus                     Opus 4.5 · Most capable for complex work",
        description: "Most capable for complex work"
    },
    {
        value: "claude-3-5-haiku-20241022",
        label: "3. Haiku                    Haiku 4.5 · Fastest for quick answers",
        description: "Fastest for quick answers"
    }
];

/**
 * Model selection UI that allows users to pick a different Claude model for their session.
 */
export function ModelPicker({
    initialModel,
    sessionModel,
    onSelect,
    onCancel,
    isStandalone = false
}: ModelPickerProps) {
    const [activeValue, setActiveValue] = useState(initialModel || DEFAULT_MODELS[0].value);

    const options = useMemo(() => {
        // We generally use the DEFAULT_MODELS. If initialModel is not in it, we might want to add it,
        // but for this specific TUI request we mainly want to show the list as requested.
        // We will stick to DEFAULT_MODELS mostly.
        const list = [...DEFAULT_MODELS];
        if (initialModel && !list.find(m => m.value === initialModel)) {
            list.push({
                value: initialModel,
                label: `4. Custom                   ${initialModel}`,
                description: "Current model"
            });
        }
        return list;
    }, [initialModel]);

    return (
        <Box flexDirection="column" paddingX={isStandalone ? 1 : 0}>
            {isStandalone && <Divider color="gray" marginBottom={1} />}

            <Box flexDirection="column" marginBottom={1}>
                <Text bold>Select model</Text>
                <Text dimColor>
                    Switch between Claude models. Applies to this session and future Claude Code sessions. For other/previous model names, specify with --model.
                </Text>
                {sessionModel && (
                    <Text dimColor>
                        Currently using {sessionModel} for this session (set by plan mode). Selecting a model will undo this.
                    </Text>
                )}
            </Box>

            <Select
                options={options}
                defaultValue={initialModel || undefined}
                onChange={(value) => onSelect(value)}
                onCancel={onCancel}
                visibleOptionCount={5}
            />

            <Box marginTop={1} flexDirection="row">
                <Text dimColor>Enter to confirm · Esc to exit</Text>
            </Box>
        </Box>
    );
}

export default ModelPicker;
