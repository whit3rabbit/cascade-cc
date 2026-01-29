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
    { value: "claude-3-5-sonnet-20241022", label: "Claude 3.5 Sonnet", description: "Best overall performance" },
    { value: "claude-3-opus-20240229", label: "Claude 3 Opus", description: "Most powerful model" },
    { value: "claude-3-5-haiku-20241022", label: "Claude 3.5 Haiku", description: "Fastest and most efficient" }
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
        // Ensure the initial model is in the list if it's custom
        const list = [...DEFAULT_MODELS];
        if (initialModel && !list.find(m => m.value === initialModel)) {
            list.push({
                value: initialModel,
                label: initialModel,
                description: "Current model"
            });
        }
        return list;
    }, [initialModel]);

    return (
        <Box flexDirection="column" paddingX={isStandalone ? 1 : 0}>
            {isStandalone && <Divider color="blue" marginBottom={1} />}

            <Box flexDirection="column" marginBottom={1}>
                <Text bold color="cyan">Select model</Text>
                <Text dimColor>
                    Switch between Claude models. Applies to this session and future Claude Code sessions.
                </Text>
                {sessionModel && (
                    <Text color="yellow">
                        Currently using {sessionModel} (set by project config or environment).
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

            {isStandalone && (
                <Box marginTop={1} flexDirection="row">
                    <ActionHint shortcut="Enter" action="confirm" />
                    <ActionHint shortcut="Esc" action="cancel" />
                </Box>
            )}
        </Box>
    );
}

export default ModelPicker;
