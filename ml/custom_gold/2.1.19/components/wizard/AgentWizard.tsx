/**
 * File: src/components/wizard/AgentWizard.tsx
 * Role: Wizard for creating new custom agents.
 */

import React, { useState } from 'react';
import { Box, Text, useInput } from 'ink';
import SelectInput from 'ink-select-input';
import TextInput from 'ink-text-input';
import { Divider } from '../common/Divider.js';

interface AgentWizardProps {
    onExit: () => void;
    onSubmit: (data: any) => void;
}

type WizardStep = 'location' | 'method' | 'description';

export function AgentWizard({ onExit, onSubmit }: AgentWizardProps) {
    const [step, setStep] = useState<WizardStep>('location');
    const [location, setLocation] = useState<'project' | 'personal'>('project');
    const [method, setMethod] = useState<'generate' | 'manual'>('generate');
    const [description, setDescription] = useState('');

    const handleLocationSelect = (item: any) => {
        setLocation(item.value);
        setStep('method');
    };

    const handleMethodSelect = (item: any) => {
        setMethod(item.value);
        setStep('description');
    };

    useInput((input, key) => {
        if (key.escape) {
            onExit();
        }
    });

    const renderLocationStep = () => (
        <Box flexDirection="column" paddingX={1} width={80}>
            <Box borderStyle="round" borderColor="gray" flexDirection="column" padding={1}>
                <Text bold>Create new agent</Text>
                <Text dimColor>Choose location</Text>

                <Box marginTop={1}>
                    <SelectInput
                        items={[
                            { label: '1. Project (.claude/agents/)', value: 'project' },
                            { label: '2. Personal (~/.claude/agents/)', value: 'personal' }
                        ]}
                        onSelect={handleLocationSelect}
                        itemComponent={({ label, isSelected }) => (
                            <Text color={isSelected ? 'cyan' : 'white'}>
                                {isSelected ? '❯ ' : '  '}
                                {label}
                            </Text>
                        )}
                    />
                </Box>
            </Box>
            <Box marginTop={1}>
                <Text dimColor>↑↓ to navigate · Enter to select · Esc to cancel</Text>
            </Box>
        </Box>
    );

    const renderMethodStep = () => (
        <Box flexDirection="column" paddingX={1} width={80}>
            <Box borderStyle="round" borderColor="gray" flexDirection="column" padding={1}>
                <Text bold>Create new agent</Text>
                <Text dimColor>Creation method</Text>

                <Box marginTop={1}>
                    <SelectInput
                        items={[
                            { label: '1. Generate with Claude (recommended)', value: 'generate' },
                            { label: '2. Manual configuration', value: 'manual' }
                        ]}
                        onSelect={handleMethodSelect}
                        itemComponent={({ label, isSelected }) => (
                            <Text color={isSelected ? 'cyan' : 'white'}>
                                {isSelected ? '❯ ' : '  '}
                                {label}
                            </Text>
                        )}
                    />
                </Box>
            </Box>
            <Box marginTop={1}>
                <Text dimColor>↑↓ to navigate · Enter to select · Esc to go back</Text>
            </Box>
        </Box>
    );

    const renderDescriptionStep = () => (
        <Box flexDirection="column" paddingX={1} width={80}>
            <Box borderStyle="round" borderColor="gray" flexDirection="column" padding={1}>
                <Text bold>Create new agent</Text>
                <Text dimColor>Describe what this agent should do and when it should be used (be comprehensive for best results)</Text>

                <Box marginTop={1} flexDirection="column">
                    <Text dimColor>e.g., Help me write unit tests for my code...</Text>
                    <Box marginTop={1}>
                        <TextInput
                            value={description}
                            onChange={setDescription}
                            onSubmit={(val) => {
                                onSubmit({
                                    location,
                                    method,
                                    description: val
                                });
                            }}
                        />
                    </Box>
                </Box>
            </Box>
            <Box marginTop={1}>
                <Text dimColor>Enter to submit · Esc to go back</Text>
            </Box>
        </Box>
    );

    switch (step) {
        case 'location': return renderLocationStep();
        case 'method': return renderMethodStep();
        case 'description': return renderDescriptionStep();
        default: return null;
    }
}
