/**
 * File: src/components/wizard/AgentWizardSteps.tsx
 * Role: React components for the multi-step agent creation wizard.
 */

import React, { useState } from 'react';
import { Box, Text } from 'ink';
import { Card } from '../common/Card.js';

interface Tool {
    name: string;
}

interface ToolSelectionStepProps {
    availableTools: Tool[];
    selectedTools: string[];
    onToggle: (toolName: string) => void;
    onConfirm: () => void;
}

/**
 * Step 1: Select tools for the agent.
 */
export function ToolSelectionStep({ availableTools, selectedTools, onToggle: _onToggle, onConfirm: _onConfirm }: ToolSelectionStepProps) {
    return (
        <Card title="Step 1: Select Tools" borderColor="cyan">
            <Box flexDirection="column">
                {availableTools.map(tool => (
                    <Box key={tool.name}>
                        <Text color={selectedTools.includes(tool.name) ? "green" : "white"}>
                            {selectedTools.includes(tool.name) ? "●" : "○"} {tool.name}
                        </Text>
                    </Box>
                ))}
            </Box>
        </Card>
    );
}

interface ModelSelectionStepProps {
    models: string[];
    selectedModel: string;
    onSelect: (model: string) => void;
}

/**
 * Step 2: Choose reasoning model.
 */
export function ModelSelectionStep({ models, selectedModel, onSelect: _onSelect }: ModelSelectionStepProps) {
    return (
        <Card title="Step 2: select Model" borderColor="magenta">
            <Box flexDirection="column">
                {models.map(model => (
                    <Box key={model}>
                        <Text color={selectedModel === model ? "yellow" : "white"}>
                            {selectedModel === model ? ">" : " "} {model}
                        </Text>
                    </Box>
                ))}
            </Box>
        </Card>
    );
}

interface SystemPromptStepProps {
    value: string;
    onChange: (value: string) => void;
    onConfirm: () => void;
    onBack: () => void;
}

/**
 * Step 3: Enter system prompt.
 */
export function SystemPromptStep({ value, onChange: _onChange, onConfirm: _onConfirm, onBack: _onBack }: SystemPromptStepProps) {
    const [error] = useState<string | null>(null);

    return (
        <Card title="Step 3: System Prompt" borderColor="blue">
            <Box flexDirection="column" marginTop={1}>
                <Text>Enter the instructions for your agent:</Text>
                <Text dimColor>Be comprehensive for best results.</Text>

                <Box marginTop={1} borderStyle="round" borderColor="gray" paddingX={1}>
                    <Text>{value || "Type your prompt here..."}</Text>
                </Box>

                {error && (
                    <Box marginTop={1}>
                        <Text color="red">{error}</Text>
                    </Box>
                )}
            </Box>
        </Card>
    );
}

interface ColorSelectionStepProps {
    onSelect: (color: string) => void;
    onBack: () => void;
}

/**
 * Step 4: Choose agent theme color.
 */
export function ColorSelectionStep({ onSelect: _onSelect, onBack: _onBack }: ColorSelectionStepProps) {
    const colors = ["cyan", "magenta", "yellow", "blue", "green", "red"];

    return (
        <Card title="Step 4: Theme Color" borderColor="magenta">
            <Box flexDirection="column">
                <Text>Choose a visual theme for this agent:</Text>
                <Box marginTop={1} flexDirection="row" gap={2}>
                    {colors.map(color => (
                        <Box key={color} paddingX={1} borderStyle="round" borderColor={color}>
                            <Text color={color}>{color}</Text>
                        </Box>
                    ))}
                </Box>
            </Box>
        </Card>
    );
}

interface AgentMetadataStepProps {
    name: string;
    description: string;
    onChangeName: (name: string) => void;
    onChangeDescription: (desc: string) => void;
    onConfirm: () => void;
}

/**
 * Step 5: Finalize name and description.
 */
export function AgentMetadataStep({ name, description, onChangeName: _onChangeName, onChangeDescription: _onChangeDescription, onConfirm: _onConfirm }: AgentMetadataStepProps) {
    return (
        <Card title="Step 5: Agent Metadata" borderColor="green">
            <Box flexDirection="column" gap={1}>
                <Box flexDirection="column">
                    <Text bold>Name:</Text>
                    <Box borderStyle="single" borderColor="gray" paddingX={1}>
                        <Text>{name || "Agent name..."}</Text>
                    </Box>
                </Box>
                <Box flexDirection="column">
                    <Text bold>Description:</Text>
                    <Box borderStyle="single" borderColor="gray" paddingX={1}>
                        <Text>{description || "Brief description of what this agent does..."}</Text>
                    </Box>
                </Box>
            </Box>
        </Card>
    );
}
