/**
 * File: src/components/menus/AgentsMenu.tsx
 * Role: Main menu for managing and creating agents.
 */

import React, { useState, useEffect } from 'react';
import { Box, Text, useInput } from 'ink';
import SelectInput from 'ink-select-input';
import { AgentWizard } from '../wizard/AgentWizard.js';
import { saveAgent, listAgents, AgentData } from '../../services/agents/AgentPersistence.js';

interface AgentsMenuProps {
    onExit: () => void;
}

export function AgentsMenu({ onExit }: AgentsMenuProps) {
    const [view, setView] = useState<'list' | 'create'>('list');
    const [userAgents, setUserAgents] = useState<any[]>([]);

    // Load agents on mount
    useEffect(() => {
        refreshAgents();
    }, []);

    const refreshAgents = () => {
        const loaded = listAgents().map(a => ({
            label: `${a.name} · ${a.model || 'inherit'}`,
            value: a.agentType,
            ...a
        }));
        setUserAgents(loaded);
    };

    const builtinAgents = [
        { label: 'Bash · inherit', value: 'bash', agentType: 'bash', description: 'Executes terminal commands' },
        { label: 'General Purpose · inherit', value: 'general-purpose', agentType: 'general-purpose', description: 'Default assistant' },
        { label: 'Explore · haiku', value: 'explore', agentType: 'explore', description: 'Read-only codebase exploration' },
        { label: 'Plan · inherit', value: 'plan', agentType: 'plan', description: 'Planning and research' },
        { label: 'Claude Code Guide · haiku', value: 'code-guide', agentType: 'code-guide', description: 'Help with Claude Code features' }
    ];

    const handleSelect = (item: any) => {
        if (item.value === 'create_new') {
            setView('create');
        } else {
            // In a real app, this would select the agent for the session
            // For now we just exit, but you could add a callback prop to set the agent
            onExit();
        }
    };

    const handleWizardSubmit = async (data: any) => {
        try {
            // If method is 'generate', we would ideally call an LLM here.
            // For this implementation, we'll create a stub or just pass the description as system prompt for now if generated.
            // In a real "generate" flow, we'd need an async loader. Only "manual" is fully supported logic-wise without LLM calls.

            const newAgent: AgentData = {
                name: data.description.split(' ').slice(0, 2).join('-').toLowerCase().replace(/[^a-z0-9-]/g, ''),
                description: data.description,
                agentType: data.description.split(' ').slice(0, 2).join('-').toLowerCase().replace(/[^a-z0-9-]/g, ''), // simplified slug
                systemPrompt: data.method === 'generate'
                    ? `You are an agent designed to: ${data.description}`
                    : data.description, // Manual mode assumes description IS usage instructions for this demo
                model: 'claude-3-5-sonnet-20241022', // Default
                scope: data.location
            };

            await saveAgent(newAgent, data.location);
            refreshAgents();
            setView('list');
        } catch (err) {
            // console.error("Failed to save agent", err);
        }
    };

    useInput((input, key) => {
        if (key.escape && view === 'list') {
            onExit();
        }
    });

    if (view === 'create') {
        return <AgentWizard onExit={() => setView('list')} onSubmit={handleWizardSubmit} />;
    }

    // List View - Custom Render for Exact TUI Match
    const [selectedIndex, setSelectedIndex] = useState(0);

    // Flatten list for index mapping: 
    // 0: Create New
    // 1..N: User Agents
    // N+1..M: Built-in Agents
    const flattenedItems = [
        { type: 'action', label: 'Create new agent', value: 'create_new' }, // Create new
        ...userAgents.map(a => ({ type: 'user', ...a })),
        ...builtinAgents.map(a => ({ type: 'builtin', ...a }))
    ];

    useInput((input, key) => {
        if (view === 'list') {
            if (key.upArrow) {
                setSelectedIndex(prev => Math.max(0, prev - 1));
            }
            if (key.downArrow) {
                setSelectedIndex(prev => Math.min(flattenedItems.length - 1, prev + 1));
            }
            if (key.return) {
                const item = flattenedItems[selectedIndex];
                handleSelect(item);
            }
        }
    });

    return (
        <Box flexDirection="column" borderStyle="round" borderColor="white" paddingX={1} width={80}>
            <Text bold>Agents</Text>
            <Text dimColor>{userAgents.length + builtinAgents.length} agents</Text>

            <Box marginTop={1} flexDirection="column">
                {/* Create New Agent */}
                <Box marginBottom={1}>
                    <Text color={selectedIndex === 0 ? 'cyan' : 'white'} bold>
                        {selectedIndex === 0 ? '❯ ' : '  '}
                        Create new agent
                    </Text>
                </Box>

                {/* User Agents Section */}
                <Text dimColor>User agents</Text>
                {userAgents.length === 0 && <Text dimColor>  No custom agents found</Text>}
                {userAgents.map((agent, i) => {
                    const globalIndex = 1 + i;
                    const isSelected = selectedIndex === globalIndex;
                    return (
                        <Text key={agent.value} color={isSelected ? 'cyan' : 'white'}>
                            {isSelected ? '❯ ' : '  '}
                            {agent.label}
                        </Text>
                    );
                })}

                {/* Built-in Agents Section */}
                <Box marginTop={1}>
                    <Text dimColor>Built-in agents (always available)</Text>
                </Box>
                {builtinAgents.map((agent, i) => {
                    const globalIndex = 1 + userAgents.length + i;
                    const isSelected = selectedIndex === globalIndex;
                    return (
                        <Text key={agent.value} color={isSelected ? 'cyan' : 'white'}>
                            {isSelected ? '❯ ' : '  '}
                            {agent.label}
                        </Text>
                    );
                })}
            </Box>

            <Box marginTop={1}>
                <Text dimColor>Press ↑↓ to navigate · Enter to select · Esc to go back</Text>
            </Box>
        </Box>
    );
}
