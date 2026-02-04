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
    onSelect: (agent: string) => void;
    onExit: () => void;
}

export function AgentsMenu({ onSelect, onExit }: AgentsMenuProps) {
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
            onSelect(item.agentType || item.value);
            onExit();
        }
    };

    const handleWizardSubmit = async (data: any) => {
        try {
            let name = data.description.split(' ').slice(0, 2).join('-').toLowerCase().replace(/[^a-z0-9-]/g, '');
            let systemPrompt = data.description;
            let description = data.description;

            if (data.method === 'generate') {
                const { generateAgent } = await import('../../services/agents/AgentDesigner.js');
                const design = await generateAgent(data.description, userAgents.map(a => a.value));
                name = design.identifier;
                systemPrompt = design.systemPrompt;
                description = design.whenToUse;
            }

            const newAgent: AgentData = {
                name,
                description,
                agentType: name,
                systemPrompt,
                model: 'claude-3-5-sonnet-20241022',
                scope: data.location
            };

            await saveAgent(newAgent, data.location);
            refreshAgents();
            setView('list');
        } catch (err) {
            // Error handling
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
