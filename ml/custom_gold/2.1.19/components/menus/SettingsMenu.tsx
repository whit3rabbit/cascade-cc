/**
 * File: src/components/menus/SettingsMenu.tsx
 * Role: Tabbed Settings/Status interface matching the TUI design.
 */

import React, { useState, useMemo, useEffect } from 'react';
import { Box, Text, useInput } from 'ink';
import SelectInput from 'ink-select-input';
import TextInput from 'ink-text-input';
import { EnvService } from '../../services/config/EnvService.js';
import { mcpClientManager } from '../../services/mcp/McpClientManager.js';
import { getSettings, updateSettings } from '../../services/config/SettingsService.js';
import { themeService } from '../../services/terminal/ThemeService.js';
import { getAuthDetails } from '../../services/auth/AuthService.js';
import os from 'os';

interface SettingsMenuProps {
    onExit: () => void;
    initialTab?: Tab;
}

type Tab = 'Status' | 'Config' | 'Usage';

export function SettingsMenu({ onExit, initialTab = 'Status' }: SettingsMenuProps) {
    const [activeTab, setActiveTab] = useState<Tab>(initialTab);
    const [settings, setSettings] = useState(getSettings());
    const [searchQuery, setSearchQuery] = useState('');
    const [statusInfo, setStatusInfo] = useState<any>({});

    // Load status info on mount
    useEffect(() => {
        const loadInfo = async () => {
            const auth = await getAuthDetails();
            const clients = mcpClientManager.getActiveClients();
            setStatusInfo({
                version: '2.1.27', // Matching user request
                sessionName: '/rename to add a name', // Placeholder
                sessionId: 'cddd10df-23a5-4a75-aad6-6af8f7dba417', // Placeholder or generate real uuid
                cwd: process.cwd(),
                loginMethod: (auth as any).email ? `Claude API Account (${(auth as any).email})` : 'Claude API Account',
                model: 'Default Sonnet 4.5 · Best for everyday tasks',
                mcpServers: clients.length > 0 ? clients.join(', ') : 'No active servers',
                memory: 'user (~/.claude/CLAUDE.md), project (CLAUDE.md)',
                settingSources: 'User settings, Project local settings'
            });
        };
        loadInfo();
    }, []);

    useInput((input, key) => {
        if (key.escape) {
            onExit();
        }

        if (key.leftArrow) {
            cycleTab(-1);
        } else if (key.rightArrow || key.tab) {
            cycleTab(1);
        }
    });

    const cycleTab = (direction: number) => {
        const tabs: Tab[] = ['Status', 'Config', 'Usage'];
        const currentIndex = tabs.indexOf(activeTab);
        let nextIndex = (currentIndex + direction) % tabs.length;
        if (nextIndex < 0) nextIndex = tabs.length - 1;
        setActiveTab(tabs[nextIndex]);
    };

    const renderTabs = () => (
        <Box marginBottom={1} borderStyle="single" borderTop={false} borderLeft={false} borderRight={false} borderColor="gray">
            <Text>  </Text>
            <Text color={activeTab === 'Status' ? 'white' : 'gray'} bold={activeTab === 'Status'}> Status </Text>
            <Text color="gray">  </Text>
            <Text color={activeTab === 'Config' ? 'white' : 'gray'} bold={activeTab === 'Config'}> Config </Text>
            <Text color="gray">  </Text>
            <Text color={activeTab === 'Usage' ? 'white' : 'gray'} bold={activeTab === 'Usage'}> Usage </Text>
            <Text color="gray">  (←/→ or tab to cycle)</Text>
        </Box>
    );

    const renderStatusTab = () => (
        <Box flexDirection="column" paddingX={2}>
            <Box flexDirection="column" marginBottom={1}>
                <Text>Version: {statusInfo.version}</Text>
                <Text>Session name: {statusInfo.sessionName}</Text>
                <Text>Session ID: {statusInfo.sessionId}</Text>
                <Text>cwd: {statusInfo.cwd}</Text>
                <Text>Login method: {statusInfo.loginMethod}</Text>
            </Box>

            <Box flexDirection="column" marginBottom={1}>
                <Text>Model: {statusInfo.model}</Text>
                <Text>MCP servers: {statusInfo.mcpServers}</Text>
                <Text>Memory: {statusInfo.memory}</Text>
                <Text>Setting sources: {statusInfo.settingSources}</Text>
            </Box>

            <Text dimColor>Esc to cancel</Text>
        </Box>
    );

    const configItems = useMemo(() => {
        // Map user requested items to actual settings or placeholders
        return [
            { label: 'Auto-compact', value: 'autoCompact', type: 'boolean', current: true },
            { label: 'Show tips', value: 'showTips', type: 'boolean', current: true },
            { label: 'Thinking mode', value: 'thinkingMode', type: 'boolean', current: true },
            { label: 'Prompt suggestions', value: 'promptSuggestions', type: 'boolean', current: true },
            { label: 'Rewind code (checkpoints)', value: 'rewindCode', type: 'boolean', current: true },
            { label: 'Verbose output', value: 'verbose', type: 'boolean', current: false },
            { label: 'Terminal progress bar', value: 'progressBar', type: 'boolean', current: true },
            { label: 'Default permission mode', value: 'permissionMode', type: 'select', current: 'Default' },
            { label: 'Respect .gitignore in file picker', value: 'gitignore', type: 'boolean', current: true },
            { label: 'Auto-update channel', value: 'updateChannel', type: 'select', current: 'latest' },
            { label: 'Theme', value: 'theme', type: 'select', current: settings.theme || 'dark' },
            { label: 'Notifications', value: 'notifications', type: 'select', current: 'Auto' },
            { label: 'Output style', value: 'outputStyle', type: 'select', current: 'default' },
            { label: 'Language', value: 'language', type: 'select', current: 'Default (English)' },
            { label: 'Editor mode', value: 'editorMode', type: 'select', current: 'normal' },
            { label: 'Show code diff footer', value: 'showCodeDiffFooter', type: 'boolean', current: true },
            { label: 'Show PR status footer', value: 'showPrStatusFooter', type: 'boolean', current: true },
            { label: 'Model', value: 'model', type: 'select', current: 'Default (recommended)' },
            { label: 'Auto-connect to IDE (external terminal)', value: 'autoConnectIde', type: 'boolean', current: true },
            { label: 'Claude in Chrome enabled by default', value: 'chromeEnabled', type: 'boolean', current: true },
        ];
    }, [settings]);

    const filteredItems = useMemo(() => {
        if (!searchQuery) return configItems;
        return configItems.filter(item => item.label.toLowerCase().includes(searchQuery.toLowerCase()));
    }, [configItems, searchQuery]);

    const handleConfigSelect = (item: any) => {
        // Toggle boolean
        if (item.type === 'boolean') {
            // In a real implementation we'd update settings
            // setSettings(prev => ({ ...prev, [item.value]: !item.current }));
        }
    };

    const renderConfigTab = () => (
        <Box flexDirection="column" paddingX={2}>
            <Box marginBottom={1}>
                <Text bold>Configure Claude Code preferences</Text>
            </Box>

            <Box borderStyle="round" borderColor="gray" marginBottom={1} paddingX={1}>
                <Text>⌕ </Text>
                <TextInput
                    value={searchQuery}
                    onChange={setSearchQuery}
                    placeholder="Search settings..."
                    focus={activeTab === 'Config'}
                />
            </Box>

            <Box flexDirection="column" height={15}>
                <SelectInput
                    items={filteredItems.map(item => ({
                        label: item.label,
                        value: item.value,
                        key: item.value
                    }))}
                    itemComponent={({ label, isSelected }) => {
                        const item = configItems.find(i => i.label === label);
                        const valueDisplay = item?.current.toString();
                        return (
                            <Box justifyContent="space-between" width="100%">
                                <Text color={isSelected ? 'cyan' : 'white'}>{label}</Text>
                                <Text color={isSelected ? 'cyan' : 'gray'}>{valueDisplay}</Text>
                            </Box>
                        );
                    }}
                    onSelect={handleConfigSelect}
                    limit={15}
                />
            </Box>

            <Box marginTop={1}>
                <Text dimColor>Type to filter · Enter/↓ to select · Esc to clear</Text>
            </Box>
        </Box>
    );

    const renderUsageTab = () => (
        <Box flexDirection="column" paddingX={2}>
            <Text>Usage statistics not available in this view yet.</Text>
            <Text>Use /cost for current session details.</Text>
        </Box>
    );

    return (
        <Box flexDirection="column" width="100%" height="100%">
            {renderTabs()}

            {activeTab === 'Status' && renderStatusTab()}
            {activeTab === 'Config' && renderConfigTab()}
            {activeTab === 'Usage' && renderUsageTab()}
        </Box>
    );
}

export default SettingsMenu;
