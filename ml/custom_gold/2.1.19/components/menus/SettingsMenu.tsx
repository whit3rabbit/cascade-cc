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
import { getAuthDetails } from '../../services/auth/AuthService.js';
import { costService } from '../../services/terminal/CostService.js';
import { BUILD_INFO } from '../../constants/build.js';

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
    const [dateRange, setDateRange] = useState<'all' | '7d' | '30d'>('all');

    // Load status info on mount
    useEffect(() => {
        const loadInfo = async () => {
            const auth = await getAuthDetails();
            const clients = mcpClientManager.getActiveClients();
            setStatusInfo({
                version: BUILD_INFO.VERSION,
                sessionName: '/rename to add a name', // Placeholder
                sessionId: EnvService.get('CLAUDE_SESSION_ID') || 'unknown',
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
        } else if (input === 'r' && activeTab === 'Usage') {
            cycleDateRange();
        }
    });

    const cycleDateRange = () => {
        const ranges: ('all' | '7d' | '30d')[] = ['all', '7d', '30d'];
        const currentIndex = ranges.indexOf(dateRange);
        const nextIndex = (currentIndex + 1) % ranges.length;
        setDateRange(ranges[nextIndex]);
    };

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
            { label: 'Auto-compact', value: 'autoCompact', type: 'boolean', current: settings.autoCompact ?? true },
            { label: 'Show tips', value: 'showTips', type: 'boolean', current: settings.showTips ?? true },
            { label: 'Thinking mode', value: 'thinkingMode', type: 'boolean', current: settings.thinkingMode ?? true },
            { label: 'Prompt suggestions', value: 'promptSuggestions', type: 'boolean', current: settings.promptSuggestions ?? true },
            { label: 'Rewind code (checkpoints)', value: 'rewindCode', type: 'boolean', current: settings.rewindCode ?? true },
            { label: 'Verbose output', value: 'verbose', type: 'boolean', current: settings.verbose ?? false },
            { label: 'Terminal progress bar', value: 'progressBar', type: 'boolean', current: settings.progressBar ?? true },
            { label: 'Default permission mode', value: 'permissionMode', type: 'select', current: settings.permissionMode || 'Default' },
            { label: 'Respect .gitignore in file picker', value: 'gitignore', type: 'boolean', current: settings.gitignore ?? true },
            { label: 'Auto-update channel', value: 'updateChannel', type: 'select', current: settings.updateChannel || 'latest' },
            { label: 'Theme', value: 'theme', type: 'select', current: settings.theme || 'dark' },
            { label: 'Vim Mode', value: 'vimModeEnabled', type: 'boolean', current: settings.vimModeEnabled || false },
            { label: 'Notifications', value: 'notifications', type: 'select', current: settings.notifications || 'Auto' },
            { label: 'Output style', value: 'outputStyle', type: 'select', current: settings.outputStyle || 'default' },
            { label: 'Language', value: 'language', type: 'select', current: settings.language || 'Default (English)' },
            { label: 'Editor mode', value: 'editorMode', type: 'select', current: settings.editorMode || 'normal' },
            { label: 'Show code diff footer', value: 'showCodeDiffFooter', type: 'boolean', current: settings.showCodeDiffFooter ?? true },
            { label: 'Show PR status footer', value: 'showPrStatusFooter', type: 'boolean', current: settings.showPrStatusFooter ?? true },
            { label: 'Model', value: 'model', type: 'select', current: settings.model || 'Default (recommended)' },
            { label: 'Auto-connect to IDE (external terminal)', value: 'autoConnectIde', type: 'boolean', current: settings.autoConnectIde ?? true },
            { label: 'Claude in Chrome enabled by default', value: 'chromeEnabled', type: 'boolean', current: settings.chromeEnabled ?? true },
        ];
    }, [settings]);

    const filteredItems = useMemo(() => {
        if (!searchQuery) return configItems;
        return configItems.filter(item => item.label.toLowerCase().includes(searchQuery.toLowerCase()));
    }, [configItems, searchQuery]);

    const handleConfigSelect = (item: any) => {
        const configItem = configItems.find(i => i.value === item.value);
        if (!configItem) return;

        // Toggle boolean
        if (configItem.type === 'boolean') {
            const newValue = !configItem.current;
            updateSettings({ [configItem.value]: newValue });
            setSettings(getSettings());
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
                        let valueDisplay = item?.current.toString();
                        if (item?.type === 'boolean') {
                            valueDisplay = item.current ? '[X]' : '[ ]';
                        }
                        return (
                            <Box justifyContent="space-between" width="100%">
                                <Text color={isSelected ? 'cyan' : 'white'}>{label}</Text>
                                <Box>
                                    <Text color={isSelected ? 'cyan' : 'gray'}>{valueDisplay}</Text>
                                </Box>
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

    const renderUsageTab = () => {
        const usageData = costService.getUsage();
        const cost = costService.calculateCost();
        const dateRangeLabels = {
            all: 'All time',
            '7d': 'Last 7 days',
            '30d': 'Last 30 days'
        };

        return (
            <Box flexDirection="column" paddingX={2}>
                <Box marginBottom={1} justifyContent="space-between">
                    <Text bold underline>Session Usage Details</Text>
                    <Text color="gray">{dateRangeLabels[dateRange]}</Text>
                </Box>
                <Box flexDirection="column">
                    <Box justifyContent="space-between">
                        <Text>Input Tokens:</Text>
                        <Text color="cyan">{usageData.inputTokens.toLocaleString()}</Text>
                    </Box>
                    <Box justifyContent="space-between">
                        <Text>Output Tokens:</Text>
                        <Text color="cyan">{usageData.outputTokens.toLocaleString()}</Text>
                    </Box>
                    <Box justifyContent="space-between">
                        <Text>Cache Write Tokens:</Text>
                        <Text color="cyan">{(usageData.cacheWriteTokens || 0).toLocaleString()}</Text>
                    </Box>
                    <Box justifyContent="space-between">
                        <Text>Cache Read Tokens:</Text>
                        <Text color="cyan">{(usageData.cacheReadTokens || 0).toLocaleString()}</Text>
                    </Box>

                    {dateRange !== 'all' && (
                        <Box marginTop={1} flexDirection="column">
                            <Text bold>Tokens per Day</Text>
                            <Text dimColor> (Historical data not available in this session)</Text>
                        </Box>
                    )}

                    <Box marginTop={1} borderStyle="single" borderColor="green" paddingX={1} justifyContent="space-between">
                        <Text bold>Estimated Cost:</Text>
                        <Text bold color="green">${cost.toFixed(4)}</Text>
                    </Box>
                </Box>
                <Box marginTop={1}>
                    <Text dimColor>r to cycle dates · Rates vary by model. Calculations are estimates.</Text>
                </Box>
            </Box>
        );
    };

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
