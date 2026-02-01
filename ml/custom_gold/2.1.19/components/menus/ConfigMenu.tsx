/**
 * File: src/components/menus/ConfigMenu.tsx
 * Role: Interactive configuration menu for changing app settings.
 */

import React, { useState } from 'react';
import { Box, Text, useInput } from 'ink';
import SelectInput from 'ink-select-input';
import { getSettings, updateSettings } from '../../services/config/SettingsService.js';
import { themeService } from '../../services/terminal/ThemeService.js';

interface ConfigMenuProps {
    onExit: () => void;
}

export function ConfigMenu({ onExit }: ConfigMenuProps) {
    const [settings, setSettings] = useState(getSettings());
    const [currentView, setCurrentView] = useState<'main' | 'theme' | 'telemetry'>('main');

    const handleSelect = (item: { label: string; value: string }) => {
        if (item.value === 'back') {
            setCurrentView('main');
        } else if (item.value === 'exit') {
            onExit();
        } else if (item.value === 'theme') {
            setCurrentView('theme');
        } else if (item.value === 'telemetry') {
            setCurrentView('telemetry');
        } else if (currentView === 'theme') {
            themeService.setTheme(item.value as any);
            setSettings(getSettings());
            setCurrentView('main');
        } else if (currentView === 'telemetry') {
            updateSettings({ telemetry: { enabled: item.value === 'enabled' } });
            setSettings(getSettings());
            setCurrentView('main');
        }
    };

    const mainItems = [
        { label: `Change Theme (${settings.theme || 'dark'})`, value: 'theme' },
        { label: `Toggle Telemetry (${settings.telemetry?.enabled ? 'ON' : 'OFF'})`, value: 'telemetry' },
        { label: 'Exit Config', value: 'exit' }
    ];

    const themeItems = [
        { label: 'Dark', value: 'dark' },
        { label: 'Light', value: 'light' },
        { label: 'Dark Daltonized', value: 'dark-daltonized' },
        { label: 'Light Daltonized', value: 'light-daltonized' },
        { label: 'Dark ANSI', value: 'dark-ansi' },
        { label: 'Light ANSI', value: 'light-ansi' },
        { label: 'Back', value: 'back' }
    ];


    const telemetryItems = [
        { label: 'Enable', value: 'enabled' },
        { label: 'Disable', value: 'disabled' },
        { label: 'Back', value: 'back' }
    ];

    useInput((input, key) => {
        if (key.escape) onExit();
    });

    return (
        <Box flexDirection="column" borderStyle="round" borderColor="cyan" padding={1} width={50}>
            <Text bold color="cyan">Configuration</Text>
            <Box marginTop={1}>
                {currentView === 'main' && (
                    <SelectInput items={mainItems} onSelect={handleSelect} />
                )}
                {currentView === 'theme' && (
                    <SelectInput items={themeItems} onSelect={handleSelect} />
                )}
                {currentView === 'telemetry' && (
                    <SelectInput items={telemetryItems} onSelect={handleSelect} />
                )}
            </Box>
            <Box marginTop={1}>
                <Text dimColor>Esc to go back/exit</Text>
            </Box>
        </Box>
    );
}
