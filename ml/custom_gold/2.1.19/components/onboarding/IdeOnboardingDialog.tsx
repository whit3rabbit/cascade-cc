import React, { useState, useEffect } from 'react';
import { Box, Text, useInput } from 'ink';
import { getTerminalName, installKeybindings } from '../../utils/ide/terminalSetup.js';
import { getSettings, updateSettings } from '../../services/config/SettingsService.js';
import { Card } from '../common/Card.js';

interface IdeOnboardingDialogProps {
    onDone: () => void;
}

export function IdeOnboardingDialog({ onDone }: IdeOnboardingDialogProps) {
    const [status, setStatus] = useState<'scanning' | 'found' | 'not-found' | 'installed' | 'error' | 'manual' | 'skipped'>('scanning');
    const [terminalName, setTerminalName] = useState<string | null>(null);
    const [message, setMessage] = useState<string>('');

    useEffect(() => {
        const checkSettings = async () => {
            const settings = getSettings();
            if (settings.shiftEnterKeyBindingInstalled) {
                onDone(); // Already installed
                return;
            }

            const term = getTerminalName();
            setTerminalName(term);

            if (term && ['vscode', 'cursor', 'windsurf', 'Code', 'Cursor', 'Windsurf'].includes(term)) {
                setStatus('found');
            } else if (term === 'iterm2' || term === 'apple_terminal' || term === 'warp' || term === 'iTerm.app') {
                setStatus('manual');
            } else {
                setStatus('not-found');
                setTimeout(onDone, 1500);
            }
        };
        checkSettings();
    }, [onDone]);

    const handleInstall = () => {
        if (!terminalName) return;

        const result = installKeybindings(terminalName);
        if (result.success) {
            setStatus('installed');
            setMessage(result.message);
            setTimeout(onDone, 2000);
        } else {
            setStatus('error');
            setMessage(result.message);
            // Allow user to continue even on error
            setTimeout(onDone, 3000);
        }
    };

    const handleSkip = () => {
        updateSettings(s => ({ ...s, shiftEnterKeyBindingInstalled: false }));
        onDone();
    };

    useInput((input, key) => {
        if (status === 'found' || status === 'manual') {
            if (key.return) {
                if (status === 'found') handleInstall();
                else handleSkip();
            }
            if (key.escape) {
                handleSkip();
            }
        }
    });

    if (status === 'scanning') {
        return <Text>Scanning environment...</Text>;
    }

    if (status === 'not-found') {
        return <Text>No supported IDE or enhanced terminal detected. Skipping advanced setup.</Text>;
    }

    if (status === 'found') {
        return (
            <Card title="Terminal Setup" borderColor="blue">
                <Box flexDirection="column" gap={1}>
                    <Text>
                        It looks like you are running in <Text bold color="cyan">{terminalName}</Text>.
                    </Text>
                    <Text>
                        For the best experience, we recommend installing the Shift+Enter keybinding
                        to insert newlines without submitting commands.
                    </Text>
                    <Box gap={2} marginTop={1}>
                        <Text color="green" bold underline>
                            Press Enter to Install
                        </Text>
                        <Text color="gray">
                            Esc to Skip
                        </Text>
                    </Box>
                </Box>
            </Card>
        );
    }

    if (status === 'manual') {
        let manualMsg = "";
        if (terminalName === 'iterm2' || terminalName === 'iTerm.app') {
            manualMsg = "In iTerm2, you can map Shift+Enter to send a hex code: \u001b\r (0x1B 0x0D).";
        } else if (terminalName === 'warp') {
            manualMsg = "Warp supports Shift+Enter by default for newlines in many contexts.";
        } else {
            manualMsg = "Check your terminal settings to map Shift+Enter if possible.";
        }

        return (
            <Card title="Terminal Setup" borderColor="blue">
                <Box flexDirection="column" gap={1}>
                    <Text>
                        Detected <Text bold color="cyan">{terminalName}</Text>. Automatic setup not supported.
                    </Text>
                    <Text color="yellow">
                        {manualMsg}
                    </Text>
                    <Box marginTop={1}>
                        <Text dimColor underline>Press Enter to Continue</Text>
                    </Box>
                </Box>
            </Card>
        );
    }

    if (status === 'installed') {
        return (
            <Card title="Success" borderColor="green">
                <Text color="green">{message}</Text>
            </Card>
        );
    }

    if (status === 'error') {
        return (
            <Card title="Error" borderColor="red">
                <Text color="red">{message}</Text>
            </Card>
        );
    }

    return null;
}
