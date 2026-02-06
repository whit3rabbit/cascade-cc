import React, { useState, useEffect } from 'react';
import { Box, Text, useInput } from 'ink';
import { Logo } from '../terminal/Logo.js';
import { Card } from '../common/Card.js';
import { IdeOnboardingDialog } from './IdeOnboardingDialog.js';
import { LspRecommendationDialog } from './LspRecommendationDialog.js';
import { updateSettings } from '../../services/config/SettingsService.js';
import { getAuthDetails } from '../../services/auth/AuthService.js';
import { themeService } from '../../services/terminal/ThemeService.js';
import SelectInput from 'ink-select-input';

interface OnboardingWorkflowProps {
    onDone: () => void;
}

type OnboardingStep = 'welcome' | 'auth' | 'theme' | 'security' | 'terminal' | 'lsp' | 'finish';

export function OnboardingWorkflow({ onDone }: OnboardingWorkflowProps) {
    const [step, setStep] = useState<OnboardingStep>('welcome');
    const [isLoggedIn, setIsLoggedIn] = useState(false);
    const [authInfo, setAuthInfo] = useState({ plan: 'Not Authenticated' });

    useEffect(() => {
        const checkAuth = async () => {
            const auth = await getAuthDetails();
            setIsLoggedIn(auth.type !== 'none');
            setAuthInfo({ plan: auth.plan });
        };
        checkAuth();
    }, []);

    const next = () => {
        if (step === 'welcome') setStep('auth');
        else if (step === 'auth') setStep('theme');
        else if (step === 'theme') setStep('security');
        else if (step === 'security') setStep('terminal');
        else if (step === 'terminal') setStep('lsp');
        else if (step === 'lsp') {
            updateSettings({ onboardingComplete: true });
            setStep('finish');
        } else if (step === 'finish') {
            onDone();
        }
    };

    useInput((input, key) => {
        const isActionKey = key.return || input === ' ';
        if (step === 'welcome' && isActionKey) next();
        if (step === 'auth' && isActionKey) next();
        if (step === 'security' && isActionKey) next();
        if (step === 'finish' && isActionKey) next();
    });

    const themeItems = [
        { label: 'Dark', value: 'dark' },
        { label: 'Light', value: 'light' },
        { label: 'Dark Daltonized', value: 'dark-daltonized' },
        { label: 'Light Daltonized', value: 'light-daltonized' },
        { label: 'Dark ANSI', value: 'dark-ansi' },
        { label: 'Light ANSI', value: 'light-ansi' }
    ];

    return (
        <Box flexDirection="column" padding={1}>
            {step === 'welcome' && (
                <Box flexDirection="column" alignItems="center">
                    <Logo
                        version="2.1.19"
                        model="claude-3-5-sonnet"
                        cwd={process.cwd()}
                        subscription={authInfo.plan}
                    />
                    <Box marginTop={1}>
                        <Text bold color="yellow">Welcome to Claude Code!</Text>
                    </Box>
                    <Box marginTop={1}>
                        <Text>Press Enter to start setup...</Text>
                    </Box>
                </Box>
            )}

            {step === 'auth' && (
                <Card title="Authentication" borderColor="blue">
                    <Box flexDirection="column">
                        <Text>Status: {isLoggedIn ? <Text color="green">Logged In</Text> : <Text color="red">Not Logged In</Text>}</Text>
                        {!isLoggedIn && (
                            <Box marginTop={1}>
                                <Text>Please run <Text bold>/login</Text> after this setup if you haven't already.</Text>
                            </Box>
                        )}
                        <Box marginTop={1}>
                            <Box borderStyle="round" borderColor="green" paddingX={1}>
                                <Text>Press Enter to Continue</Text>
                            </Box>
                        </Box>
                    </Box>
                </Card>
            )}

            {step === 'theme' && (
                <Box flexDirection="column">
                    <Box marginBottom={1}>
                        <Text bold>Choose your theme:</Text>
                    </Box>
                    <SelectInput items={themeItems} onSelect={(item) => {
                        themeService.setTheme(item.value as any);
                        next();
                    }} />
                </Box>
            )}

            {step === 'security' && (
                <Card title="Security & Privacy" borderColor="yellow">
                    <Box flexDirection="column" gap={1}>
                        <Text>• Claude can make mistakes. Always review generated code.</Text>
                        <Text>• Only use Claude with codebases you trust.</Text>
                        <Text>• Permissions will be requested for sensitive operations.</Text>
                        <Box marginTop={1}>
                            <Text dimColor>Press Enter to acknowledge</Text>
                        </Box>
                    </Box>
                </Card>
            )}

            {step === 'terminal' && (
                <IdeOnboardingDialog onDone={next} />
            )}

            {step === 'lsp' && (
                <LspRecommendationDialog onDone={next} />
            )}

            {step === 'finish' && (
                <Box flexDirection="column" alignItems="center">
                    <Text bold color="green">Setup Complete!</Text>
                    <Box marginTop={1}>
                        <Text>You're all set to use Claude Code.</Text>
                    </Box>
                    <Box marginTop={1}>
                        <Text dimColor>Press Enter to start your session</Text>
                    </Box>
                </Box>
            )}
        </Box>
    );
}
