import React, { useMemo, useState, useEffect } from 'react';
import { Box, Text } from 'ink';
import { DoctorService, DiagnosticInfo } from '../../services/terminal/DoctorService.js';

const DiagnosticSection = ({ title, children }: { title: string, children: React.ReactNode }) => (
    <Box flexDirection="column" marginBottom={1}>
        <Text bold underline color="cyan">{title}</Text>
        <Box flexDirection="column" paddingLeft={2}>
            {children}
        </Box>
    </Box>
);

const DiagnosticItem = ({ label, status, message, details }: {
    label: string,
    status: 'ok' | 'warn' | 'error' | 'capped' | 'not-set',
    message: string,
    details?: string
}) => {
    const statusColor = {
        ok: 'green',
        warn: 'yellow',
        error: 'red',
        capped: 'yellow',
        'not-set': 'gray'
    }[status] || 'white';

    const statusSymbol = {
        ok: '✓',
        warn: '⚠',
        error: '✗',
        capped: '!',
        'not-set': '○'
    }[status] || '?';

    return (
        <Box flexDirection="column">
            <Box>
                <Box width={2}>
                    <Text color={statusColor}>{statusSymbol}</Text>
                </Box>
                <Box width={25}>
                    <Text bold>{label}:</Text>
                </Box>
                <Text color={statusColor}>{message}</Text>
            </Box>
            {details && (
                <Box paddingLeft={2}>
                    <Text dimColor italic>{details}</Text>
                </Box>
            )}
        </Box>
    );
};

export const Doctor: React.FC = () => {
    const [diagnostics, setDiagnostics] = useState<DiagnosticInfo | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        DoctorService.getDiagnosticInfo()
            .then(setDiagnostics)
            .catch((e: Error) => setError(e.message));
    }, []);

    if (error) {
        return (
            <Box flexDirection="column" padding={1}>
                <Text color="red" bold>Error running diagnostics:</Text>
                <Text>{error}</Text>
            </Box>
        );
    }

    if (!diagnostics) {
        return (
            <Box padding={1}>
                <Text>Running diagnostics...</Text>
            </Box>
        );
    }

    const summary = useMemo(() => {
        let ok = 0;
        let warn = 0;
        let errorCount = 0;

        const checkStatus = (status: 'ok' | 'capped' | 'error' | 'not-set') => {
            if (status === 'ok') ok++;
            else if (status === 'capped' || status === 'warn' as any) warn++;
            else if (status === 'error') errorCount++;
        };

        diagnostics.envVars.forEach(v => checkStatus(v.status));
        if (diagnostics.ripgrepStatus.workingDirectory) ok++; else errorCount++;
        if (diagnostics.ghStatus.installed && diagnostics.ghStatus.authenticated) ok++; else warn++;

        return { ok, warn, errorCount };
    }, [diagnostics]);

    return (
        <Box flexDirection="column" padding={1}>
            <Box marginBottom={1} borderStyle="single" borderColor="cyan" paddingX={2} flexDirection="column">
                <Text bold color="cyan">CLAUDE DOCTOR REPORT</Text>
                <Box marginTop={1}>
                    <Text color="green">Passed: {summary.ok}</Text>
                    <Text> | </Text>
                    <Text color="yellow">Warnings: {summary.warn}</Text>
                    <Text> | </Text>
                    <Text color="red">Errors: {summary.errorCount}</Text>
                </Box>
            </Box>

            <DiagnosticSection title="Installation Information">
                <DiagnosticItem label="Version" status="ok" message={diagnostics.version} />
                <DiagnosticItem label="Type" status="ok" message={diagnostics.installationType} />
                <DiagnosticItem label="Path" status="ok" message={diagnostics.installationPath} />
                <DiagnosticItem label="Auto Updates" status="ok" message={diagnostics.autoUpdates} />
                <DiagnosticItem
                    label="Ripgrep"
                    status={diagnostics.ripgrepStatus.workingDirectory ? 'ok' : 'error'}
                    message={`${diagnostics.ripgrepStatus.mode} mode`}
                    details={diagnostics.ripgrepStatus.systemPath || undefined}
                />
            </DiagnosticSection>

            <DiagnosticSection title="Environment Variables">
                {diagnostics.envVars.map(v => (
                    <DiagnosticItem
                        key={v.name}
                        label={v.name}
                        status={v.status}
                        message={v.message}
                    />
                ))}
            </DiagnosticSection>

            <DiagnosticSection title="External Tools">
                <DiagnosticItem
                    label="GitHub CLI"
                    status={diagnostics.ghStatus.installed ? (diagnostics.ghStatus.authenticated ? 'ok' : 'warn') : 'error'}
                    message={diagnostics.ghStatus.installed ? (diagnostics.ghStatus.authenticated ? 'Authenticated' : 'Not Authenticated') : 'Not Installed'}
                />
                <DiagnosticItem
                    label="Git"
                    status={diagnostics.gitStatus.isRepo ? 'ok' : 'warn'}
                    message={diagnostics.gitStatus.isRepo ? 'Inside Repository' : 'Not a Repository'}
                />
            </DiagnosticSection>

            {diagnostics.warnings.length > 0 && (
                <DiagnosticSection title="Warnings & Issues">
                    {diagnostics.warnings.map((w, i) => (
                        <Box key={i} flexDirection="column" marginBottom={1}>
                            <Text color="yellow" bold>Issue: {w.issue}</Text>
                            <Text dimColor>Fix: {w.fix}</Text>
                        </Box>
                    ))}
                </DiagnosticSection>
            )}

            <Box marginTop={1}>
                <Text dimColor>To fix common issues, try running </Text>
                <Text bold color="cyan">claude update</Text>
                <Text dimColor> or check the documentation at </Text>
                <Text color="blue" underline>https://docs.anthropic.com/claude-code</Text>
            </Box>
        </Box>
    );
};
