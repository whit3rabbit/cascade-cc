import React, { useMemo, useState, useEffect } from 'react';
import { Box, Text } from 'ink';
import { DoctorService, DoctorDiagnostics, HealthCheckResult } from '../../services/terminal/DoctorService.js';

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
    status: 'ok' | 'warn' | 'error' | 'valid' | 'invalid' | 'capped' | 'not-set',
    message: string,
    details?: string
}) => {
    const statusColor = {
        ok: 'green',
        warn: 'yellow',
        error: 'red',
        valid: 'green',
        invalid: 'red',
        capped: 'yellow',
        'not-set': 'gray'
    }[status];

    const statusSymbol = {
        ok: '✓',
        warn: '⚠',
        error: '✗',
        valid: '✓',
        invalid: '✗',
        capped: '!',
        'not-set': '○'
    }[status];

    return (
        <Box flexDirection="column">
            <Box>
                <Box width={2}>
                    <Text color={statusColor}>{statusSymbol}</Text>
                </Box>
                <Box width={20}>
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
    const [diagnostics, setDiagnostics] = useState<DoctorDiagnostics | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        DoctorService.getDiagnostics()
            .then(setDiagnostics)
            .catch(e => setError(e.message));
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

        const checkStatus = (status: string) => {
            if (status === 'ok' || status === 'valid') ok++;
            else if (status === 'warn' || status === 'capped') warn++;
            else if (status === 'error' || status === 'invalid') errorCount++;
        };

        diagnostics.healthChecks.forEach(c => checkStatus(c.status));
        diagnostics.environmentVariables.forEach(v => checkStatus(v.status));

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
                <DiagnosticItem label="Version" status="ok" message={diagnostics.installation.version} />
                <DiagnosticItem label="Type" status="ok" message={diagnostics.installation.type} />
                <DiagnosticItem label="Install Method" status="ok" message={diagnostics.installation.installMethod} />
                <DiagnosticItem label="Auto Updates" status="ok" message={diagnostics.installation.autoUpdates} />
                <DiagnosticItem
                    label="Ripgrep"
                    status={diagnostics.installation.ripgrep.ok ? 'ok' : 'error'}
                    message={`${diagnostics.installation.ripgrep.mode} mode`}
                    details={diagnostics.installation.ripgrep.details}
                />
            </DiagnosticSection>

            <DiagnosticSection title="Environment Variables">
                {diagnostics.environmentVariables.map(v => (
                    <DiagnosticItem
                        key={v.name}
                        label={v.name}
                        status={v.status}
                        message={v.status === 'not-set' ? 'Not Set' : v.value}
                        details={v.message}
                    />
                ))}
            </DiagnosticSection>

            <DiagnosticSection title="Health Checks">
                {diagnostics.healthChecks.map((c, i) => (
                    <DiagnosticItem
                        key={i}
                        label={c.name}
                        status={c.status}
                        message={c.message}
                        details={c.details}
                    />
                ))}
            </DiagnosticSection>

            {diagnostics.versionLock?.locked && (
                <DiagnosticSection title="Version Lock">
                    <DiagnosticItem
                        label="Locked Version"
                        status="warn"
                        message={diagnostics.versionLock.version || 'Unknown'}
                        details={`Source: ${diagnostics.versionLock.source}`}
                    />
                </DiagnosticSection>
            )}

            {diagnostics.permissions.unreachableRules.length > 0 && (
                <DiagnosticSection title="Unreachable Permission Rules">
                    {diagnostics.permissions.unreachableRules.map((rule, i) => (
                        <DiagnosticItem key={i} label={`Rule ${i + 1}`} status="warn" message={rule} />
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
