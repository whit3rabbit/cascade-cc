import React, { useState, useEffect } from 'react';
import { Box, Text, useInput } from 'ink';
import { DoctorService, DiagnosticInfo } from '../../services/terminal/DoctorService.js';

interface DoctorProps {
    onExit?: () => void;
}

export const Doctor: React.FC<DoctorProps> = ({ onExit }) => {
    const [diagnostics, setDiagnostics] = useState<DiagnosticInfo | null>(null);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        DoctorService.getDiagnosticInfo()
            .then(setDiagnostics)
            .catch((e: Error) => setError(e.message));
    }, []);

    useInput((_input, key) => {
        if (!onExit) return;
        if (key.escape || key.return) {
            onExit();
        }
    });

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

    return (
        <Box flexDirection="column" padding={1}>
            <Box flexDirection="column" marginBottom={1}>
                <Text bold>Diagnostics</Text>
                <Text>└ Currently running: {diagnostics.installationType} ({diagnostics.version})</Text>
                {diagnostics.packageManager && <Text>└ Package manager: {diagnostics.packageManager}</Text>}
                <Text>└ Path: {diagnostics.installationPath}</Text>
                <Text>└ Invoked: {diagnostics.invokedBinary}</Text>
                <Text>└ Config install method: {diagnostics.configInstallMethod}</Text>
                <Text>
                    └ Search: {diagnostics.ripgrepStatus.workingDirectory ? 'OK' : 'Not working'} (
                    {diagnostics.ripgrepStatus.mode === 'builtin'
                        ? 'bundled'
                        : (diagnostics.ripgrepStatus.systemPath || 'system')}
                    )
                </Text>
            </Box>

            <Box flexDirection="column" marginBottom={1}>
                <Text bold>Updates</Text>
                <Text>└ Auto-updates: {diagnostics.packageManager ? 'Managed by package manager' : diagnostics.autoUpdates}</Text>
                <Text>└ Auto-update channel: {diagnostics.updateChannel}</Text>
                <Text>└ Stable version: {diagnostics.stableVersion || 'unknown'}</Text>
                <Text>└ Latest version: {diagnostics.latestVersion || 'unknown'}</Text>
            </Box>

            <Box flexDirection="column" marginBottom={1}>
                <Text bold>Version Locks</Text>
                {diagnostics.versionLocks.length === 0 ? (
                    <Text>└ No active version locks</Text>
                ) : (
                    diagnostics.versionLocks.map((lock) => (
                        <Text key={`${lock.version}-${lock.pid}`}>
                            └ {lock.version}: PID {lock.pid} {lock.isProcessRunning ? '(running)' : '(stale)'}
                        </Text>
                    ))
                )}
            </Box>

            <Text dimColor>Press Enter to continue…</Text>
        </Box>
    );
};
