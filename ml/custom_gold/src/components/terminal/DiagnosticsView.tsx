import React, { useEffect, useState } from "react";
import { Box, Text, useInput } from "ink";
import { getDiagnosticsInfo, DiagnosticsInfo } from "../../services/terminal/DiagnosticsService.js";

export const DiagnosticsView: React.FC<{ onDone: (msg: string) => void }> = ({ onDone }) => {
    const [info, setInfo] = useState<DiagnosticsInfo | null>(null);

    useInput((input, key) => {
        if (key.return || key.escape || (key.ctrl && input === "c") || input) {
            onDone("Claude Code diagnostics dismissed");
        }
    });

    useEffect(() => {
        getDiagnosticsInfo().then(setInfo);
    }, []);

    if (!info) {
        return (
            <Box paddingX={1} paddingTop={1}>
                <Text dimColor>Checking installation status...</Text>
            </Box>
        );
    }

    return (
        <Box flexDirection="column" gap={1} paddingX={1} paddingTop={1}>
            <Box flexDirection="column">
                <Text bold>Diagnostics</Text>
                <Text>└ Currently running: {info.installationType} ({info.version})</Text>
                {info.packageManager && <Text>└ Package manager: {info.packageManager}</Text>}
                <Text>└ Path: {info.installationPath}</Text>
                <Text>└ Invoked: {info.invokedBinary}</Text>
                <Text>└ Config install method: {info.configInstallMethod}</Text>
                <Text>└ Auto-updates: {info.packageManager ? "Managed by package manager" : info.autoUpdates}</Text>
                {info.hasUpdatePermissions !== null && (
                    <Text>└ Update permissions: {info.hasUpdatePermissions ? "Yes" : "No (requires sudo)"}</Text>
                )}
                <Text>└ Search: {info.ripgrepStatus.working ? "OK" : "Not working"} ({info.ripgrepStatus.mode === "builtin" ? "vendor" : (info.ripgrepStatus.systemPath || info.ripgrepStatus.mode)})</Text>

                {info.recommendation && (
                    <>
                        <Text> </Text>
                        <Text color="warning">Recommendation: {info.recommendation.split('\n')[0]}</Text>
                        <Text dimColor>{info.recommendation.split('\n')[1]}</Text>
                    </>
                )}

                {info.multipleInstallations.length > 1 && (
                    <>
                        <Text> </Text>
                        <Text color="warning">Warning: Multiple installations found</Text>
                        {info.multipleInstallations.map((inst, i) => (
                            <Text key={i}>└ {inst.type} at {inst.path}</Text>
                        ))}
                    </>
                )}

                {info.warnings.length > 0 && (
                    <>
                        <Text> </Text>
                        {info.warnings.map((w, i) => (
                            <Box key={i} flexDirection="column">
                                <Text color="warning">Warning: {w.issue}</Text>
                                <Text>Fix: {w.fix}</Text>
                            </Box>
                        ))}
                    </>
                )}

                <Box marginTop={1}>
                    <Text dimColor>Press any key to exit diagnostics</Text>
                </Box>
            </Box>
        </Box>
    );
};
