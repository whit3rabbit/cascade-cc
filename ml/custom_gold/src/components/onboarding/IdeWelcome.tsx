import React, { useEffect, useMemo } from "react";
import { Box, Text, useInput } from "ink";
import { getIdeDisplayName } from "../../services/ide/IdeIntegration.js";
import { IDE_REGISTRY } from "../../services/ide/IdeRegistry.js";
import { useCtrlExit } from "../../hooks/useCtrlExit.js";

interface IdeWelcomeProps {
    onDone: () => void;
    installationStatus?: {
        ideType?: string;
        installedVersion?: string;
    };
}

const ideOnboardingShownByTerminal: Record<string, boolean> = {};

function getTerminalId(): string {
    return process.env.TERM_PROGRAM || process.env.TERMINAL_EMULATOR || "unknown";
}

function hasIdeOnboardingBeenShown(terminalId: string): boolean {
    return ideOnboardingShownByTerminal[terminalId] === true;
}

function markIdeOnboardingShown(terminalId: string) {
    ideOnboardingShownByTerminal[terminalId] = true;
}

function isJetBrainsIde(ideType?: string): boolean {
    if (!ideType) return false;
    const entry = IDE_REGISTRY[ideType.toLowerCase()];
    return entry?.ideKind === "jetbrains";
}

export function IdeWelcome({ onDone, installationStatus }: IdeWelcomeProps) {
    const ctrlExit = useCtrlExit();
    const ideType = installationStatus?.ideType ?? getTerminalId();
    const ideName = getIdeDisplayName(ideType);
    const version = installationStatus?.installedVersion;
    const typeLabel = isJetBrainsIde(ideType) ? "plugin" : "extension";
    const keyBinding = process.platform === "darwin" ? "Cmd+Option+K" : "Ctrl+Alt+K";
    const terminalId = useMemo(() => getTerminalId(), []);

    useEffect(() => {
        if (!hasIdeOnboardingBeenShown(terminalId)) {
            markIdeOnboardingShown(terminalId);
        }
    }, [terminalId]);

    useInput((_, key) => {
        if (key.escape || key.return) {
            onDone();
        }
    });

    return (
        <Box flexDirection="column">
            <Box flexDirection="column" borderStyle="round" borderColor="ide" paddingX={1} gap={1}>
                <Box>
                    <Text color="claude">✻ </Text>
                    <Box flexDirection="column">
                        <Text>Welcome to <Text bold>Claude Code</Text> for <Text color="ide" bold>{ideName}</Text></Text>
                        {version && <Text dimColor>installed {typeLabel} v{version}</Text>}
                    </Box>
                </Box>

                <Box flexDirection="column" paddingLeft={1} gap={1}>
                    <Text>• Claude has context of <Text color="suggestion">⧉ open files</Text> and <Text color="suggestion">⧉ selected lines</Text></Text>
                    <Text>• Review Claude Code's changes <Text color="diffAddedWord">+11</Text> <Text color="diffRemovedWord">-22</Text> in the comfort of your IDE</Text>
                    <Text>• Cmd+Esc <Text dimColor>for Quick Launch</Text></Text>
                    <Text>• {keyBinding} <Text dimColor>to reference files or lines in your input</Text></Text>
                </Box>
            </Box>

            <Box marginLeft={3}>
                {ctrlExit.pending ? (
                    <Text dimColor>Press {ctrlExit.keyName} again to exit</Text>
                ) : (
                    <Text dimColor>Press Enter to continue</Text>
                )}
            </Box>
        </Box>
    );
}
