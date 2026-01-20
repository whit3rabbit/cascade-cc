import React, { useCallback, useMemo, useState } from "react";
import { Box, Text, useInput } from "ink";
import { PermissionSelect } from "../permissions/PermissionComponents.js";
import { useCtrlExit } from "../../hooks/useCtrlExit.js";
import { getSettings, updateSettings } from "../../services/terminal/settings.js";
import { basename, sep } from "path";

type IdeConnection = {
    name: string;
    port: number;
    url: string;
    authToken?: string;
    workspaceFolders: string[];
    ideRunningInWindows?: boolean;
    isValid: boolean;
};

type DynamicMcpConfig = {
    ide?: {
        type: "sse-ide" | "ws-ide";
        url: string;
        ideName?: string;
        authToken?: string;
        ideRunningInWindows?: boolean;
        scope: "dynamic";
    };
};

function trackEvent(_name: string, _payload?: Record<string, any>) { }
function markIdeIntegrationFlow(_name: string) { }
function isRemoteSession(): boolean {
    return process.env.CLAUDE_CODE_REMOTE === "true";
}

async function listAvailableIdes(_includeClosed: boolean): Promise<IdeConnection[]> {
    return [];
}

function listRunningIdes(): string[] {
    return [];
}

function formatIdeName(idePath: string) {
    if (!idePath) return "Unknown IDE";
    return basename(idePath);
}

function shouldShowEnableAutoConnectDialog() {
    const settings = getSettings("userSettings") as any;
    return !isRemoteSession() && settings.autoConnectIde !== true && settings.hasIdeAutoConnectDialogBeenShown !== true;
}

function shouldShowDisableAutoConnectDialog() {
    const settings = getSettings("userSettings") as any;
    return !isRemoteSession() && settings.autoConnectIde === true;
}

// --- Enable Auto-connect Dialog (E59) ---
export function EnableAutoConnectDialog({ onComplete }: { onComplete: () => void }) {
    const exitState = useCtrlExit(async () => onComplete());

    const handleChoice = useCallback(
        (value: string) => {
            const enabled = value === "yes";
            updateSettings("userSettings", {
                autoConnectIde: enabled,
                hasIdeAutoConnectDialogBeenShown: true
            });
            onComplete();
        },
        [onComplete]
    );

    useInput((_input, key) => {
        if (key.escape) onComplete();
    });

    return (
        <Box marginTop={1} flexDirection="column">
            <Box flexDirection="column" borderStyle="round" borderColor="ide" paddingX={2} paddingY={1} width="100%">
                <Box marginBottom={1}>
                    <Text color="ide">Do you wish to enable auto-connect to IDE?</Text>
                </Box>
                <Box flexDirection="column" paddingX={1}>
                    <PermissionSelect
                        options={[
                            { label: "Yes", value: "yes" },
                            { label: "No", value: "no" }
                        ]}
                        onChange={handleChoice}
                        defaultValue="yes"
                        onCancel={onComplete}
                    />
                </Box>
                <Box marginTop={1}>
                    <Text dimColor>You can also configure this in /config or with the --ide flag</Text>
                </Box>
            </Box>
            <Box paddingX={1}>
                <Text dimColor>
                    {exitState.pending ? `Press ${exitState.keyName} again to exit` : "Enter to confirm"}
                </Text>
            </Box>
        </Box>
    );
}

// --- Disable Auto-connect Dialog (C59) ---
export function DisableAutoConnectDialog({ onComplete }: { onComplete: (confirmed: boolean) => void }) {
    const exitState = useCtrlExit(async () => onComplete(false));

    const handleChoice = useCallback(
        (value: string) => {
            const shouldDisable = value === "yes";
            if (shouldDisable) {
                updateSettings("userSettings", { autoConnectIde: false });
            }
            onComplete(shouldDisable);
        },
        [onComplete]
    );

    useInput((_input, key) => {
        if (key.escape) onComplete(false);
    });

    return (
        <Box marginTop={1} flexDirection="column">
            <Box flexDirection="column" borderStyle="round" borderColor="ide" paddingX={2} paddingY={1} width="100%">
                <Box marginBottom={1}>
                    <Text color="ide">Do you wish to disable auto-connect to IDE?</Text>
                </Box>
                <Box flexDirection="column" paddingX={1}>
                    <PermissionSelect
                        options={[
                            { label: "Yes", value: "yes" },
                            { label: "No", value: "no" }
                        ]}
                        onChange={handleChoice}
                        defaultValue="yes"
                        onCancel={() => onComplete(false)}
                    />
                </Box>
                <Box marginTop={1}>
                    <Text dimColor>You can also configure this in /config</Text>
                </Box>
            </Box>
            <Box paddingX={1}>
                <Text dimColor>
                    {exitState.pending ? `Press ${exitState.keyName} again to exit` : "Enter to confirm"}
                </Text>
            </Box>
        </Box>
    );
}

function findSelectedIde(ides: IdeConnection[], dynamicConfig?: DynamicMcpConfig) {
    const ideConfig = dynamicConfig?.ide;
    if (!ideConfig || (ideConfig.type !== "sse-ide" && ideConfig.type !== "ws-ide")) return null;
    return ides.find((ide) => ide.url === ideConfig.url) ?? null;
}

// --- IDE Integration View (l47) ---
export function IDEIntegrationView({
    availableIDEs,
    unavailableIDEs,
    selectedIDE,
    onClose,
    onSelect
}: {
    availableIDEs: IdeConnection[];
    unavailableIDEs: IdeConnection[];
    selectedIDE?: IdeConnection | null;
    onClose: () => void;
    onSelect: (ide?: IdeConnection) => void;
}) {
    const exitState = useCtrlExit(async () => onClose());
    const [focusedPort, setFocusedPort] = useState(selectedIDE?.port?.toString() ?? "None");
    const [showEnableDialog, setShowEnableDialog] = useState(false);
    const [showDisableDialog, setShowDisableDialog] = useState(false);

    const handleSelect = useCallback(
        (port: string) => {
            if (port !== "None" && shouldShowEnableAutoConnectDialog()) {
                setShowEnableDialog(true);
                return;
            }
            if (port === "None" && shouldShowDisableAutoConnectDialog()) {
                setShowDisableDialog(true);
                return;
            }
            const next = availableIDEs.find((ide) => ide.port === Number.parseInt(port, 10));
            onSelect(next);
        },
        [availableIDEs, onSelect]
    );

    const nameCounts = useMemo(() => {
        return availableIDEs.reduce<Record<string, number>>((acc, ide) => {
            acc[ide.name] = (acc[ide.name] || 0) + 1;
            return acc;
        }, {});
    }, [availableIDEs]);

    const options = useMemo(() => {
        const ideOptions = availableIDEs.map((ide) => {
            const showFolders = (nameCounts[ide.name] || 0) > 1 && ide.workspaceFolders.length > 0;
            return {
                label: ide.name,
                value: ide.port.toString(),
                description: showFolders ? formatPathList(ide.workspaceFolders) : undefined
            };
        });
        return ideOptions.concat([{ label: "None", value: "None", description: undefined }]);
    }, [availableIDEs, nameCounts]);

    useInput((_input, key) => {
        if (key.escape) onClose();
    });

    if (showEnableDialog) {
        return <EnableAutoConnectDialog onComplete={() => handleSelect(focusedPort)} />;
    }

    if (showDisableDialog) {
        return <DisableAutoConnectDialog onComplete={() => onSelect(undefined)} />;
    }

    return (
        <Box marginTop={1} flexDirection="column">
            <Box flexDirection="column" borderStyle="round" borderColor="ide" paddingX={2} paddingY={1} width="100%">
                <Box flexDirection="column">
                    <Text color="ide" bold>Select IDE</Text>
                    <Text dimColor>Connect to an IDE for integrated development features.</Text>
                    {availableIDEs.length === 0 && (
                        <Box marginTop={1}>
                            <Text dimColor>
                                {isJetBrainsIde()
                                    ? `No available IDEs detected. Please install the plugin and restart your IDE:
https://docs.claude.com/s/claude-code-jetbrains`
                                    : "No available IDEs detected. Make sure your IDE has the Claude Code extension or plugin installed and is running."}
                            </Text>
                        </Box>
                    )}
                </Box>

                {availableIDEs.length !== 0 && (
                    <Box flexDirection="column" paddingX={1} marginTop={1}>
                        <PermissionSelect
                            defaultValue={focusedPort}
                            defaultFocusValue={focusedPort}
                            options={options}
                            onFocus={(value) => setFocusedPort(value)}
                            onChange={(value) => {
                                setFocusedPort(value);
                                handleSelect(value);
                            }}
                            onCancel={onClose}
                        />
                    </Box>
                )}

                {availableIDEs.length !== 0 && !isRemoteSession() && (
                    <Box marginTop={1}>
                        <Text dimColor>※ Tip: You can enable auto-connect to IDE in /config or with the --ide flag</Text>
                    </Box>
                )}

                {unavailableIDEs.length > 0 && (
                    <Box marginTop={1} flexDirection="column">
                        <Text dimColor>
                            Found {unavailableIDEs.length} other running IDE(s). However, their workspace/project directories do not match the current cwd.
                        </Text>
                        <Box marginTop={1} flexDirection="column">
                            {unavailableIDEs.map((ide, index) => (
                                <Box key={`${ide.name}-${index}`} paddingLeft={3}>
                                    <Text dimColor>• {ide.name}: {formatPathList(ide.workspaceFolders)}</Text>
                                </Box>
                            ))}
                        </Box>
                    </Box>
                )}
            </Box>

            <Box paddingX={1}>
                <Text dimColor>
                    {exitState.pending ? `Press ${exitState.keyName} again to exit` : availableIDEs.length !== 0 ? "Enter to confirm · Esc to cancel" : "Esc to cancel"}
                </Text>
            </Box>
        </Box>
    );
}

// --- IDE Extension Install Dialog (n47) ---
export function InstallIDEExtensionDialog({
    runningIDEs,
    onSelectIDE,
    onDone
}: {
    runningIDEs: string[];
    onSelectIDE: (idePath: string) => void;
    onDone: (message?: any, options?: any) => void;
}) {
    const exitState = useCtrlExit(async () => onDone("IDE selection cancelled", { display: "system" }));
    const [focused, setFocused] = useState(runningIDEs[0] ?? "");

    const options = runningIDEs.map((idePath) => ({
        label: formatIdeName(idePath),
        value: idePath
    }));

    useInput((_input, key) => {
        if (key.escape) onDone("IDE selection cancelled", { display: "system" });
    });

    return (
        <>
            <Box flexDirection="column" borderStyle="round" borderColor="ide" marginTop={1} paddingX={2} paddingY={1} width="100%">
                <Box marginBottom={1}>
                    <Text color="ide">Select IDE to install extension:</Text>
                </Box>
                <Box flexDirection="column" paddingX={1}>
                    <PermissionSelect
                        defaultFocusValue={focused}
                        options={options}
                        onFocus={(value) => setFocused(value)}
                        onChange={(value) => {
                            setFocused(value);
                            onSelectIDE(value);
                        }}
                        onCancel={() => onDone("IDE selection cancelled", { display: "system" })}
                    />
                </Box>
            </Box>
            <Box paddingLeft={3}>
                <Text dimColor>
                    {exitState.pending ? `Press ${exitState.keyName} again to exit` : "Enter to confirm · Esc to cancel"}
                </Text>
            </Box>
        </>
    );
}

export function formatPathList(paths: string[], maxWidth = 100) {
    if (paths.length === 0) return "";
    const root = process.cwd();
    const shown = paths.slice(0, 2);
    const hasMore = paths.length > 2;
    const ellipsisWidth = hasMore ? 3 : 0;
    const separators = (shown.length - 1) * 2;
    const available = maxWidth - separators - ellipsisWidth;
    const perPath = Math.floor(available / shown.length);

    let joined = shown
        .map((rawPath) => {
            let display = rawPath;
            if (display.startsWith(root + sep)) display = display.slice(root.length + 1);
            if (display.length <= perPath) return display;
            return `…${display.slice(-(perPath - 1))}`;
        })
        .join(", ");

    if (hasMore) joined += ", …";
    return joined;
}

export const IDECommand = {
    type: "local-jsx",
    name: "ide",
    description: "Manage IDE integrations and show status",
    isEnabled: () => true,
    isHidden: false,
    argumentHint: "[open]",
    async call(onDone: any, context: any) {
        trackEvent("tengu_ext_ide_command", {});
        markIdeIntegrationFlow("ide-integration");

        const {
            options: { dynamicMcpConfig } = {},
            onChangeDynamicMcpConfig,
            onInstallIDEExtension
        } = context;

        const ides = await listAvailableIdes(true);
        if (ides.length === 0 && onInstallIDEExtension && !isRemoteSession()) {
            const runningIdes = listRunningIdes();
            const install = (idePath: string) => {
                if (!onInstallIDEExtension) return;
                if (onInstallIDEExtension(idePath)) {
                    onDone(`Installed plugin to ${formatIdeName(idePath)}\nPlease restart your IDE completely for it to take effect`);
                } else {
                    onDone(`Installed extension to ${formatIdeName(idePath)}`);
                }
            };

            if (runningIdes.length > 1) {
                return (
                    <InstallIDEExtensionDialog
                        runningIDEs={runningIdes}
                        onSelectIDE={install}
                        onDone={() => onDone("No IDE selected.", { display: "system" })}
                    />
                );
            }
            if (runningIdes.length === 1) {
                const idePath = runningIdes[0];
                return <AutoInstallRunner idePath={idePath} onInstall={install} />;
            }
        }

        const available = ides.filter((ide) => ide.isValid);
        const unavailable = ides.filter((ide) => !ide.isValid);
        const selected = findSelectedIde(available, dynamicMcpConfig);

        return (
            <IDEIntegrationView
                availableIDEs={available}
                unavailableIDEs={unavailable}
                selectedIDE={selected ?? undefined}
                onClose={() => onDone("IDE selection cancelled", { display: "system" })}
                onSelect={async (ide) => {
                    try {
                        if (!onChangeDynamicMcpConfig) {
                            onDone("Error connecting to IDE.");
                            return;
                        }
                        const nextConfig = { ...(dynamicMcpConfig || {}) };
                        if (selected) delete nextConfig.ide;

                        if (!ide) {
                            onDone(selected ? `Disconnected from ${selected.name}.` : "No IDE selected.");
                        } else {
                            const url = ide.url;
                            nextConfig.ide = {
                                type: url.startsWith("ws:") ? "ws-ide" : "sse-ide",
                                url,
                                ideName: ide.name,
                                authToken: ide.authToken,
                                ideRunningInWindows: ide.ideRunningInWindows,
                                scope: "dynamic"
                            };
                            onDone(`Connected to ${ide.name}.`);
                        }
                        onChangeDynamicMcpConfig(nextConfig);
                    } catch {
                        onDone("Error connecting to IDE.");
                    }
                }}
            />
        );
    },
    userFacingName() {
        return "ide";
    }
};

function isJetBrainsIde(): boolean {
    return Boolean(process.env.JETBRAINS_IDE);
}

function AutoInstallRunner({ idePath, onInstall }: { idePath: string; onInstall: (idePath: string) => void }) {
    React.useEffect(() => {
        onInstall(idePath);
    }, [idePath, onInstall]);
    return null;
}
