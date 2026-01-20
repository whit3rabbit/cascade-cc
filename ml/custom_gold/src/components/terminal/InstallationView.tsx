// Logic from chunk_601.ts (Native Binary Installation)

import React, { useEffect, useState } from "react";
import { Box, Text } from "ink";
import { homedir } from "node:os";
import { join } from "node:path";
import { figures } from "../../vendor/terminalFigures.js";
import { getSettings } from "../../services/terminal/settings.js";
import { log } from "../../services/logger/loggerService.js";
import { runNativeUpdate, setupLauncher, UpdateResult } from "../../services/updater/NativeUpdater.js";

type SetupMessage = { message: string };
type InstallResult = {
    latestVersion?: string;
    wasUpdated?: boolean;
    lockFailed?: boolean;
    lockHolderPid?: number;
};
type CleanupResult = { removed: number; errors: string[]; warnings: string[] };
type OnDone = (message: string, meta?: { display?: string }) => void;

type InstallState =
    | { type: "checking" }
    | { type: "installing"; version: string }
    | { type: "setting-up" }
    | { type: "set-up"; messages: string[] }
    | { type: "success"; version: string; setupMessages?: string[] }
    | { type: "error"; message: string }
    | { type: "cleaning-npm" };

const logger = log("install");

function trackEvent(_name: string, _payload?: Record<string, any>) { }

// Replaced by imports

export function getBinaryPath() {
    const isWindows = process.platform === "win32";
    const home = homedir();
    if (isWindows) {
        return join(home, ".local", "bin", "claude.exe").replace(/\//g, "\\");
    }
    return "~/.local/bin/claude";
}

export function SetupNotes({ messages }: { messages: string[] }) {
    if (messages.length === 0) return null;
    return (
        <Box flexDirection="column" gap={0} marginBottom={1}>
            <Text>
                <Text color="warning">
                    {figures.warning} Setup notes:
                </Text>
            </Text>
            {messages.map((message, index) => (
                <Box key={index} marginLeft={2}>
                    <Text dimColor>• {message}</Text>
                </Box>
            ))}
        </Box>
    );
}

export function InstallationView({
    onDone,
    force,
    target
}: {
    onDone: OnDone;
    force?: boolean;
    target?: string;
}) {
    const [state, setState] = useState<InstallState>({ type: "checking" });

    useEffect(() => {
        async function run() {
            try {
                logger.info(`Install: Starting installation process (force=${force ?? false}, target=${target ?? ""})`);
                const settings = getSettings("userSettings") as { autoUpdatesChannel?: string };
                const channelOrVersion = target || settings.autoUpdatesChannel || "latest";
                setState({ type: "installing", version: channelOrVersion });
                logger.info(
                    `Install: Calling installLatest(channelOrVersion=${channelOrVersion}, force=true, forceReinstall=${force ?? false})`
                );
                const result: UpdateResult = await runNativeUpdate(channelOrVersion, true, force);
                logger.info(
                    `Install: runNativeUpdate returned version=${result.latestVersion}, wasUpdated=${result.wasUpdated}, lockFailed=${result.lockFailed}`
                );
                if (result.lockFailed) {
                    throw new Error(
                        "Could not install - another process is currently installing Claude. Please try again in a moment."
                    );
                }
                if (!result.latestVersion) {
                    logger.error("Install: Failed to retrieve version information during install");
                }
                if (!result.wasUpdated) {
                    logger.info("Install: Already up to date");
                }

                setState({ type: "setting-up" });
                const setupMessages = await setupLauncher(true);
                if (setupMessages.length > 0) {
                    setupMessages.forEach((message) => logger.info(`Install: Setup message: ${message.message}`));
                }

                // Cleanup logic ...
                // Note: removeAllLegacyVersions and removeShellAliases are likely missing from current deob status or in AutoUpdater.ts
                // I'll keep them as placeholders or comment out if they cause issues.
                // For now, I'll keep the structure.

                trackEvent("tengu_claude_install_command", {
                    has_version: result.latestVersion ? 1 : 0,
                    forced: force ? 1 : 0
                });

                const setupNotes: string[] = [];
                if (setupMessages.length > 0) {
                    setState({ type: "set-up", messages: setupMessages.map((message) => message.message) });
                    setTimeout(() => {
                        setState({
                            type: "success",
                            version: result.latestVersion || "current",
                            setupMessages: [...setupMessages.map((message) => message.message), ...setupNotes]
                        });
                    }, 2000);
                } else {
                    logger.info("Install: Shell PATH already configured");
                    setState({
                        type: "success",
                        version: result.latestVersion || "current",
                        setupMessages: setupNotes.length > 0 ? setupNotes : undefined
                    });
                }
            } catch (error) {
                logger.error(`Install command failed: ${error}`);
                setState({
                    type: "error",
                    message: error instanceof Error ? error.message : String(error)
                });
            }
        }

        run();
    }, [force, target]);

    useEffect(() => {
        if (state.type === "success") {
            const timer = setTimeout(() => {
                onDone("Claude Code installation completed successfully", { display: "system" });
            }, 2000);
            return () => clearTimeout(timer);
        }
        if (state.type === "error") {
            const timer = setTimeout(() => {
                onDone("Claude Code installation failed", { display: "system" });
            }, 3000);
            return () => clearTimeout(timer);
        }
    }, [state, onDone]);

    return (
        <Box flexDirection="column" marginTop={1}>
            {state.type === "checking" && <Text color="claude">Checking installation status...</Text>}
            {state.type === "cleaning-npm" && <Text color="warning">Cleaning up old npm installations...</Text>}
            {state.type === "installing" && (
                <Text color="claude">Installing Claude Code native build {state.version}...</Text>
            )}
            {state.type === "setting-up" && (
                <Text color="claude">Setting up launcher and shell integration...</Text>
            )}
            {state.type === "set-up" && <SetupNotes messages={state.messages} />}
            {state.type === "success" && (
                <Box flexDirection="column" gap={1}>
                    <Box>
                        <Text color="success">
                            {figures.tick}{" "}
                        </Text>
                        <Text color="success" bold>
                            Claude Code successfully installed!
                        </Text>
                    </Box>
                    <Box marginLeft={2} flexDirection="column" gap={1}>
                        {state.version !== "current" && (
                            <Box>
                                <Text dimColor>Version: </Text>
                                <Text color="claude">{state.version}</Text>
                            </Box>
                        )}
                        <Box>
                            <Text dimColor>Location: </Text>
                            <Text color="text">{getBinaryPath()}</Text>
                        </Box>
                    </Box>
                    <Box marginLeft={2} flexDirection="column" gap={1}>
                        <Box marginTop={1}>
                            <Text dimColor>Next: Run </Text>
                            <Text color="claude" bold>
                                claude --help
                            </Text>
                            <Text dimColor> to get started</Text>
                        </Box>
                    </Box>
                    {state.setupMessages && <SetupNotes messages={state.setupMessages} />}
                </Box>
            )}
            {state.type === "error" && (
                <Box flexDirection="column" gap={1}>
                    <Box>
                        <Text color="error">
                            {figures.cross}{" "}
                        </Text>
                        <Text color="error">Installation failed</Text>
                    </Box>
                    <Text color="error">{state.message}</Text>
                    <Box marginTop={1}>
                        <Text dimColor>Try running with --force to override checks</Text>
                    </Box>
                </Box>
            )}
        </Box>
    );
}

export const installCommand = {
    type: "local-jsx",
    name: "install",
    description: "Install Claude Code native build",
    argumentHint: "[options]",
    async call(onDone: OnDone, _context: any, args: string[] = []) {
        const force = args.includes("--force");
        const target = args.find((arg) => !arg.startsWith("--"));
        return <InstallationView onDone={onDone} force={force} target={target} />;
    }
};
