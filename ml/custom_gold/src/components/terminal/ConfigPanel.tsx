// Logic from chunk_541.ts (Config Panel UI)

import React, { useRef, useState } from "react";
import { Box, Text, useInput } from "ink";
import { figures } from "../../vendor/terminalFigures.js";
import { useAppState } from "../../contexts/AppStateContext.js";

const DEFAULT_OUTPUT_STYLE = "default";

const THEME_LABELS: Record<string, string> = {
    dark: "Dark mode",
    light: "Light mode",
    "dark-daltonized": "Dark mode (colorblind-friendly)",
    "light-daltonized": "Light mode (colorblind-friendly)",
    "dark-ansi": "Dark mode (ANSI colors only)",
    "light-ansi": "Light mode (ANSI colors only)"
};

const PERMISSION_MODES = ["default", "plan", "bypassPermissions", "review", "auto"];

type ConfigItem = {
    id: string;
    label: React.ReactNode;
    value: any;
    type: "boolean" | "enum" | "managedEnum";
    options?: string[];
    onChange?: (value: any) => void;
};

type ConfigPanelProps = {
    onClose: (message: string, meta?: any) => void;
    context: { options?: { mcpClients?: any[] } };
    setTabsHidden: (hidden: boolean) => void;
    setIsWarning: (value: boolean) => void;
    setHideMargin: (value: boolean) => void;
};

type Settings = {
    autoCompactEnabled?: boolean;
    terminalProgressBarEnabled?: boolean;
    fileCheckpointingEnabled?: boolean;
    respectGitignore?: boolean;
    preferredNotifChannel?: string;
    editorMode?: string;
    diffTool?: string;
    autoConnectIde?: boolean;
    autoInstallIdeExtension?: boolean;
    claudeInChromeDefaultEnabled?: boolean;
    customApiKeyResponses?: { approved?: string[]; rejected?: string[] };
    theme?: string;
};

type UserSettings = {
    spinnerTipsEnabled?: boolean;
    permissions?: { defaultMode?: string };
    autoUpdatesChannel?: string;
    minimumVersion?: string;
    outputStyle?: string;
};

function getSettings(): Settings {
    return {
        autoCompactEnabled: false,
        terminalProgressBarEnabled: true,
        fileCheckpointingEnabled: true,
        respectGitignore: true,
        preferredNotifChannel: "auto",
        editorMode: "normal",
        diffTool: "auto",
        autoConnectIde: false,
        autoInstallIdeExtension: true,
        claudeInChromeDefaultEnabled: true,
        customApiKeyResponses: { approved: [], rejected: [] },
        theme: "dark"
    };
}

function getUserSettings(): UserSettings {
    return {
        spinnerTipsEnabled: true,
        permissions: { defaultMode: "default" },
        autoUpdatesChannel: "latest",
        outputStyle: DEFAULT_OUTPUT_STYLE
    };
}

function updateSettings(_updater: (settings: Settings) => Settings) {
    void _updater;
}

function updateUserSettings(_scope: string, _settings: Record<string, any>) {
    return { error: null as Error | null };
}

function trackEvent(_name: string, _payload?: Record<string, any>) {
    void _name;
    void _payload;
}

function hasMcpClients(clients?: any[]): boolean {
    return Boolean(clients && clients.length > 0);
}

function isFeatureEnabled(_flag?: string): boolean {
    return true;
}

function canEditExternalIncludes(): boolean {
    return false;
}

function getManagedAutoUpdatesReason(): string | null {
    return null;
}

function isExternalTerminal(): boolean {
    return false;
}

function maskApiKey(key: string): string {
    if (!key) return "";
    const tail = key.slice(-4);
    return `****${tail}`;
}

function formatPermissionMode(mode: string): string {
    switch (mode) {
        case "default":
            return "Default";
        case "plan":
            return "Plan";
        case "bypassPermissions":
            return "Bypass permissions";
        case "review":
            return "Review";
        case "auto":
            return "Auto";
        default:
            return mode;
    }
}

function parsePermissionMode(mode: string): string {
    if (PERMISSION_MODES.includes(mode)) return mode;
    return "default";
}

function formatModelValue(model: string | null): string {
    return model === null ? "Default (recommended)" : model;
}

const ShortcutGroup: React.FC<{ children: React.ReactNode }> = ({ children }) => (
    <Box flexDirection="row" gap={1}>
        {children}
    </Box>
);

const ShortcutHint: React.FC<{ shortcut: string; action: string }> = ({ shortcut, action }) => (
    <Text>
        {shortcut} {action}
    </Text>
);

function ThemeSelector({
    initialTheme,
    onThemeSelect,
    onCancel
}: {
    initialTheme: string;
    onThemeSelect: (theme: string) => void;
    onCancel: () => void;
}) {
    const themes = Object.keys(THEME_LABELS);
    const [index, setIndex] = useState(Math.max(themes.indexOf(initialTheme), 0));

    useInput((_input, key) => {
        if (key.upArrow) setIndex((value) => Math.max(0, value - 1));
        if (key.downArrow) setIndex((value) => Math.min(themes.length - 1, value + 1));
        if (key.return) onThemeSelect(themes[index]);
        if (key.escape) onCancel();
    });

    return (
        <Box flexDirection="column">
            <Text bold>Select theme</Text>
            {themes.map((theme, idx) => (
                <Text key={theme} color={idx === index ? "suggestion" : undefined}>
                    {idx === index ? figures.pointer : " "} {THEME_LABELS[theme] ?? theme}
                </Text>
            ))}
        </Box>
    );
}

function ModelPicker({ initial, onSelect, onCancel }: { initial: string | null; onSelect: (m: string) => void; onCancel: () => void }) {
    const models = ["sonnet", "opus", "haiku", "inherit"];
    const [index, setIndex] = useState(Math.max(models.indexOf(initial ?? ""), 0));

    useInput((_input, key) => {
        if (key.upArrow) setIndex((value) => Math.max(0, value - 1));
        if (key.downArrow) setIndex((value) => Math.min(models.length - 1, value + 1));
        if (key.return) onSelect(models[index]);
        if (key.escape) onCancel();
    });

    return (
        <Box flexDirection="column">
            <Text bold>Select model</Text>
            {models.map((model, idx) => (
                <Text key={model} color={idx === index ? "suggestion" : undefined}>
                    {idx === index ? figures.pointer : " "} {model}
                </Text>
            ))}
        </Box>
    );
}

function ExternalIncludesDialog({ onDone }: { onDone: () => void }) {
    useInput((_input, key) => {
        if (key.return || key.escape) onDone();
    });

    return (
        <Box flexDirection="column">
            <Text>External CLAUDE.md includes</Text>
        </Box>
    );
}

function OutputStylePicker({ initialStyle, onComplete, onCancel }: { initialStyle: string; onComplete: (style?: string) => void; onCancel: () => void }) {
    const options = ["default", "compact", "verbose"];
    const [index, setIndex] = useState(Math.max(options.indexOf(initialStyle), 0));

    useInput((_input, key) => {
        if (key.upArrow) setIndex((value) => Math.max(0, value - 1));
        if (key.downArrow) setIndex((value) => Math.min(options.length - 1, value + 1));
        if (key.return) onComplete(options[index]);
        if (key.escape) onCancel();
    });

    return (
        <Box flexDirection="column">
            <Text bold>Select output style</Text>
            {options.map((style, idx) => (
                <Text key={style} color={idx === index ? "suggestion" : undefined}>
                    {idx === index ? figures.pointer : " "} {style}
                </Text>
            ))}
        </Box>
    );
}

function AutoUpdateChannelDialog({ currentVersion, onChoice }: { currentVersion: string; onChoice: (value: string) => void }) {
    useInput((_input, key) => {
        if (key.escape) onChoice("cancel");
        if (key.return) onChoice("stay");
    });

    return (
        <Box flexDirection="column">
            <Text>Switch to stable channel?</Text>
            <Text dimColor>Current version: {currentVersion}</Text>
        </Box>
    );
}

export function ConfigPanel({ onClose, context, setTabsHidden, setIsWarning, setHideMargin }: ConfigPanelProps) {
    const [theme, setTheme] = useState("dark");
    const [settings, setSettings] = useState<Settings>(getSettings());
    const settingsRef = useRef(settings);
    const [userSettings, setUserSettings] = useState<UserSettings>(getUserSettings());
    const userSettingsRef = useRef(userSettings);
    const [outputStyle, setOutputStyle] = useState(userSettings.outputStyle ?? DEFAULT_OUTPUT_STYLE);
    const outputStyleRef = useRef(outputStyle);
    const [selectedIndex, setSelectedIndex] = useState(0);
    const [{ mainLoopModel, verbose, thinkingEnabled, promptSuggestionEnabled }, setAppState] = useAppState();
    const [changeLog, setChangeLog] = useState<Record<string, any>>({});
    const [activeModal, setActiveModal] = useState<number | null>(null);
    const hasClients = hasMcpClients(context.options?.mcpClients);
    const allowCheckpointing = !isFeatureEnabled(process.env.CLAUDE_CODE_DISABLE_FILE_CHECKPOINTING);
    const allowExternalIncludes = canEditExternalIncludes();
    const managedAutoUpdatesReason = getManagedAutoUpdatesReason();

    async function updateModelSetting(nextModel: string) {
        trackEvent("tengu_config_model_changed", { from_model: mainLoopModel, to_model: nextModel });
        setAppState((prev: any) => ({ ...prev, mainLoopModel: nextModel }));
        setChangeLog((prev) => ({ ...prev, model: nextModel }));
    }

    function updateVerboseSetting(nextVerbose: boolean) {
        updateSettings((prev) => ({ ...prev, verbose: nextVerbose } as Settings));
        setSettings({ ...getSettings(), verbose: nextVerbose } as Settings);
        setAppState((prev: any) => ({ ...prev, verbose: nextVerbose }));
        setChangeLog((prev) => {
            if ("verbose" in prev) {
                const { verbose: _unused, ...rest } = prev;
                return rest;
            }
            return { ...prev, verbose: nextVerbose };
        });
    }

    const configItems: ConfigItem[] = [
        {
            id: "autoCompactEnabled",
            label: "Auto-compact",
            value: settings.autoCompactEnabled,
            type: "boolean",
            onChange(next) {
                updateSettings((prev) => ({ ...prev, autoCompactEnabled: next }));
                setSettings({ ...getSettings(), autoCompactEnabled: next });
                trackEvent("tengu_auto_compact_setting_changed", { enabled: next });
            }
        },
        {
            id: "spinnerTipsEnabled",
            label: "Show tips",
            value: userSettings.spinnerTipsEnabled ?? true,
            type: "boolean",
            onChange(next) {
                updateUserSettings("localSettings", { spinnerTipsEnabled: next });
                setUserSettings((prev) => ({ ...prev, spinnerTipsEnabled: next }));
                trackEvent("tengu_tips_setting_changed", { enabled: next });
            }
        },
        {
            id: "thinkingEnabled",
            label: "Thinking mode",
            value: thinkingEnabled,
            type: "boolean",
            onChange(next) {
                setAppState((prev: any) => ({ ...prev, thinkingEnabled: next }));
                updateUserSettings("userSettings", { alwaysThinkingEnabled: next ? undefined : false });
                trackEvent("tengu_thinking_toggled", { enabled: next });
            }
        },
        ...(isFeatureEnabled("tengu_prompt_suggestion")
            ? [
                  {
                      id: "promptSuggestionEnabled",
                      label: "Prompt suggestions",
                      value: promptSuggestionEnabled,
                      type: "boolean",
                      onChange(next) {
                          setAppState((prev: any) => ({ ...prev, promptSuggestionEnabled: next }));
                          updateUserSettings("userSettings", { promptSuggestionEnabled: next ? undefined : false });
                      }
                  } as ConfigItem
              ]
            : []),
        ...(allowCheckpointing
            ? [
                  {
                      id: "fileCheckpointingEnabled",
                      label: "Rewind code (checkpoints)",
                      value: settings.fileCheckpointingEnabled,
                      type: "boolean",
                      onChange(next) {
                          updateSettings((prev) => ({ ...prev, fileCheckpointingEnabled: next }));
                          setSettings({ ...getSettings(), fileCheckpointingEnabled: next });
                          trackEvent("tengu_file_history_snapshots_setting_changed", { enabled: next });
                      }
                  } as ConfigItem
              ]
            : []),
        {
            id: "verbose",
            label: "Verbose output",
            value: verbose,
            type: "boolean",
            onChange: updateVerboseSetting
        },
        {
            id: "terminalProgressBarEnabled",
            label: "Terminal progress bar",
            value: settings.terminalProgressBarEnabled,
            type: "boolean",
            onChange(next) {
                updateSettings((prev) => ({ ...prev, terminalProgressBarEnabled: next }));
                setSettings({ ...getSettings(), terminalProgressBarEnabled: next });
                trackEvent("tengu_terminal_progress_bar_setting_changed", { enabled: next });
            }
        },
        {
            id: "defaultPermissionMode",
            label: "Default permission mode",
            value: userSettings.permissions?.defaultMode || "default",
            options: PERMISSION_MODES.filter((mode) => !["bypassPermissions"].includes(mode)),
            type: "enum",
            onChange(next) {
                const parsed = parsePermissionMode(next);
                const result = updateUserSettings("userSettings", {
                    permissions: {
                        ...userSettings.permissions,
                        defaultMode: parsed
                    }
                });
                if (result.error) {
                    trackEvent("tengu_config_changed", { error: String(result.error) });
                    return;
                }
                setUserSettings((prev) => ({
                    ...prev,
                    permissions: {
                        ...prev.permissions,
                        defaultMode: parsed
                    }
                }));
                setChangeLog((prev) => ({ ...prev, defaultPermissionMode: next }));
                trackEvent("tengu_config_changed", { setting: "defaultPermissionMode", value: next });
            }
        },
        {
            id: "respectGitignore",
            label: "Respect .gitignore in file picker",
            value: settings.respectGitignore,
            type: "boolean",
            onChange(next) {
                updateSettings((prev) => ({ ...prev, respectGitignore: next }));
                setSettings({ ...getSettings(), respectGitignore: next });
                trackEvent("tengu_respect_gitignore_setting_changed", { enabled: next });
            }
        },
        ...(managedAutoUpdatesReason || isFeatureEnabled(undefined)
            ? [
                  {
                      id: "autoUpdatesChannel",
                      label: "Auto-update channel",
                      value: managedAutoUpdatesReason ? "disabled" : userSettings.autoUpdatesChannel ?? "latest",
                      type: "managedEnum",
                      onChange() {}
                  } as ConfigItem
              ]
            : []),
        {
            id: "theme",
            label: "Theme",
            value: theme,
            type: "managedEnum",
            onChange: setTheme
        },
        {
            id: "notifChannel",
            label: "Notifications",
            value: settings.preferredNotifChannel,
            options: [
                "auto",
                "iterm2",
                "terminal_bell",
                "iterm2_with_bell",
                "kitty",
                "ghostty",
                "notifications_disabled"
            ],
            type: "enum",
            onChange(next) {
                updateSettings((prev) => ({ ...prev, preferredNotifChannel: next }));
                setSettings({ ...getSettings(), preferredNotifChannel: next });
            }
        },
        {
            id: "outputStyle",
            label: "Output style",
            value: outputStyle,
            type: "managedEnum",
            onChange: () => {}
        },
        {
            id: "editorMode",
            label: "Editor mode",
            value: settings.editorMode === "emacs" ? "normal" : settings.editorMode || "normal",
            options: ["normal", "vim"],
            type: "enum",
            onChange(next) {
                updateSettings((prev) => ({ ...prev, editorMode: next }));
                setSettings({ ...getSettings(), editorMode: next });
                trackEvent("tengu_editor_mode_changed", { mode: next, source: "config_panel" });
            }
        },
        {
            id: "model",
            label: "Model",
            value: formatModelValue(mainLoopModel),
            type: "managedEnum",
            onChange: updateModelSetting
        },
        ...(hasClients
            ? [
                  {
                      id: "diffTool",
                      label: "Diff tool",
                      value: settings.diffTool ?? "auto",
                      options: ["terminal", "auto"],
                      type: "enum",
                      onChange(next) {
                          updateSettings((prev) => ({ ...prev, diffTool: next }));
                          setSettings({ ...getSettings(), diffTool: next });
                          trackEvent("tengu_diff_tool_changed", { tool: next, source: "config_panel" });
                      }
                  } as ConfigItem
              ]
            : []),
        ...(!isExternalTerminal()
            ? [
                  {
                      id: "autoConnectIde",
                      label: "Auto-connect to IDE (external terminal)",
                      value: settings.autoConnectIde ?? false,
                      type: "boolean",
                      onChange(next) {
                          updateSettings((prev) => ({ ...prev, autoConnectIde: next }));
                          setSettings({ ...getSettings(), autoConnectIde: next });
                          trackEvent("tengu_auto_connect_ide_changed", { enabled: next, source: "config_panel" });
                      }
                  } as ConfigItem
              ]
            : []),
        ...(isExternalTerminal()
            ? [
                  {
                      id: "autoInstallIdeExtension",
                      label: "Auto-install IDE extension",
                      value: settings.autoInstallIdeExtension ?? true,
                      type: "boolean",
                      onChange(next) {
                          updateSettings((prev) => ({ ...prev, autoInstallIdeExtension: next }));
                          setSettings({ ...getSettings(), autoInstallIdeExtension: next });
                          trackEvent("tengu_auto_install_ide_extension_changed", {
                              enabled: next,
                              source: "config_panel"
                          });
                      }
                  } as ConfigItem
              ]
            : []),
        {
            id: "claudeInChromeDefaultEnabled",
            label: "Claude in Chrome enabled by default",
            value: settings.claudeInChromeDefaultEnabled ?? true,
            type: "boolean",
            onChange(next) {
                updateSettings((prev) => ({ ...prev, claudeInChromeDefaultEnabled: next }));
                setSettings({ ...getSettings(), claudeInChromeDefaultEnabled: next });
                trackEvent("tengu_claude_in_chrome_setting_changed", { enabled: next });
            }
        },
        ...(allowExternalIncludes
            ? [
                  {
                      id: "showExternalIncludesDialog",
                      label: "External CLAUDE.md includes",
                      value: getUserSettings().permissions?.defaultMode ? "true" : "false",
                      type: "managedEnum",
                      onChange() {}
                  } as ConfigItem
              ]
            : []),
        ...(process.env.ANTHROPIC_API_KEY
            ? [
                  {
                      id: "apiKey",
                      label: (
                          <Text>
                              Use custom API key: <Text bold>{maskApiKey(process.env.ANTHROPIC_API_KEY)}</Text>
                          </Text>
                      ),
                      value: Boolean(
                          process.env.ANTHROPIC_API_KEY &&
                              settings.customApiKeyResponses?.approved?.includes(maskApiKey(process.env.ANTHROPIC_API_KEY))
                      ),
                      type: "boolean",
                      onChange(next) {
                          updateSettings((prev) => {
                              const updated: Settings = { ...prev };
                              if (!updated.customApiKeyResponses) {
                                  updated.customApiKeyResponses = { approved: [], rejected: [] };
                              }
                              if (!updated.customApiKeyResponses.approved) updated.customApiKeyResponses.approved = [];
                              if (!updated.customApiKeyResponses.rejected) updated.customApiKeyResponses.rejected = [];
                              if (process.env.ANTHROPIC_API_KEY) {
                                  const masked = maskApiKey(process.env.ANTHROPIC_API_KEY);
                                  if (next) {
                                      updated.customApiKeyResponses.approved = [
                                          ...(updated.customApiKeyResponses.approved ?? []).filter((entry) => entry !== masked),
                                          masked
                                      ];
                                      updated.customApiKeyResponses.rejected =
                                          (updated.customApiKeyResponses.rejected ?? []).filter((entry) => entry !== masked);
                                  } else {
                                      updated.customApiKeyResponses.approved =
                                          (updated.customApiKeyResponses.approved ?? []).filter((entry) => entry !== masked);
                                      updated.customApiKeyResponses.rejected = [
                                          ...(updated.customApiKeyResponses.rejected ?? []).filter((entry) => entry !== masked),
                                          masked
                                      ];
                                  }
                              }
                              return updated;
                          });
                          setSettings(getSettings());
                      }
                  } as ConfigItem
              ]
            : [])
    ];

    useInput((input, key) => {
        if (key.escape && activeModal === null) {
            const changes = Object.entries(changeLog).map(([keyName, value]) => {
                trackEvent("tengu_config_changed", { key: keyName, value });
                return `Set ${keyName} to ${String(value)}`;
            });
            const wasApiKeyEnabled = Boolean(
                process.env.ANTHROPIC_API_KEY &&
                    settingsRef.current.customApiKeyResponses?.approved?.includes(
                        maskApiKey(process.env.ANTHROPIC_API_KEY)
                    )
            );
            const isApiKeyEnabled = Boolean(
                process.env.ANTHROPIC_API_KEY &&
                    settings.customApiKeyResponses?.approved?.includes(maskApiKey(process.env.ANTHROPIC_API_KEY))
            );

            if (wasApiKeyEnabled !== isApiKeyEnabled) {
                changes.push(`${isApiKeyEnabled ? "Enabled" : "Disabled"} custom API key`);
                trackEvent("tengu_config_changed", { key: "env.ANTHROPIC_API_KEY", value: isApiKeyEnabled });
            }
            if (settings.theme !== settingsRef.current.theme) {
                changes.push(`Set theme to ${String(settings.theme)}`);
            }
            if (settings.preferredNotifChannel !== settingsRef.current.preferredNotifChannel) {
                changes.push(`Set notifications to ${String(settings.preferredNotifChannel)}`);
            }
            if (outputStyle !== outputStyleRef.current) {
                changes.push(`Set output style to ${String(outputStyle)}`);
            }
            if (settings.editorMode !== settingsRef.current.editorMode) {
                changes.push(`Set editor mode to ${String(settings.editorMode || "emacs")}`);
            }
            if (settings.diffTool !== settingsRef.current.diffTool) {
                changes.push(`Set diff tool to ${String(settings.diffTool)}`);
            }
            if (settings.autoConnectIde !== settingsRef.current.autoConnectIde) {
                changes.push(`${settings.autoConnectIde ? "Enabled" : "Disabled"} auto-connect to IDE`);
            }
            if (settings.autoInstallIdeExtension !== settingsRef.current.autoInstallIdeExtension) {
                changes.push(`${settings.autoInstallIdeExtension ? "Enabled" : "Disabled"} auto-install IDE extension`);
            }
            if (settings.autoCompactEnabled !== settingsRef.current.autoCompactEnabled) {
                changes.push(`${settings.autoCompactEnabled ? "Enabled" : "Disabled"} auto-compact`);
            }
            if (settings.respectGitignore !== settingsRef.current.respectGitignore) {
                changes.push(`${settings.respectGitignore ? "Enabled" : "Disabled"} respect .gitignore in file picker`);
            }
            if (settings.terminalProgressBarEnabled !== settingsRef.current.terminalProgressBarEnabled) {
                changes.push(`${settings.terminalProgressBarEnabled ? "Enabled" : "Disabled"} terminal progress bar`);
            }
            if (userSettings.autoUpdatesChannel !== userSettingsRef.current.autoUpdatesChannel) {
                changes.push(`Set auto-update channel to ${String(userSettings.autoUpdatesChannel ?? "latest")}`);
            }

            if (changes.length > 0) {
                onClose(changes.join("\n"));
            } else {
                onClose("Config dialog dismissed", { display: "system" });
            }
            return;
        }
        if (activeModal !== null) return;

        function handleItemAction() {
            const item = configItems[selectedIndex];
            if (!item || !item.onChange) return;
            if (item.type === "boolean") {
                item.onChange(!item.value);
                return;
            }
            if (item.id === "theme" && key.return) {
                setActiveModal(0);
                setTabsHidden(true);
                setHideMargin(true);
                return;
            }
            if (item.id === "model" && key.return) {
                setActiveModal(1);
                setTabsHidden(true);
                return;
            }
            if (item.id === "showExternalIncludesDialog" && key.return) {
                setActiveModal(2);
                setTabsHidden(true);
                setIsWarning(true);
                return;
            }
            if (item.id === "outputStyle" && key.return) {
                setActiveModal(3);
                setTabsHidden(true);
                return;
            }
            if (item.id === "autoUpdatesChannel" && key.return) {
                if ((userSettings.autoUpdatesChannel ?? "latest") === "latest") {
                    setActiveModal(4);
                    setTabsHidden(true);
                } else {
                    updateUserSettings("userSettings", { autoUpdatesChannel: "latest", minimumVersion: undefined });
                    setUserSettings((prev) => ({ ...prev, autoUpdatesChannel: "latest", minimumVersion: undefined }));
                    trackEvent("tengu_autoupdate_channel_changed", { channel: "latest" });
                }
                return;
            }
            if (item.type === "enum" && item.options) {
                const nextIndex = (item.options.indexOf(item.value) + 1) % item.options.length;
                item.onChange(item.options[nextIndex]);
            }
        }

        if (key.return || input === " ") {
            handleItemAction();
            return;
        }
        if (key.upArrow || (key.ctrl && input === "p") || (!key.ctrl && !key.shift && input === "k")) {
            setSelectedIndex((value) => Math.max(0, value - 1));
        }
        if (key.downArrow || (key.ctrl && input === "n") || (!key.ctrl && !key.shift && input === "j")) {
            setSelectedIndex((value) => Math.min(configItems.length - 1, value + 1));
        }
    });

    if (activeModal === 0) {
        return (
            <>
                <ThemeSelector
                    initialTheme={theme}
                    onThemeSelect={(nextTheme) => {
                        setTheme(nextTheme);
                        setActiveModal(null);
                        setHideMargin(false);
                        setTabsHidden(false);
                    }}
                    onCancel={() => {
                        setActiveModal(null);
                        setHideMargin(false);
                        setTabsHidden(false);
                    }}
                />
                <Box marginLeft={1}>
                    <Text dimColor italic>
                        <ShortcutGroup>
                            <ShortcutHint shortcut="Enter" action="select" />
                            <ShortcutHint shortcut="Esc" action="cancel" />
                        </ShortcutGroup>
                    </Text>
                </Box>
            </>
        );
    }

    if (activeModal === 1) {
        return (
            <>
                <ModelPicker
                    initial={mainLoopModel}
                    onSelect={(model) => {
                        updateModelSetting(model);
                        setActiveModal(null);
                        setTabsHidden(false);
                    }}
                    onCancel={() => {
                        setActiveModal(null);
                        setTabsHidden(false);
                    }}
                />
                <Text dimColor>
                    <ShortcutGroup>
                        <ShortcutHint shortcut="Enter" action="confirm" />
                        <ShortcutHint shortcut="Esc" action="cancel" />
                    </ShortcutGroup>
                </Text>
            </>
        );
    }

    if (activeModal === 2) {
        return (
            <>
                <ExternalIncludesDialog
                    onDone={() => {
                        setActiveModal(null);
                        setTabsHidden(false);
                        setIsWarning(false);
                    }}
                />
                <Text dimColor>
                    <ShortcutGroup>
                        <ShortcutHint shortcut="Enter" action="confirm" />
                        <ShortcutHint shortcut="Esc" action="disable external includes" />
                    </ShortcutGroup>
                </Text>
            </>
        );
    }

    if (activeModal === 3) {
        return (
            <>
                <OutputStylePicker
                    initialStyle={outputStyle}
                    onComplete={(nextStyle) => {
                        setOutputStyle(nextStyle ?? DEFAULT_OUTPUT_STYLE);
                        setActiveModal(null);
                        setTabsHidden(false);
                        updateUserSettings("localSettings", { outputStyle: nextStyle });
                        trackEvent("tengu_output_style_changed", {
                            style: nextStyle ?? DEFAULT_OUTPUT_STYLE,
                            source: "config_panel",
                            settings_source: "localSettings"
                        });
                    }}
                    onCancel={() => {
                        setActiveModal(null);
                        setTabsHidden(false);
                    }}
                />
                <Text dimColor>
                    <ShortcutGroup>
                        <ShortcutHint shortcut="Enter" action="confirm" />
                        <ShortcutHint shortcut="Esc" action="cancel" />
                    </ShortcutGroup>
                </Text>
            </>
        );
    }

    if (activeModal === 4) {
        return (
            <AutoUpdateChannelDialog
                currentVersion="2.0.76"
                onChoice={(choice) => {
                    setActiveModal(null);
                    setTabsHidden(false);
                    if (choice === "cancel") return;
                    const nextSettings: UserSettings = { autoUpdatesChannel: "stable" };
                    if (choice === "stay") nextSettings.minimumVersion = "2.0.76";
                    updateUserSettings("userSettings", nextSettings as any);
                    setUserSettings((prev) => ({ ...prev, ...nextSettings }));
                    trackEvent("tengu_autoupdate_channel_changed", {
                        channel: "stable",
                        minimum_version_set: choice === "stay"
                    });
                }}
            />
        );
    }

    return (
        <Box flexDirection="column" width="100%" marginY={1} gap={1}>
            <Text>Configure Claude Code preferences</Text>
            <Box flexDirection="column">
                {configItems.map((item, index) => {
                    const isSelected = index === selectedIndex;
                    return (
                        <Box key={item.id} flexDirection="row">
                            <Box width={44}>
                                <Text color={isSelected ? "suggestion" : undefined}>
                                    {isSelected ? figures.pointer : " "} {item.label}
                                </Text>
                            </Box>
                            <Box>
                                {item.type === "boolean" ? (
                                    <Text color={isSelected ? "suggestion" : undefined}>{String(item.value)}</Text>
                                ) : item.id === "theme" ? (
                                    <Text color={isSelected ? "suggestion" : undefined}>
                                        {THEME_LABELS[String(item.value)] || String(item.value)}
                                    </Text>
                                ) : item.id === "notifChannel" ? (
                                    <Text color={isSelected ? "suggestion" : undefined}>
                                        {(() => {
                                            switch (String(item.value)) {
                                                case "auto":
                                                    return "Auto";
                                                case "iterm2":
                                                    return "iTerm2 (OSC 9)";
                                                case "terminal_bell":
                                                    return "Terminal Bell (\\a)";
                                                case "kitty":
                                                    return "Kitty (OSC 99)";
                                                case "ghostty":
                                                    return "Ghostty (OSC 777)";
                                                case "iterm2_with_bell":
                                                    return "iTerm2 w/ Bell";
                                                case "notifications_disabled":
                                                    return "Disabled";
                                                default:
                                                    return String(item.value);
                                            }
                                        })()}
                                    </Text>
                                ) : item.id === "defaultPermissionMode" ? (
                                    <Text color={isSelected ? "suggestion" : undefined}>
                                        {formatPermissionMode(String(item.value))}
                                    </Text>
                                ) : item.id === "autoUpdatesChannel" && managedAutoUpdatesReason ? (
                                    <Box flexDirection="column">
                                        <Text color={isSelected ? "suggestion" : undefined}>disabled</Text>
                                        <Text dimColor>({managedAutoUpdatesReason})</Text>
                                    </Box>
                                ) : (
                                    <Text color={isSelected ? "suggestion" : undefined}>{String(item.value)}</Text>
                                )}
                            </Box>
                        </Box>
                    );
                })}
            </Box>
            <Text dimColor>
                <ShortcutGroup>
                    <ShortcutHint shortcut="Enter/Space" action="change" />
                    <ShortcutHint shortcut="Esc" action="cancel" />
                </ShortcutGroup>
            </Text>
        </Box>
    );
}
