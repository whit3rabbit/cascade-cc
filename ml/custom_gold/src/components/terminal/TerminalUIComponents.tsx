import React from "react";
import { Box, Text } from "ink";
import { figures } from "../../vendor/terminalFigures.js";
// import { Link } from "../../vendor/inkLink.js";
import { TasksDialog } from "./TasksDialog.js";
import { useAppState } from "../../contexts/AppStateContext.js";
// import { MessageViewAdapter } from "./MessageAdapter.js";
import { Select } from "../shared/Select.js";

import { TerminalInput } from "./TerminalInput.js";
export { Separator } from "./Separator.js";

// -- Stubs or Imports --
export const BashDialog = TasksDialog;

// -- Helper Components --

function AgentModeIndicator({ isLoading, agentName, themeColor }: { isLoading?: boolean; agentName?: string; themeColor?: string }) {
    const displayColor = themeColor;
    const prefix = agentName ? `[${agentName}]` : "";
    return (
        <Text color={displayColor} dimColor={isLoading}>
            {prefix}{figures.pointer}{"\u00A0"}
        </Text>
    );
}

function ImageThumbnail({ imageId, backgroundColor, isSelected = false }: { imageId: string; backgroundColor?: string; isSelected?: boolean }) {
    const label = `[Image #${imageId}]`;
    return (
        <Text backgroundColor={backgroundColor} inverse={isSelected} bold={isSelected} underline={isSelected}>
            {label}
        </Text>
    );
}

// -- Exported Components --

export const PromptModeIndicator: React.FC<{ mode: string; isLoading?: boolean; hideAgentPrefix?: boolean }> = ({ mode, isLoading, hideAgentPrefix: _hideAgentPrefix }) => {
    return (
        <Box
            alignItems="flex-start"
            alignSelf="flex-start"
            flexWrap="nowrap"
            justifyContent="flex-start"
        >
            {mode === "bash" ? (
                <Text color="red" dimColor={isLoading}>
                    !{"\u00A0"}
                </Text>
            ) : mode === "background" ? (
                <Text color="blue" dimColor={isLoading}>
                    &{"\u00A0"}
                </Text>
            ) : (
                <AgentModeIndicator isLoading={isLoading} />
            )}
        </Box>
    );
};

export const VimTerminalInput: React.FC<any> = (props) => {
    return <TerminalInput {...props} vimMode={true} />;
};

export const PreInputOverlays: React.FC<{ isLoading?: boolean }> = ({ isLoading }) => {
    const [state] = useAppState();
    const { queuedCommands } = state;

    if (isLoading) {
        return (
            <Box paddingX={2} marginBottom={1} width="100%">
                <Text dimColor>Thinking...</Text>
            </Box>
        );
    }

    if (!queuedCommands || queuedCommands.length === 0) return null;

    // Render queued commands overlay
    return (
        <Box flexDirection="column" marginTop={1}>
            {queuedCommands.map((cmd: any, index: number) => (
                <Box key={index} paddingX={2}>
                    <Text dimColor>Queued: </Text>
                    <Text>{cmd.value}</Text>
                </Box>
            ))}
        </Box>
    );
};

export const TopPromptAddon: React.FC<any> = () => null;

export const PromptStashIndicator: React.FC<{ hasStash: boolean }> = ({ hasStash }) => {
    if (!hasStash) return null;
    return (
        <Box paddingLeft={2}>
            <Text dimColor>
                {figures.pointer || ">"} Stashed (auto-restores after submit)
            </Text>
        </Box>
    );
};

export const PastedContentIndicator: React.FC<any> = ({ pastedContents, isSelected = false, selectedIndex = 0 }) => {
    const images = Object.values(pastedContents || {}).filter((c: any) => c.type === "image");
    if (images.length === 0) return null;

    const hint = isSelected
        ? images.length > 1
            ? "(←/→ select · backspace remove · ↓ cancel)"
            : "(backspace remove · ↓ cancel)"
        : "(↑ to select)";

    return (
        <Box flexDirection="row" gap={1} paddingX={1} flexWrap="wrap">
            {images.map((img: any, idx: number) => (
                <ImageThumbnail
                    key={img.id}
                    imageId={img.id}
                    isSelected={isSelected && idx === selectedIndex}
                />
            ))}
            <Text dimColor>{hint}</Text>
        </Box>
    );
};

export const ModelPicker: React.FC<any> = ({ initial: _initial, sessionModel, onSelect, onCancel: _onCancel, isStandaloneCommand }) => {
    return (
        <Box flexDirection="column" width="100%">
            {isStandaloneCommand && (
                <Box borderStyle="single" borderTop={false} borderLeft={false} borderRight={false} borderColor="gray" />
            )}
            <Box flexDirection="column" paddingX={isStandaloneCommand ? 1 : 0}>
                <Box marginBottom={1} flexDirection="column">
                    <Text color="blue" bold>Select model</Text>
                    <Text dimColor>Switch between Claude models.</Text>
                    {sessionModel && <Text dimColor>Currently using {sessionModel} for this session.</Text>}
                </Box>
                <Select
                    options={[{ label: "Claude 3.5 Sonnet", value: "claude-3-5-sonnet" }]}
                    onChange={onSelect}
                />
            </Box>
        </Box>
    );
};

export const ThinkingModePicker: React.FC<any> = ({ currentValue: _currentValue, onSelect, onCancel: _onCancel, isMidConversation }) => {
    return (
        <Box flexDirection="column" width="100%">
            <Box marginBottom={1} flexDirection="column">
                <Text color="blue" bold>Toggle thinking mode</Text>
                <Text dimColor>Enable or disable thinking for this session.</Text>
                {isMidConversation && (
                    <Text color="yellow">Changing mid-conversation may reduce quality.</Text>
                )}
            </Box>
            <Select
                options={[
                    { label: "Enabled", value: "true" },
                    { label: "Disabled", value: "false" }
                ]}
                onChange={(val: string) => onSelect(val === "true")}
            />
        </Box>
    );
};

// --- Terminal Status Line Components ---

function HelpOverlay({ dimColor, fixedWidth, paddingX }: any) {
    return (
        <Box paddingX={paddingX} flexDirection="row" gap={2}>
            <Box flexDirection="column" width={fixedWidth ? 22 : undefined}>
                <Text dimColor={dimColor}>! for bash mode</Text>
                <Text dimColor={dimColor}>/ for commands</Text>
                <Text dimColor={dimColor}>@ for file paths</Text>
                <Text dimColor={dimColor}>& for background</Text>
            </Box>
            <Box flexDirection="column" width={fixedWidth ? 35 : undefined}>
                <Text dimColor={dimColor}>double tap esc to clear input</Text>
                <Text dimColor={dimColor}>Tab to auto-accept edits</Text>
                <Text dimColor={dimColor}>ctrl + o for verbose output</Text>
                <Text dimColor={dimColor}>ctrl + t to show todos</Text>
            </Box>
            <Box flexDirection="column">
                <Text dimColor={dimColor}>ctrl + _ to undo</Text>
                <Text dimColor={dimColor}>ctrl + z to suspend</Text>
                <Text dimColor={dimColor}>ctrl + v to paste images</Text>
                <Text dimColor={dimColor}>ctrl + k to switch model</Text>
                <Text dimColor={dimColor}>ctrl + s to stash prompt</Text>
            </Box>
        </Box>
    );
}

function StatusLineMain({ exitMessage, isPasting }: any) {
    if (exitMessage?.show) {
        return <Text color="error" bold>Exiting... {exitMessage.key}</Text>;
    }
    if (isPasting) {
        return <Text color="warning" bold>PASTING...</Text>;
    }
    return (
        <Box flexDirection="row" gap={1}>
            <Text dimColor>? for shortcuts</Text>
        </Box>
    );
}

function StatusLineRight({ apiKeyStatus, autoUpdaterResult, verbose }: any) {
    return (
        <Box flexDirection="row" gap={2}>
            {verbose && (
                <Text color="background" backgroundColor="blue" bold> VERBOSE </Text>
            )}
            {autoUpdaterResult && (
                <Text color="background" backgroundColor="green" bold> UPDATE AVAILABLE </Text>
            )}
            <Box gap={1}>
                <Text dimColor>API:</Text>
                <Text color={apiKeyStatus === "active" ? "success" : "error"} bold>
                    {String(apiKeyStatus || "UNKNOWN").toUpperCase()}
                </Text>
            </Box>
        </Box>
    );
}

export const TerminalStatusLine: React.FC<any> = ({
    helpOpen,
    suggestions,
    selectedSuggestion,
    ...props
}) => {
    // If suggestions are present, show them (stubbed for now logic)
    if (suggestions && suggestions.length > 0) {
        return (
            <Box paddingX={2} flexDirection="column">
                {suggestions.map((s: any, i: number) => (
                    <Text key={i} color={i === selectedSuggestion ? "blue" : undefined}>
                        {s.label || s.displayText} {s.description ? <Text dimColor>({s.description})</Text> : ""}
                    </Text>
                ))}
            </Box>
        );
    }

    // If help is open button pressed (?)
    if (helpOpen) {
        return <HelpOverlay dimColor={true} fixedWidth={true} paddingX={2} />;
    }

    // Default status line
    return (
        <Box
            flexDirection="row"
            justifyContent="space-between"
            paddingX={1}
            width="100%"
        >
            <StatusLineMain {...props} />
            <StatusLineRight {...props} />
        </Box>
    );
};
