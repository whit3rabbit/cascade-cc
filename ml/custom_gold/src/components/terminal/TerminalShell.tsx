import React, { useState, useEffect } from "react";
import { Box, useApp } from "ink";
import { useAgentMainLoop } from "../../services/terminal/AgentMainLoop.js";
import { TerminalInteractionController } from "./TerminalController.js";
import { ChatFeed } from "./ChatFeed.js";
import { useAppState } from "../../contexts/AppStateContext.js";

// Placeholder for commands if not loaded from state
const DEFAULT_COMMANDS: any[] = [
    { name: "/help", description: "Show help" },
    { name: "/clear", description: "Clear the screen" },
    { name: "/compact", description: "Compact conversation history" }
];

/**
 * The primary TUI container for Claude Code.
 * Renders the ChatFeed (history) and the TerminalInteractionController (input area).
 */
export const TerminalShell: React.FC<{
    initialPrompt?: string;
    initialMessages?: any[];
    options?: any;
    children?: React.ReactNode;
}> = ({ initialPrompt, initialMessages = [], options }) => {
    const { exit } = useApp();
    const [appState, setAppState] = useAppState();

    // Core Agent Loop
    const {
        messages,
        isResponding,
        onQuery,
        onInterrupt
    } = useAgentMainLoop(initialMessages);

    // Local UI State
    const [input, setInput] = useState("");
    const [mode, setMode] = useState("prompt");
    const [cursorOffset, setCursorOffset] = useState(0); // Managed by controller usually, but needed for some props
    const [pastedContents, setPastedContents] = useState<Record<string, any>>({});
    const [vimMode, setVimMode] = useState("insert");
    const [showBashesDialog, setShowBashesDialog] = useState(false);
    const [showDiffDialog, setShowDiffDialog] = useState(false);
    const [tasksSelected, setTasksSelected] = useState(false);
    const [diffSelected, setDiffSelected] = useState(false);
    const [isSearchingHistory, setIsSearchingHistory] = useState(false);
    const [stashedPrompt, setStashedPrompt] = useState<{ text: string; cursorOffset: number } | undefined>(undefined);
    const [submitCount, setSubmitCount] = useState(0);

    // Initial prompt handling
    useEffect(() => {
        if (initialPrompt && submitCount === 0) {
            onQuery(initialPrompt);
            setSubmitCount(prev => prev + 1);
        }
    }, []);

    const handleSubmit = async (value: string, ctx: any) => {
        // Clear input state
        setInput("");
        setPastedContents({});
        setSubmitCount(prev => prev + 1);
        setMode("prompt");

        // Execute query
        await onQuery(value);

        // Context callbacks if needed
        if (ctx?.resetHistory) ctx.resetHistory();
    };

    const handleExit = () => {
        exit();
    };

    // Derived props
    const toolPermissionContext = appState.toolPermissionContext;
    const setToolPermissionContext = (ctx: any) => {
        setAppState(prev => ({ ...prev, toolPermissionContext: ctx }));
    };

    // Tools wrapper
    const tools = appState.mcp?.tools?.map((t: any) => ({ name: t.name })) || [];

    return (
        <Box flexDirection="column" height="100%" width="100%">
            <Box flexGrow={1} flexDirection="column" width="100%">
                <ChatFeed
                    messages={messages}
                    normalizedMessageHistory={[]}
                    tools={tools}
                    commands={DEFAULT_COMMANDS}
                    verbose={appState.verbose}
                    toolJSX={null}
                    toolUseConfirmQueue={[]}
                    inProgressToolUseIDs={new Set()}
                    isMessageSelectorVisible={false}
                    conversationId="primary-session"
                    screen="feed"
                    screenToggleId="0"
                    streamingToolUses={[]}
                    agentDefinitions={appState.agentDefinitions}
                    hideLogo={false} // This triggers AppDashboard -> Banner
                    isLoading={isResponding}
                />
            </Box>

            <Box flexShrink={0} width="100%">
                <TerminalInteractionController
                    debug={appState.verbose}
                    ideSelection={null}
                    toolPermissionContext={toolPermissionContext}
                    setToolPermissionContext={setToolPermissionContext}
                    apiKeyStatus="active" // or derived from state
                    commands={DEFAULT_COMMANDS}
                    agents={appState.agentDefinitions?.activeAgents || []}
                    isLoading={isResponding}
                    verbose={appState.verbose}
                    messages={messages}
                    onAutoUpdaterResult={() => { }}
                    autoUpdaterResult={null}
                    input={input}
                    onInputChange={setInput}
                    mode={mode}
                    onModeChange={setMode}
                    stashedPrompt={stashedPrompt}
                    setStashedPrompt={setStashedPrompt}
                    submitCount={submitCount}
                    onShowMessageSelector={() => { }}
                    mcpClients={appState.mcp.clients}
                    pastedContents={pastedContents}
                    setPastedContents={setPastedContents}
                    vimMode={vimMode}
                    setVimMode={setVimMode}
                    showBashesDialog={showBashesDialog}
                    setShowBashesDialog={setShowBashesDialog}
                    showDiffDialog={showDiffDialog}
                    setShowDiffDialog={setShowDiffDialog}
                    tasksSelected={tasksSelected}
                    setTasksSelected={setTasksSelected}
                    diffSelected={diffSelected}
                    setDiffSelected={setDiffSelected}
                    onExit={handleExit}
                    getToolUseContext={() => ({})}
                    onSubmit={handleSubmit}
                    isSearchingHistory={isSearchingHistory}
                    setIsSearchingHistory={setIsSearchingHistory}
                />
            </Box>
        </Box>
    );
};
