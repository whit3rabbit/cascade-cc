import * as React from "react";
import { TerminalEventEmitter } from "../utils/shared/eventEmitter.js";
import { ThemeProvider } from "../services/terminal/themeManager.js";
import { ErrorDisplay } from "../services/terminal/errorMessage.js";
import {
    InternalAppContext,
    InternalStdinContext,
    InternalFocusContext
} from "./inkContexts.js";
import { parseTerminalInput, KeypressEvent } from "./terminalInputParser.js";
import { SHOW_CURSOR, HIDE_CURSOR, ENABLE_BRACKETED_PASTE, DISABLE_BRACKETED_PASTE } from "./terminalSequences.js";

export interface TerminalAppProps {
    stdin: NodeJS.ReadStream;
    stdout: NodeJS.WriteStream;
    stderr: NodeJS.WriteStream;
    initialTheme: any;
    exitOnCtrlC: boolean;
    onExit: (error?: Error) => void;
    terminalColumns: number;
    terminalRows: number;
    ink2?: boolean;
    children: React.ReactNode;
}

export interface TerminalAppState {
    isFocusEnabled: boolean;
    activeFocusId?: string;
    focusables: { id: string; isActive: boolean }[];
    error?: Error;
}

/**
 * Root React component for the CLI UI.
 * Deobfuscated from KeA in chunk_203.ts.
 */
export class TerminalApp extends React.PureComponent<TerminalAppProps, TerminalAppState> {
    static displayName = "InternalApp";

    state: TerminalAppState = {
        isFocusEnabled: true,
        activeFocusId: undefined,
        focusables: [],
        error: undefined
    };

    private rawModeEnabledCount = 0;
    private internal_eventEmitter = new TerminalEventEmitter();
    private keyParseState = { mode: "NORMAL", incomplete: "", pasteBuffer: "" };

    static getDerivedStateFromError(error: Error) {
        return { error };
    }

    isRawModeSupported() {
        return this.props.stdin.isTTY;
    }

    componentDidMount() {
        if (this.props.stdout.isTTY) this.props.stdout.write(HIDE_CURSOR);
    }

    componentWillUnmount() {
        if (this.props.stdout.isTTY) this.props.stdout.write(SHOW_CURSOR);
        if (this.isRawModeSupported()) this.handleSetRawMode(false);
    }

    componentDidCatch(error: Error) {
        this.handleExit(error);
    }

    handleSetRawMode = (enabled: boolean) => {
        const { stdin, stdout } = this.props;
        if (!this.isRawModeSupported()) return;

        if (enabled) {
            if (this.rawModeEnabledCount === 0) {
                stdin.setRawMode(true);
                stdin.on("readable", this.handleReadable);
                stdout.write(ENABLE_BRACKETED_PASTE);
            }
            this.rawModeEnabledCount++;
        } else {
            if (--this.rawModeEnabledCount === 0) {
                stdout.write(DISABLE_BRACKETED_PASTE);
                stdin.setRawMode(false);
                stdin.off("readable", this.handleReadable);
            }
        }
    };

    handleReadable = () => {
        let chunk;
        while ((chunk = this.props.stdin.read()) !== null) {
            this.processInput(chunk);
        }
    };

    processInput = (data: string | Buffer | null) => {
        const [keys, newState] = parseTerminalInput(this.keyParseState, data || "");
        this.keyParseState = newState;

        for (const key of keys) {
            this.handleInput(key.sequence);
            this.internal_eventEmitter.emit("input", new KeypressEvent(key));
        }
    };

    handleInput = (input: string) => {
        if (input === "\x03" && this.props.exitOnCtrlC) this.handleExit();
        // Handle focus navigation, suspend, etc.
    };

    handleExit = (error?: Error) => {
        if (this.isRawModeSupported()) this.handleSetRawMode(false);
        this.props.onExit(error);
    };

    // Focus management helpers...
    addFocusable = (id: string, options: { autoFocus?: boolean } = {}) => {
        this.setState(s => ({
            activeFocusId: !s.activeFocusId && options.autoFocus ? id : s.activeFocusId,
            focusables: [...s.focusables, { id, isActive: true }]
        }));
    };

    render() {
        const { error } = this.state;
        return (
            <InternalAppContext.Provider value={{ exit: this.handleExit }}>
                <ThemeProvider initialState={this.props.initialTheme}>
                    <InternalStdinContext.Provider value={{
                        stdin: this.props.stdin,
                        setRawMode: this.handleSetRawMode,
                        isRawModeSupported: this.isRawModeSupported(),
                        internal_exitOnCtrlC: this.props.exitOnCtrlC,
                        internal_eventEmitter: this.internal_eventEmitter
                    }}>
                        <InternalFocusContext.Provider value={{
                            activeId: this.state.activeFocusId,
                            add: (id: string, options?: { autoFocus?: boolean }) => this.addFocusable(id, options || {}),
                            remove: (id) => this.setState(s => ({ focusables: s.focusables.filter(f => f.id !== id) })),
                            activate: (id) => { },
                            deactivate: (id) => { },
                            enableFocus: () => this.setState({ isFocusEnabled: true }),
                            disableFocus: () => this.setState({ isFocusEnabled: false }),
                            focusNext: () => { },
                            focusPrevious: () => { },
                            focus: (id) => this.setState({ activeFocusId: id })
                        }}>
                            {error ? <ErrorDisplay error={error} /> : this.props.children}
                        </InternalFocusContext.Provider>
                    </InternalStdinContext.Provider>
                </ThemeProvider>
            </InternalAppContext.Provider>
        );
    }
}
