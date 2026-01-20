import * as React from "react";
import { TerminalEventEmitter } from "../utils/shared/eventEmitter.js";

/**
 * Context for global app control (e.g., exit).
 * Deobfuscated from ltA in chunk_201.ts.
 */
export const InternalAppContext = React.createContext({
    exit: () => { }
});
InternalAppContext.displayName = "InternalAppContext";

/**
 * Context for stdin and raw mode management.
 * Deobfuscated from itA in chunk_201.ts.
 */
export const InternalStdinContext = React.createContext({
    stdin: process.stdin as any,
    internal_eventEmitter: new TerminalEventEmitter(),
    setRawMode: (_mode: boolean) => { },
    isRawModeSupported: false,
    internal_exitOnCtrlC: true
});
InternalStdinContext.displayName = "InternalStdinContext";

/**
 * Context for focusable component management.
 * Deobfuscated from ntA in chunk_201.ts.
 */
export const InternalFocusContext = React.createContext({
    activeId: undefined as string | undefined,
    add: (_id: string, _options?: { autoFocus?: boolean }) => { },
    remove: (_id: string) => { },
    activate: (_id: string) => { },
    deactivate: (_id: string) => { },
    enableFocus: () => { },
    disableFocus: () => { },
    focusNext: () => { },
    focusPrevious: () => { },
    focus: (_id: string) => { }
});
InternalFocusContext.displayName = "InternalFocusContext";

/**
 * Context for terminal size (columns/rows).
 * Deobfuscated from nNA in chunk_203.ts.
 */
export const TerminalSizeContext = React.createContext<{
    columns: number;
    rows: number;
} | null>(null);
TerminalSizeContext.displayName = "TerminalSizeContext";

export function useTerminalSize() {
    const context = React.useContext(TerminalSizeContext);
    if (!context) {
        throw new Error("useTerminalSize must be used within an Ink App component");
    }
    return context;
}
