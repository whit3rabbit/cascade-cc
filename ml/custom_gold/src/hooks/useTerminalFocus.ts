import { useState, useCallback, useEffect } from "react";

// Assuming en1 and S0A are imported or defined in focusTracker.ts or similar
import { initFocusTracking, onFocusChange } from "../services/terminal/focusTracker.js";

/**
 * Hook to track whether the terminal is focused.
 * Also provides a filter to remove focus escape sequences from input.
 */
export function useTerminalFocus() {
    const [isFocused, setIsFocused] = useState(true);
    const [wasUnfocusedTyped, setWasUnfocusedTyped] = useState(false);

    useEffect(() => {
        if (!process.stdout.isTTY) return;

        // Subscribe to focus changes
        const unsubscribe = onFocusChange((focused) => {
            setIsFocused(focused);
            setWasUnfocusedTyped(false);
        });

        // Ensure focus tracking is initialized
        initFocusTracking();

        return unsubscribe;
    }, []);

    useEffect(() => {
        // Telemetry for typing while not focused
        if (!isFocused && wasUnfocusedTyped) {
            // logTelemetry("tengu_typing_without_terminal_focus", {});
        }
    }, [isFocused, wasUnfocusedTyped]);

    const filterFocusSequences = useCallback((input: string, key: any) => {
        // Remove focus-in/focus-out sequences
        if (input === "\x1B[I" || input === "\x1B[O" || input === "[I" || input === "[O") {
            return "";
        }

        // If input received while not focused, track it
        if ((input || key) && !isFocused) {
            setWasUnfocusedTyped(true);
        }

        return input;
    }, [isFocused]);

    return {
        isFocused: isFocused || wasUnfocusedTyped,
        filterFocusSequences
    };
}
