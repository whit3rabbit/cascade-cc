import { useState, useCallback, useEffect, useRef } from "react";
import { useInput } from "ink";

/**
 * Hook to handle Ctrl-C and Ctrl-D for graceful exit or custom actions.
 */
export function useCtrlExit(onExit?: () => Promise<void>) {
    const [state, setState] = useState<{
        pending: boolean;
        keyName: "Ctrl-C" | "Ctrl-D" | null;
    }>({
        pending: false,
        keyName: null
    });
    const lastPressRef = useRef(0);
    const resetTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    const clearPending = useCallback(() => {
        if (resetTimeoutRef.current) {
            clearTimeout(resetTimeoutRef.current);
            resetTimeoutRef.current = null;
        }
    }, []);

    useEffect(() => {
        return () => {
            clearPending();
        };
    }, [clearPending]);

    const handleExit = useCallback(
        async (key: "Ctrl-C" | "Ctrl-D") => {
            const now = Date.now();
            const withinWindow = now - lastPressRef.current <= 800;

            if (withinWindow && resetTimeoutRef.current) {
                clearPending();
                setState({ pending: false, keyName: null });
                if (onExit) {
                    await onExit();
                } else {
                    process.exit(0);
                }
                return;
            }

            setState({ pending: true, keyName: key });
            clearPending();
            resetTimeoutRef.current = setTimeout(() => {
                setState({ pending: false, keyName: null });
                resetTimeoutRef.current = null;
            }, 800);
            lastPressRef.current = now;
        },
        [clearPending, onExit]
    );

    useInput((input, key) => {
        if (key.ctrl && input === "c") {
            handleExit("Ctrl-C");
        }
        if (key.ctrl && input === "d") {
            handleExit("Ctrl-D");
        }
    });

    return state;
}
