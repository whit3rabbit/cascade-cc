import * as React from "react";

const DOUBLE_TAP_TIMEOUT = 800;

/**
 * Hook for detecting timed double-tap actions (e.g. double Return).
 * Deobfuscated from _g in chunk_206.ts.
 */
export function useDoubleTap(
    setActive: (active: boolean) => void,
    onDoubleTap: () => void,
    onSingleTap?: () => void
) {
    const lastTapTime = React.useRef(0);
    const timerId = React.useRef<NodeJS.Timeout | undefined>(undefined);

    const clearTimer = React.useCallback(() => {
        if (timerId.current) {
            clearTimeout(timerId.current);
            timerId.current = undefined;
        }
    }, []);

    React.useEffect(() => {
        return () => clearTimer();
    }, [clearTimer]);

    return React.useCallback(() => {
        const now = Date.now();
        if (now - lastTapTime.current <= DOUBLE_TAP_TIMEOUT && timerId.current !== undefined) {
            clearTimer();
            setActive(false);
            onDoubleTap();
        } else {
            onSingleTap?.();
            setActive(true);
            clearTimer();
            timerId.current = setTimeout(() => {
                setActive(false);
                timerId.current = undefined;
            }, DOUBLE_TAP_TIMEOUT);
        }
        lastTapTime.current = now;
    }, [setActive, onDoubleTap, onSingleTap, clearTimer]);
}
