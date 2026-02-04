import { useCallback, useEffect, useRef } from 'react';

/**
 * Hook to handle double-action logic (double-press within a timeout).
 * Renamed from 'Xy' in the original source.
 */
export function useDoubleAction(
    setPending: (pending: boolean) => void,
    onConfirm: () => void,
    onFirstPress?: () => void,
    timeoutMs: number = 800
) {
    const lastPressTime = useRef(0);
    const timeoutId = useRef<NodeJS.Timeout | undefined>(undefined);

    const clearActiveTimeout = useCallback(() => {
        if (timeoutId.current) {
            clearTimeout(timeoutId.current);
            timeoutId.current = undefined;
        }
    }, []);

    useEffect(() => {
        return () => {
            clearActiveTimeout();
        };
    }, [clearActiveTimeout]);

    return useCallback(() => {
        const now = Date.now();
        if (now - lastPressTime.current <= timeoutMs && timeoutId.current !== undefined) {
            clearActiveTimeout();
            setPending(false);
            onConfirm();
        } else {
            onFirstPress?.();
            setPending(true);
            clearActiveTimeout();
            timeoutId.current = setTimeout(() => {
                setPending(false);
                timeoutId.current = undefined;
            }, timeoutMs);
        }
        lastPressTime.current = now;
    }, [setPending, onConfirm, onFirstPress, clearActiveTimeout, timeoutMs]);
}
