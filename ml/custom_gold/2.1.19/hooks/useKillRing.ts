import { useState, useCallback, useRef } from 'react';

/**
 * Hook to manage an Emacs-style Kill Ring for terminal input.
 */
export function useKillRing() {
    const [killRing, setKillRing] = useState<string[]>([]);
    const [killIndex, setKillIndex] = useState(0);
    const lastActionWasKill = useRef(false);

    const push = useCallback((text: string, mode: 'prepend' | 'append' = 'append') => {
        if (!text) return;

        if (lastActionWasKill.current && killRing.length > 0) {
            setKillRing(prev => {
                const newRing = [...prev];
                if (mode === 'prepend') {
                    newRing[0] = text + newRing[0];
                } else {
                    newRing[0] = newRing[0] + text;
                }
                return newRing;
            });
        } else {
            setKillRing(prev => [text, ...prev].slice(0, 60)); // Limit to 60 items as in gold
            setKillIndex(0);
        }
        lastActionWasKill.current = true;
    }, [killRing]);

    const resetKillAction = useCallback(() => {
        lastActionWasKill.current = false;
    }, []);

    const yank = useCallback(() => {
        if (killRing.length === 0) return null;
        return killRing[killIndex % killRing.length];
    }, [killRing, killIndex]);

    const rotate = useCallback(() => {
        if (killRing.length <= 1) return null;
        const newIndex = (killIndex + 1) % killRing.length;
        setKillIndex(newIndex);
        return killRing[newIndex];
    }, [killRing, killIndex]);

    return {
        push,
        yank,
        rotate,
        resetKillAction,
        isLastActionKill: lastActionWasKill.current,
        killRingSize: killRing.length
    };
}
