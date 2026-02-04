import { useState, useCallback } from 'react';

export interface UseStashProps {
    value: string;
    onChange: (value: string) => void;
    onNotify?: (message: string) => void;
}

export function useStash({ value, onChange, onNotify }: UseStashProps) {
    const [stashedValue, setStashedValue] = useState<string | null>(null);

    const handleStash = useCallback(() => {
        // If there is input, stash it
        if (value.trim()) {
            setStashedValue(value);
            onChange('');
            onNotify?.('Prompt stashed (Ctrl+S to restore)');
        }
        // If input is empty but we have a stash, restore it
        else if (stashedValue) {
            onChange(stashedValue);
            setStashedValue(null);
            onNotify?.('Prompt restored');
        }
    }, [value, onChange, stashedValue, onNotify]);

    return {
        stashedValue,
        handleStash
    };
}
