import { useState, useCallback, useRef } from 'react';

export interface MacroStep {
    char: string;
    key: any;
}

export interface UseMacrosProps {
    onInput: (char: string, key: any, bypassRecording: boolean) => void;
}

/**
 * Hook to manage macro recording and asynchronous playback.
 * Addresses the state batching issues in UserPromptMessage.tsx.
 */
export function useMacros({ onInput }: UseMacrosProps) {
    const [recordedMacros, setRecordedMacros] = useState<Record<string, MacroStep[]>>({});
    const [isRecording, setIsRecording] = useState(false);
    const [recordingRegister, setRecordingRegister] = useState<string | null>(null);
    const macroBuffer = useRef<MacroStep[]>([]);
    const isPlaying = useRef(false);

    const startRecording = useCallback((register: string) => {
        setRecordingRegister(register);
        setIsRecording(true);
        macroBuffer.current = [];
    }, []);

    const stopRecording = useCallback(() => {
        if (recordingRegister) {
            setRecordedMacros(prev => ({
                ...prev,
                [recordingRegister]: [...macroBuffer.current]
            }));
        }
        setIsRecording(false);
        setRecordingRegister(null);
    }, [recordingRegister]);

    const recordStep = useCallback((char: string, key: any) => {
        if (isRecording && !isPlaying.current) {
            macroBuffer.current.push({ char, key });
        }
    }, [isRecording]);

    const playMacro = useCallback(async (register: string) => {
        const macro = recordedMacros[register];
        if (!macro || isPlaying.current) return;

        isPlaying.current = true;
        for (const step of macro) {
            onInput(step.char, step.key, true);
            // Small delay to allow state updates to settle
            await new Promise(resolve => setTimeout(resolve, 0));
        }
        isPlaying.current = false;
    }, [recordedMacros, onInput]);

    return {
        isRecording,
        recordingRegister,
        startRecording,
        stopRecording,
        recordStep,
        playMacro,
        isPlaying: isPlaying.current
    };
}
