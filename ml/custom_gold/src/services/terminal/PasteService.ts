
import { useState, useRef, useEffect, useCallback } from "react";

/**
 * Paste handler for terminal, supports bracketed paste mode.
 * Derived from chunk_499.ts (Dl2)
 */
export function usePasteHandler(options: {
    onPaste: (text: string) => void;
    onImagePaste?: (base64: string, mediaType: string, dimensions?: any) => void;
}) {
    const { onPaste, onImagePaste } = options;
    const [isPasting, setIsPasting] = useState(false);
    const pasteBuffer = useRef<string[]>([]);
    const timeoutId = useRef<NodeJS.Timeout | null>(null);

    // This would typically hook into process.stdin directly if needed,
    // but in Ink, it often comes through as rapid 'data' events.

    const processPasteBuffer = useCallback(() => {
        const fullText = pasteBuffer.current.join("");
        // Detect if it's an image path (Claude has special handling for macOS screenshots)
        if (onImagePaste && (fullText.startsWith("/") || fullText.includes("Screenshot"))) {
            // Simplified: check if path exists and use image processor
            // ...
        }

        onPaste(fullText);
        pasteBuffer.current = [];
        setIsPasting(false);
    }, [onPaste, onImagePaste]);

    const handleInput = useCallback((input: string, key: any) => {
        // Detecting bracketed paste mode: \x1B[200~ ... \x1B[201~
        if (input.includes("\x1B[200~")) {
            setIsPasting(true);
            const content = input.replace("\x1B[200~", "");
            if (content) pasteBuffer.current.push(content);
            return true;
        }

        if (isPasting) {
            if (input.includes("\x1B[201~")) {
                const content = input.replace("\x1B[201~", "");
                if (content) pasteBuffer.current.push(content);
                processPasteBuffer();
            } else {
                pasteBuffer.current.push(input);
            }
            return true;
        }

        // HEURISTIC: if input is very long (like a normal paste without bracketed mode)
        if (input.length > 20) {
            onPaste(input);
            return true;
        }

        return false;
    }, [isPasting, processPasteBuffer, onPaste]);

    return {
        handleInput,
        isPasting
    };
}
