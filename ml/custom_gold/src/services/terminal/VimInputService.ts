
import { useState, useRef } from "react";
import { useStandardTerminalInput, TerminalInputOptions } from "./StandardInputService.js";
import { TextCursor } from "../../utils/terminal/TextCursor.js";

export type VimMode = "NORMAL" | "INSERT";

/**
 * Vim-mode terminal input hook.
 * Derived from chunk_499.ts (Vl2)
 */
export function useVimTerminalInput(options: TerminalInputOptions) {
    const [mode, setMode] = useState<VimMode>("INSERT");
    const opRef = useRef<string>(""); // For commands like 'd', 'c', 'f'
    const opCountRef = useRef<string>(""); // For counts like '3'
    const motionRef = useRef<string | null>(null);

    const standardInput = useStandardTerminalInput({
        ...options,
        // Override onInput logic for Vim state machine
    });

    const switchToInsert = (offset?: number) => {
        if (offset !== undefined) standardInput.setOffset(offset);
        setMode("INSERT");
    };

    const switchToNormal = () => {
        setMode("NORMAL");
    };

    const getCount = () => {
        const count = parseInt(opCountRef.current, 10);
        opCountRef.current = "";
        return isNaN(count) ? 1 : count;
    };

    const handleVimInput = (input: string, key: any) => {
        if (key.ctrl) {
            standardInput.onInput(input, key);
            return;
        }

        if (key.escape && mode === "INSERT") {
            switchToNormal();
            return;
        }

        if (mode === "INSERT") {
            standardInput.onInput(input, key);
            return;
        }

        // NORMAL mode logic
        const cursor = TextCursor.fromText(options.value, options.columns, standardInput.offset);

        // Movement keys
        if ("0123456789".includes(input)) {
            if (input === "0" && opCountRef.current === "") {
                standardInput.setOffset(cursor.startOfLine().offset);
                return;
            }
            opCountRef.current += input;
            return;
        }

        switch (input) {
            case "i": switchToInsert(); break;
            case "I": switchToInsert(cursor.startOfLine().offset); break;
            case "a": switchToInsert(cursor.right().offset); break;
            case "A": switchToInsert(cursor.endOfLine().offset); break;
            case "h": standardInput.setOffset(cursor.left().offset); break;
            case "l": standardInput.setOffset(cursor.right().offset); break;
            case "j": standardInput.setOffset(cursor.down().offset); break;
            case "k": standardInput.setOffset(cursor.up().offset); break;
            case "w": standardInput.setOffset(cursor.nextWord().offset); break;
            case "b": standardInput.setOffset(cursor.prevWord().offset); break;
            case "e": standardInput.setOffset(cursor.endOfWord().offset); break;
            case "G": standardInput.setOffset(cursor.startOfLastLine().offset); break;
            case "x": {
                const count = getCount();
                let next = cursor;
                for (let i = 0; i < count; i++) next = next.del();
                options.onChange(next.text);
                standardInput.setOffset(next.offset);
                break;
            }
            case "d": {
                if (opRef.current === "d") {
                    // dd -> delete line
                    // Simplified: delete current logical line
                    const next = cursor.deleteToLogicalLineEnd();
                    options.onChange(next.text);
                    standardInput.setOffset(next.offset);
                    opRef.current = "";
                } else {
                    opRef.current = "d";
                }
                break;
            }
            case "c": {
                if (opRef.current === "c") {
                    const next = cursor.deleteToLogicalLineEnd();
                    options.onChange(next.text);
                    switchToInsert(next.offset);
                    opRef.current = "";
                } else {
                    opRef.current = "c";
                }
                break;
            }
            case "u":
                // Undo logic (would call options.onUndo if available)
                break;
            case ".":
                // Repeat last operation logic
                break;
        }
    };

    return {
        ...standardInput,
        onInput: handleVimInput,
        mode,
        setMode
    };
}
