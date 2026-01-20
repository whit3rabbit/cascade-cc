import { useState, useRef, useCallback, useEffect } from "react";
import chalk from "chalk";
import { TextCursor } from "../utils/terminal/TextCursor.js";
import { useStandardTerminalInput, TerminalInputOptions } from "../services/terminal/StandardInputService.js";
import { useNotifications } from "../services/terminal/NotificationService.js";

// Constants from chunk_499.ts
const MAX_VIM_COUNT = 10000;
const PASTE_SCAN_TIMEOUT = 50;
const PASTE_FLUSH_TIMEOUT = 100;

/**
 * Vim-mode hook based on chunk_499.ts (Vl2)
 */
export function useVimInput(options: TerminalInputOptions & {
    onModeChange?: (mode: "NORMAL" | "INSERT") => void,
    onUndo?: () => void,
    onModeToggle?: (mode: "NORMAL" | "INSERT") => void
}) {
    const [mode, setModeState] = useState<"NORMAL" | "INSERT">("INSERT");
    const prefixRef = useRef<string>("");     // G in chunk
    const lastActionRef = useRef<any>(null); // Z in chunk
    const insertedTextRef = useRef<string>(""); // Y in chunk
    const countStringRef = useRef<string>(""); // J in chunk
    const pendingOpRef = useRef<string | null>(null); // X in chunk

    const standardInput = useStandardTerminalInput({
        ...options,
        inputFilter: options.inputFilter
    });

    const setMode = (m: "NORMAL" | "INSERT") => {
        setModeState(m);
        options.onModeChange?.(m);
        options.onModeToggle?.(m);
    };

    const getCount = (reset = true) => {
        const val = parseInt(countStringRef.current, 10);
        if (isNaN(val)) {
            if (reset) countStringRef.current = "";
            return 1;
        }
        const count = Math.min(val, MAX_VIM_COUNT);
        if (reset) countStringRef.current = "";
        return count;
    };

    // V in chunk
    const applyMotion = (key: string, cursor: TextCursor, count = 1): TextCursor | null => {
        // Simplified motion mapping matching K in chunk
        const getBaseMotion = (k: string, c: TextCursor) => {
            switch (k) {
                case "h": return c.left();
                case "l": return c.right();
                case "j": return c.downLogicalLine();
                case "k": return c.upLogicalLine();
                case "0": return c.startOfLogicalLine();
                case "^": return c.firstNonBlankInLogicalLine();
                case "$": return c.endOfLogicalLine();
                case "w": return c.nextWord();
                case "e": return c.endOfWord();
                case "b": return c.prevWord();
                case "W": return c.nextWORD();
                case "E": return c.endOfWORD();
                case "B": return c.prevWORD();
                case "gg": return c.startOfFirstLine();
                case "G": return c.startOfLastLine();
                default: return null;
            }
        };

        if ((key === "d" || key === "c") && prefixRef.current === key) {
            return cursor.startOfLine();
        }

        let current = cursor;
        for (let i = 0; i < count; i++) {
            const next = getBaseMotion(key, current);
            if (!next) return i === 0 ? null : current;
            current = next;
        }
        return current;
    };

    // H in chunk
    const applyOperation = (opType: "move" | "delete" | "change", motionKey: string, cursor: TextCursor, count = 1) => {
        const isDouble = (motionKey === "d" || motionKey === "c") && prefixRef.current === motionKey;
        const currentOffset = standardInput.offset;
        const isChange = opType === "change";

        if (isDouble) {
            const logicalStart = cursor.startOfLogicalLine();
            if (!options.value.includes("\n")) {
                if (opType !== "move") options.onChange("");
                return { newOffset: 0, switchToInsert: isChange };
            } else {
                const lines = options.value.split("\n");
                const pos = cursor.getPosition();
                const lineIdx = pos.line;
                const linesToDelete = Math.min(count, lines.length - lineIdx);

                if (opType === "delete") {
                    lines.splice(lineIdx, linesToDelete);
                    const newText = lines.join("\n");
                    options.onChange(newText);
                    // Determine new offset
                    const newCursor = TextCursor.fromText(newText, options.columns, logicalStart.offset);
                    return { newOffset: newCursor.offset, switchToInsert: false };
                } else if (isChange) {
                    for (let i = 0; i < linesToDelete; i++) lines[lineIdx + i] = "";
                    options.onChange(lines.join("\n"));
                    return { newOffset: logicalStart.offset, switchToInsert: true };
                }
            }
            return { newOffset: logicalStart.offset, switchToInsert: isChange };
        }

        const target = applyMotion(motionKey, cursor, count);
        if (!target || cursor.equals(target)) {
            return { newOffset: currentOffset, switchToInsert: isChange };
        }

        if (opType === "move") {
            return { newOffset: target.offset, switchToInsert: false };
        } else {
            let from = cursor.offset <= target.offset ? cursor : target;
            let to = cursor.offset <= target.offset ? target : cursor;

            if (motionKey === "e" && cursor.offset <= target.offset) {
                to = to.right();
            } else if ((motionKey === "w" || motionKey === "W") && isChange) {
                // CW logic from L in chunk (simplified)
            }

            const modification = from.modifyText(to, "");
            options.onChange(modification.text);
            return {
                newOffset: isChange ? from.offset : modification.offset,
                switchToInsert: isChange
            };
        }
    };

    // M in chunk
    const repeatLastAction = (cursor: TextCursor) => {
        const action = lastActionRef.current;
        if (!action) return;

        switch (action.type) {
            case "delete":
            case "change":
                if (action.motion) {
                    const { newOffset, switchToInsert } = applyOperation(action.type, action.motion, cursor, action.count || 1);
                    standardInput.setOffset(newOffset);
                    if (switchToInsert) setMode("INSERT");
                }
                break;
            case "insert":
                if (action.text) {
                    const next = cursor.insert(action.text);
                    options.onChange(next.text);
                    standardInput.setOffset(next.offset);
                }
                break;
            case "x": {
                let next = cursor;
                for (let i = 0; i < (action.count || 1); i++) next = next.del();
                options.onChange(next.text);
                standardInput.setOffset(next.offset);
                break;
            }
            case "o":
            case "O": {
                // handle line opening repetition
                break;
            }
        }
    };

    const onInput = (input: string, key: any) => {
        const cursor = TextCursor.fromText(options.value, options.columns, standardInput.offset);

        if (key.ctrl) {
            standardInput.onInput(input, key);
            return;
        }

        if (key.escape && mode === "INSERT") {
            if (insertedTextRef.current) {
                lastActionRef.current = { type: "insert", text: insertedTextRef.current };
                insertedTextRef.current = "";
            }
            setMode("NORMAL");
            return;
        }

        if (mode === "NORMAL" && pendingOpRef.current) {
            const op = pendingOpRef.current;
            // Handle dd/cc
            if ((op === "delete" && input === "d") || (op === "change" && input === "c")) {
                const count = getCount();
                const { newOffset, switchToInsert } = applyOperation(op as any, input, cursor, count);
                standardInput.setOffset(newOffset);
                lastActionRef.current = { type: op, motion: input, count };
                pendingOpRef.current = null;
                prefixRef.current = "";
                if (switchToInsert) setMode("INSERT");
                return;
            }

            // Handle f/t/F/T search motions
            if (prefixRef.current && "fFtT".includes(prefixRef.current)) {
                // delegate search
            }

            if ("0123456789".includes(input)) {
                countStringRef.current += input;
                return;
            }

            const count = getCount();
            const { newOffset, switchToInsert } = applyOperation(op as any, input, cursor, count);
            standardInput.setOffset(newOffset);
            lastActionRef.current = { type: op, motion: input, count };
            pendingOpRef.current = null;
            prefixRef.current = "";
            if (switchToInsert) setMode("INSERT");
            return;
        }

        if (mode === "NORMAL") {
            if ("0123456789".includes(input)) {
                if (input === "0" && countStringRef.current === "") {
                    const { newOffset } = applyOperation("move", "0", cursor);
                    standardInput.setOffset(newOffset);
                    return;
                }
                countStringRef.current += input;
                return;
            }

            switch (input) {
                case ".": repeatLastAction(cursor); break;
                case "u": options.onUndo?.(); break;
                case "i": countStringRef.current = ""; insertedTextRef.current = ""; setMode("INSERT"); break;
                case "I": countStringRef.current = ""; setMode("INSERT"); standardInput.setOffset(cursor.startOfLogicalLine().offset); break;
                case "a": countStringRef.current = ""; setMode("INSERT"); standardInput.setOffset(cursor.right().offset); break;
                case "A": countStringRef.current = ""; setMode("INSERT"); standardInput.setOffset(cursor.endOfLogicalLine().offset); break;
                case "o": {
                    const next = cursor.endOfLogicalLine().insert("\n");
                    options.onChange(next.text);
                    lastActionRef.current = { type: "o" };
                    setMode("INSERT");
                    standardInput.setOffset(next.offset);
                    break;
                }
                case "O": {
                    const next = cursor.startOfLogicalLine().insert("\n");
                    options.onChange(next.text);
                    lastActionRef.current = { type: "O" };
                    setMode("INSERT");
                    standardInput.setOffset(next.startOfLogicalLine().offset);
                    break;
                }
                case "h": case "l": case "j": case "k": case "^": case "$": case "w": case "e": case "b": case "W": case "E": case "B": case "G": {
                    const count = getCount();
                    const { newOffset } = applyOperation("move", input, cursor, count);
                    standardInput.setOffset(newOffset);
                    break;
                }
                case "g": prefixRef.current = "g"; break;
                case "f": case "F": case "t": case "T": prefixRef.current = input; break;
                case "x": {
                    const count = getCount();
                    let next = cursor;
                    for (let i = 0; i < count; i++) if (!next.isAtEnd()) next = next.del();
                    options.onChange(next.text);
                    standardInput.setOffset(next.offset);
                    lastActionRef.current = { type: "x", count };
                    break;
                }
                case "d": prefixRef.current = "d"; pendingOpRef.current = "delete"; break;
                case "D": {
                    const count = getCount();
                    const { newOffset } = applyOperation("delete", "$", cursor, count);
                    standardInput.setOffset(newOffset);
                    break;
                }
                case "c": prefixRef.current = "c"; pendingOpRef.current = "change"; break;
                case "C": {
                    const count = getCount();
                    const { newOffset } = applyOperation("change", "$", cursor, count);
                    standardInput.setOffset(newOffset);
                    setMode("INSERT");
                    break;
                }
            }
            return;
        }

        if (mode === "INSERT") {
            if (input && !key.ctrl && !key.meta) insertedTextRef.current += input;
            else if (key.backspace || key.delete) insertedTextRef.current = insertedTextRef.current.slice(0, -1);
            standardInput.onInput(input, key);
        }
    };

    return {
        ...standardInput,
        onInput,
        mode,
        setMode
    };
}

/**
 * Placeholder rendering hook based on chunk_499.ts (El2)
 */
export function usePlaceholder({
    placeholder,
    value,
    showCursor,
    focus,
    terminalFocus = true
}: {
    placeholder?: string,
    value: string,
    showCursor?: boolean,
    focus?: boolean,
    terminalFocus?: boolean
}) {
    let renderedPlaceholder: string | undefined;
    if (placeholder) {
        renderedPlaceholder = chalk.dim(placeholder);
        if (showCursor && focus && terminalFocus) {
            renderedPlaceholder = placeholder.length > 0
                ? chalk.inverse(placeholder[0]) + chalk.dim(placeholder.slice(1))
                : chalk.inverse(" ");
        }
    }
    const showPlaceholder = value.length === 0 && Boolean(placeholder);
    return { renderedPlaceholder, showPlaceholder };
}

/**
 * Paste handling hook based on chunk_499.ts (Dl2)
 */
export function useTerminalPaste({ onPaste, onImagePaste }: {
    onPaste?: (text: string) => void,
    onImagePaste?: (base64: string, mediaType: string, dimensions?: any) => void
}) {
    const [isPasting, setIsPasting] = useState(false);
    const [pasteState, setPasteState] = useState<{ chunks: string[], timeoutId: NodeJS.Timeout | null }>({ chunks: [], timeoutId: null });
    const isPastingRef = useRef(false);

    useEffect(() => {
        const stdin = process.stdin;
        if (!stdin) return;

        const handleData = (data: Buffer) => {
            const str = data.toString();
            if (str.includes("\x1b[200~")) {
                setIsPasting(true);
                isPastingRef.current = true;
            }
            if (str.includes("\x1b[201~")) {
                setIsPasting(false);
                isPastingRef.current = false;
            }
        };

        stdin.on("data", handleData);
        return () => {
            stdin.off("data", handleData);
        };
    }, []);

    const wrappedOnInput = useCallback((input: string, key: any, onInput: (input: string, key: any) => void) => {
        if (isPastingRef.current || input.length > 10) {
            setPasteState(prev => {
                if (prev.timeoutId) clearTimeout(prev.timeoutId);
                const nextChunks = [...prev.chunks, input];
                const nextTimeout = setTimeout(() => {
                    const fullText = nextChunks.join("").replace(/\x1b\[200~/g, "").replace(/\x1b\[201~/g, "");
                    onPaste?.(fullText);
                    setPasteState({ chunks: [], timeoutId: null });
                }, PASTE_FLUSH_TIMEOUT);
                return { chunks: nextChunks, timeoutId: nextTimeout };
            });
        } else {
            onInput(input, key);
        }
    }, [onPaste]);

    return { isPasting, wrappedOnInput, pasteState };
}
