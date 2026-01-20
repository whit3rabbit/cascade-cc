
import { useCallback, useRef, useEffect } from "react";
import { useNotifications } from "./NotificationService.js";
import { TextCursor } from "../../utils/terminal/TextCursor.js";

export interface TerminalInputOptions {
    value: string;
    onChange: (value: string) => void;
    onSubmit?: (value: string) => void;
    onExit?: () => void;
    onExitMessage?: (message: string, source?: string) => void;
    onHistoryUp?: () => void;
    onHistoryDown?: () => void;
    onHistoryReset?: () => void;
    mask?: string;
    multiline?: boolean;
    columns: number;
    onImagePaste?: (base64: string, mediaType: string) => void;
    cursorOffset?: number;
    onOffsetChange?: (offset: number) => void;
    inputFilter?: (input: string, key: any) => string;
}

/**
 * Standard (Emacs-like) terminal input hook.
 * Derived from chunk_583.ts (lW1)
 */
export function useStandardTerminalInput(options: TerminalInputOptions) {
    const {
        value,
        onChange,
        onSubmit,
        onExit,
        onExitMessage,
        onHistoryUp,
        onHistoryDown,
        onHistoryReset,
        columns,
        onImagePaste,
        cursorOffset = 0,
        onOffsetChange,
        inputFilter
    } = options;

    const { addNotification, removeNotification } = useNotifications();
    const cursor = TextCursor.fromText(value, columns, cursorOffset);

    const setOffset = (newOffset: number) => {
        onOffsetChange?.(newOffset);
    };

    // Double-escape to clear logic
    const escapeTimestamp = useRef<number>(0);
    const handleEscape = () => {
        const now = Date.now();
        if (now - escapeTimestamp.current < 1000) {
            removeNotification("escape-again-to-clear");
            if (value.trim() !== "") {
                // Persistent history logic would go here
            }
            onChange("");
            setOffset(0);
            onHistoryReset?.();
        } else {
            addNotification({
                key: "escape-again-to-clear",
                text: "Esc to clear again",
                priority: "immediate",
                timeoutMs: 1000
            });
        }
        escapeTimestamp.current = now;
    };

    // Double Ctrl-C logic
    const ctrlCTimestamp = useRef<number>(0);
    const handleCtrlC = () => {
        if (value !== "") {
            onChange("");
            setOffset(0);
            onHistoryReset?.();
            removeNotification("confirm-exit");
            return;
        }

        const now = Date.now();
        if (now - ctrlCTimestamp.current < 800) {
            removeNotification("confirm-exit");
            onExit?.();
        } else {
            onExitMessage?.("Ctrl-C", "system");
            addNotification({
                key: "confirm-exit",
                text: "Press Ctrl-C again to exit",
                priority: "immediate",
                timeoutMs: 1000
            });
        }
        ctrlCTimestamp.current = now;
    };

    const handleInput = (input: string, key: any) => {
        // Apply filter if provided (e.g. for focus sequences)
        const processedInput = inputFilter ? inputFilter(input, key) : input;
        if (processedInput === "" && input !== "") return;

        // Ignore terminal focus reporting sequences
        if (processedInput === "\x1b[I" || processedInput === "\x1b[O") return;

        let nextCursor: TextCursor = cursor;

        // Ctrl keys
        if (key.ctrl) {
            // Check for both raw control character and normalized character
            if (processedInput === "\u0003" || processedInput === "c") {
                handleCtrlC();
                return;
            }

            switch (processedInput) {
                case "a": nextCursor = cursor.startOfLine(); break;
                case "b": nextCursor = cursor.left(); break;
                case "d":
                    if (value === "") { onExitMessage?.("Ctrl-D", "system"); onExit?.(); return; }
                    nextCursor = cursor.del();
                    break;
                case "e": nextCursor = cursor.endOfLine(); break;
                case "f": nextCursor = cursor.right(); break;
                case "h": nextCursor = cursor.backspace(); break;
                case "k": {
                    const { cursor: c } = cursor.deleteToLineEnd();
                    nextCursor = c;
                    break;
                }
                case "l":
                    if (value.trim() !== "") onHistoryReset?.();
                    nextCursor = TextCursor.fromText("", columns, 0);
                    break;
                case "u": {
                    const { cursor: c } = cursor.deleteToLineStart();
                    nextCursor = c;
                    break;
                }
                case "w": {
                    const { cursor: c } = cursor.deleteWordBefore();
                    nextCursor = c;
                    break;
                }
                case "y": {
                    // Yank/Paste logic (simplified)
                    break;
                }
                case "p": nextCursor = cursor.up(); if (nextCursor.equals(cursor)) onHistoryUp?.(); break;
                case "n": nextCursor = cursor.down(); if (nextCursor.equals(cursor)) onHistoryDown?.(); break;
            }
        }
        // Arrows and navigation
        else if (key.leftArrow) nextCursor = (key.meta || key.ctrl) ? cursor.prevWord() : cursor.left();
        else if (key.rightArrow) nextCursor = (key.meta || key.ctrl) ? cursor.nextWord() : cursor.right();
        else if (key.upArrow) {
            nextCursor = cursor.up();
            if (nextCursor.equals(cursor)) onHistoryUp?.();
        }
        else if (key.downArrow) {
            nextCursor = cursor.down();
            if (nextCursor.equals(cursor)) onHistoryDown?.();
        }
        else if (key.backspace || processedInput === "\x7f" || processedInput === "\b") {
            nextCursor = key.meta ? cursor.deleteWordBefore().cursor : cursor.backspace();
        }
        else if (key.delete) nextCursor = key.meta ? cursor.deleteToLineEnd().cursor : cursor.del();
        else if (key.home) nextCursor = cursor.startOfLine();
        else if (key.end || key.pageDown) nextCursor = cursor.endOfLine();
        else if (key.pageUp) nextCursor = cursor.startOfLine();
        else if (key.escape) { handleEscape(); return; }
        else if (key.return) {
            if (options.multiline && value.endsWith("\\")) {
                nextCursor = cursor.backspace().insert("\n");
            } else if (key.meta) {
                nextCursor = cursor.insert("\n");
            } else {
                onSubmit?.(value);
                return;
            }
        }
        // Regular typing - only if not other control keys and not empty
        else if (processedInput && !key.ctrl && !key.meta) {
            nextCursor = cursor.insert(processedInput.replace(/\r/g, "\n"));
        }

        if (!nextCursor.equals(cursor)) {
            if (nextCursor.text !== cursor.text) onChange(nextCursor.text);
            setOffset(nextCursor.offset);
        }
    };

    return {
        onInput: handleInput,
        renderedValue: cursor.render({
            maskChar: options.mask,
            cursorChar: " "
        }),
        offset: cursorOffset,
        setOffset
    };
}
