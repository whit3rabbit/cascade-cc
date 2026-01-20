import { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import { useNotifications } from '../../services/terminal/NotificationService.js';
import { useInput, Text } from 'ink';
import { useAppState } from '../../contexts/AppStateContext.js';
import { logEvent } from '../../services/telemetry/TelemetryService.js';
import * as path from 'path';
import { readHistoryStream } from '../../services/terminal/promptHistory.js';

// --- Constants & Types ---

export const STASH_HINT_TIMEOUT_MS = 10000;
export const MAX_PASTE_SIZE = 5000;
export const INLINE_SUGGESTION_HINT = "Tab to complete";

export const thinkingKeywordColors: Record<string, string> = {
    low: "blue",
    medium: "magenta",
    high: "red"
};

export const thinkingKeywordShimmerColors: Record<string, string> = {
    low: "cyan",
    medium: "pink",
    high: "yellow"
};

export const OPTION_META_HINTS: Record<string, string> = {
    "meta+f": "Find",
    "meta+b": "Back"
};

export const KNOWN_BORDER_COLORS = ["red", "green", "blue", "yellow", "magenta", "cyan", "white"];

export const borderColorLookup: Record<string, string> = {
    red: "red",
    green: "green",
    blue: "blue",
    yellow: "yellow",
    magenta: "magenta",
    cyan: "cyan",
    white: "white"
};

// --- Helper Functions ---

export function getShellSnapshot(): any {
    return process.env;
}

export function getNextAttachmentId(messages: any[]): number {
    let maxId = 0;
    for (const msg of messages) {
        if (msg.type === "user") {
            // Only check if message format allows attachments with IDs
            if (msg.imagePasteIds) {
                for (const id of msg.imagePasteIds) {
                    if (id > maxId) maxId = id;
                }
            }
            if (Array.isArray(msg.message?.content)) {
                for (const content of msg.message.content) {
                    if (content.type === "text") {
                        // Assuming parseAttachmentRefs logic here if needed, but keeping it simple based on r17
                        // const refs = parseAttachmentRefs(content.text);
                        // for (const ref of refs) if (ref.id > maxId) maxId = ref.id;
                    }
                }
            }
        }
    }
    return maxId + 1;
}

export function normalizeHistorySearchEntry(value: string): string {
    return value.trim();
}

export function findThinkingKeywords(value: string): any[] {
    const matches = [];
    // Assuming simple regex for now as LX1 logic was not fully visible but likely regex based
    const regex = /thinking/gi;
    let match;
    while ((match = regex.exec(value)) !== null) {
        matches.push({ start: match.index, end: match.index + match[0].length, word: match[0] });
    }
    return matches;
}

export function findWarningKeywords(value: string): any[] {
    const matches = [];
    const regex = /warning|error|danger/gi;
    let match;
    while ((match = regex.exec(value)) !== null) {
        matches.push({ start: match.index, end: match.index + match[0].length, word: match[0] });
    }
    return matches;
}

export function getThinkingKeywordLevel(value: string): { level: "none" | "low" | "medium" | "high" } {
    if (value.length > 50) return { level: "high" };
    if (value.length > 20) return { level: "medium" };
    return { level: "low" };
}

export function isRainbowKeyword(value: string): boolean {
    return value.toLowerCase() === "rainbow";
}

export function getSessionHints() {
    return { hasUsedStash: false, hasSeenTasksHint: false };
}

export function updateSessionHints(updater: (value: any) => any): void {
    // This would typically update a persistent store or context
    // Placeholder implementation
}

export function resetInputHints(): void {
    // Placeholder implementation
}

export function detectInputModeFromPrefix(value: string): string {
    if (value.startsWith("/")) return "command";
    if (value.startsWith("!")) return "bash";
    return "prompt";
}

export function isQueuedCommandMode(mode: string): boolean {
    return mode === "queued";
}

export function trackEvent(name: string, payload: Record<string, any>): void {
    logEvent(name, payload);
}

export function trackSuggestionTiming(type: string, text: string | null): void {
    // Implementation driven by context
}

export function registerAttachment(attachment: any): void {
    // console.log("Registered attachment", attachment);
}

export function logAttachmentPaste(attachment: any): void {
    // console.log("Logged attachment paste", attachment);
}

export function normalizePastedText(value: string): string {
    return value.replace(/\r\n/g, "\n");
}

export function countLines(value: string): number {
    return value.split("\n").length;
}

export function formatPasteAsAttachment(id: number, lineCount: number): string {
    return `[Attachment #${id} (${lineCount} lines)]`;
}

export function createMessageSelectorHandler(onShow: () => void, onToggle: () => void): () => void {
    return onToggle;
}

export const pathUtils = {
    relative: (from: string, to: string) => {
        return path.relative(from, to);
    }
};

export function getWorkspaceRoot(): string {
    return process.cwd();
}

export function getPlatformId(): string {
    return process.platform === "darwin" ? "macos" : process.platform;
}

export function trackShortcut(name: string): void {
    // console.log(`[Shortcut] ${name}`);
}

export function openExternalEditor(input: string): string | null {
    return input;
}

export function getNextPromptMode(permissionContext: any, teamContext: any): string {
    return "prompt";
}

export function setPlanModeExit(value: boolean): void {
}

export function shouldPromptModeSwitch(current: string, next: string): boolean {
    return current !== next;
}

export function setAutoAcceptMode(value: boolean): void {
}

export function setEnableEditsToast(value: boolean): void {
}

export function applyModeChange(permissionContext: any, action: any): any {
    return { ...permissionContext, mode: action.mode };
}

export function dimText(value: string): string {
    return value;
}

export function isVimInputEnabled(): boolean {
    return false;
}

export function getBorderColorOverride(): string | null {
    return null;
}

export function getActiveBanner(): { text: string; bgColor: string } | null {
    return null;
}

export function parseAttachmentRefs(text: string): { id: number }[] {
    return [];
}

// Logic based on YG1 from chunk_387.ts
export async function popQueuedInput(
    input: string,
    cursorOffset: number,
    getState: () => Promise<any>,
    setState: (value: any) => void
): Promise<{ text: string; cursorOffset: number } | null> {
    const state = await getState();
    if (state.queuedCommands.length === 0) return null;

    // Filter editable vs non-editable commands logic (simplified)
    const editableCommands = state.queuedCommands.filter((c: any) => c.mode !== "agent-notification" && c.mode !== "bash-notification");
    const nonEditableCommands = state.queuedCommands.filter((c: any) => c.mode === "agent-notification" || c.mode === "bash-notification");

    if (editableCommands.length === 0) return null;

    const values = editableCommands.map((c: any) => c.value);
    const text = [...values, input].filter(Boolean).join("\n");
    const newCursorOffset = values.join("\n").length + 1 + cursorOffset;

    setState((prev: any) => ({
        ...prev,
        queuedCommands: nonEditableCommands
    }));

    return { text, cursorOffset: newCursorOffset };
}

// --- Hooks ---

// Logic based on Wt2 from chunk_512.ts
export function useHistorySearch(
    onSubmit: (entry: any) => void,
    input: string,
    onInputChange: (value: string) => void,
    setCursorOffset: (value: number) => void,
    cursorOffset: number,
    onModeChange: (mode: string) => void,
    mode: string,
    isSearching: boolean,
    setIsSearching: (value: boolean) => void
) {
    const [historyQuery, setHistoryQuery] = useState("");
    const [historyFailedMatch, setHistoryFailedMatch] = useState(false);
    const [savedInput, setSavedInput] = useState("");
    const [savedCursor, setSavedCursor] = useState(0);
    const [savedMode, setSavedMode] = useState("prompt");
    const [historyMatch, setHistoryMatch] = useState<any>(undefined);

    const historyIterator = useRef<AsyncGenerator<any> | undefined>(undefined);
    const visitedHistory = useRef(new Set<string>());
    const abortController = useRef<AbortController | null>(null);

    const cancelSearch = useCallback(() => {
        setIsSearching(false);
        setHistoryQuery("");
        setHistoryFailedMatch(false);
        setSavedInput("");
        setSavedCursor(0);
        setSavedMode("prompt");
        setHistoryMatch(undefined);

        if (historyIterator.current) {
            historyIterator.current.return(undefined);
            historyIterator.current = undefined;
        }
        visitedHistory.current.clear();
    }, [setIsSearching]);

    const performSearch = useCallback(async (isNextArgs: boolean, signal?: AbortSignal) => {
        if (!isSearching) return;

        if (historyQuery.length === 0) {
            if (historyIterator.current) {
                historyIterator.current.return(undefined);
                historyIterator.current = undefined;
            }
            visitedHistory.current.clear();
            setHistoryMatch(undefined);
            setHistoryFailedMatch(false);
            onInputChange(savedInput);
            setCursorOffset(savedCursor);
            onModeChange(savedMode);
            return;
        }

        if (!isNextArgs) {
            if (historyIterator.current) {
                historyIterator.current.return(undefined);
            }
            historyIterator.current = readHistoryStream();
            visitedHistory.current.clear();
        }

        if (!historyIterator.current) return;

        while (true) {
            if (signal?.aborted) return;

            const next = await historyIterator.current.next();
            if (next.done) {
                setHistoryFailedMatch(true);
                return;
            }

            const item = next.value;
            const display = item.display;
            const matchIndex = display.lastIndexOf(historyQuery);

            if (matchIndex !== -1 && !visitedHistory.current.has(display)) {
                visitedHistory.current.add(display);
                setHistoryMatch(item);
                setHistoryFailedMatch(false);

                const rawMode = detectInputModeFromPrefix(display);
                onModeChange(rawMode);
                const text = (rawMode === "bash" && display.startsWith("!")) ? display.slice(1) : display;
                onInputChange(text);

                // Position cursor at end of match
                // Chunk_512 logic: G(TA !== -1 ? TA : d); where TA is lastIndexOf, d is also lastIndexOf logic?
                // The original logic seems to try to position cursor near the match query.
                // Assuming simple end of text for now, or match index + query length?
                // Replicating: let TA = K0A(v).lastIndexOf(W);
                // K0A is likely text stripping prefix. 
                const strippedText = (rawMode === "bash" && display.startsWith("!")) ? display.slice(1) : display;
                const matchInStripped = strippedText.lastIndexOf(historyQuery);

                if (matchInStripped !== -1) {
                    setCursorOffset(matchInStripped + historyQuery.length);
                } else {
                    setCursorOffset(text.length);
                }
                return;
            }
        }
    }, [isSearching, historyQuery, savedInput, savedCursor, savedMode, onInputChange, setCursorOffset, onModeChange]);

    useInput((keyInput, key) => {
        if (isSearching) {
            if (key.ctrl && keyInput === "r") {
                performSearch(true);
            } else if (key.escape || key.tab) {
                if (historyMatch) {
                    const rawMode = detectInputModeFromPrefix(historyMatch.display);
                    const text = ((rawMode === "bash" && historyMatch.display.startsWith("!"))
                        ? historyMatch.display.slice(1)
                        : historyMatch.display);
                    onInputChange(text);
                    onModeChange(rawMode);
                    setCursorOffset(text.length);
                }
                cancelSearch();
            } else if ((key.ctrl && keyInput === "c") || (key.backspace && historyQuery === "")) {
                onInputChange(savedInput);
                setCursorOffset(savedCursor);
                onModeChange(savedMode);
                cancelSearch();
            } else if (key.return) {
                if (historyQuery.length === 0) {
                    onSubmit({ display: savedInput, pastedContents: {} }); // Mock?
                } else if (historyMatch) {
                    onSubmit(historyMatch);
                }
                cancelSearch();
            }
        } else if (key.ctrl && keyInput === "r") {
            trackEvent("history-search", {});
            setIsSearching(true);
            setSavedInput(input);
            setSavedCursor(cursorOffset);
            setSavedMode(mode);
            setHistoryQuery("");

            historyIterator.current = readHistoryStream();
            visitedHistory.current.clear();
        }
    }, { isActive: true });

    useEffect(() => {
        if (abortController.current) abortController.current.abort();
        const ac = new AbortController();
        abortController.current = ac;

        performSearch(false, ac.signal);

        return () => {
            ac.abort();
        };
    }, [historyQuery]);

    return {
        historyQuery,
        setHistoryQuery,
        historyMatch,
        historyFailedMatch
    };
}

// Logic based on Ht2 from chunk_512.ts
export function usePromptSuggestion({ inputValue, isAssistantResponding }: { inputValue: string; isAssistantResponding: boolean }) {
    const [state, setState] = useAppState();
    const { text, promptId, shownAt, acceptedAt, generationRequestId } = state.promptSuggestion;

    const suggestion = (isAssistantResponding || inputValue.length > 0) ? null : text;
    const isVisible = text && shownAt > 0;

    const markAccepted = useCallback(() => {
        if (!isVisible) return;
        setState((prev: any) => ({
            ...prev,
            promptSuggestion: {
                ...prev.promptSuggestion,
                acceptedAt: Date.now()
            }
        }));
    }, [isVisible, setState]);

    const markShown = useCallback(() => {
        if (suggestion && shownAt === 0) {
            setState((prev: any) => ({
                ...prev,
                promptSuggestion: {
                    ...prev.promptSuggestion,
                    shownAt: Date.now()
                }
            }));
        }
    }, [suggestion, shownAt, setState]);

    const logOutcomeAtSubmission = useCallback((submittedText: string) => {
        if (!isVisible) return;
        const accepted = acceptedAt > shownAt;
        const matched = accepted || submittedText === text;
        const timestamp = matched ? (acceptedAt || Date.now()) : Date.now();

        logEvent("tengu_prompt_suggestion", {
            outcome: matched ? "accepted" : "ignored",
            prompt_id: promptId,
            ...(generationRequestId && { generationRequestId }),
            ...(matched && { acceptMethod: accepted ? "tab" : "enter" }),
            similarity: Math.round(submittedText.length / (text?.length || 1) * 100) / 100
        });

        // Reset suggestion
        setState((prev: any) => ({
            ...prev,
            promptSuggestion: {
                text: null,
                promptId: null,
                shownAt: 0,
                acceptedAt: 0,
                generationRequestId: null
            }
        }));
    }, [isVisible, acceptedAt, shownAt, text, promptId, generationRequestId, setState]);

    return {
        suggestion,
        markAccepted,
        markShown,
        logOutcomeAtSubmission
    };
}

// Logic based on il2 from chunk_501.ts
export function useInputUndoBuffer({ maxBufferSize, debounceMs }: { maxBufferSize: number; debounceMs: number }) {
    const [buffer, setBuffer] = useState<any[]>([]);
    const [index, setIndex] = useState(-1);
    const lastPushTime = useRef(0);
    const timeoutRef = useRef<NodeJS.Timeout | null>(null);

    const pushToBuffer = useCallback((text: string, cursorOffset: number, pastedContents: Record<string, any> = {}) => {
        const now = Date.now();
        if (timeoutRef.current) {
            clearTimeout(timeoutRef.current);
            timeoutRef.current = null;
        }

        if (now - lastPushTime.current < debounceMs) {
            timeoutRef.current = setTimeout(() => {
                pushToBuffer(text, cursorOffset, pastedContents);
            }, debounceMs);
            return;
        }

        lastPushTime.current = now;
        setBuffer((prev: any[]) => {
            const currentSlice = index >= 0 ? prev.slice(0, index + 1) : prev;
            const last = currentSlice[currentSlice.length - 1];
            if (last && last.text === text) return currentSlice;

            const newItem = { text, cursorOffset, pastedContents, timestamp: now };
            const newBuffer = [...currentSlice, newItem];
            return newBuffer.length > maxBufferSize ? newBuffer.slice(-maxBufferSize) : newBuffer;
        });
        setIndex(prev => {
            const nextIndex = prev >= 0 ? prev + 1 : buffer.length;
            return Math.min(nextIndex, maxBufferSize - 1);
        });
    }, [maxBufferSize, debounceMs, index, buffer.length]);

    const undo = useCallback(() => {
        if (index < 0 || buffer.length === 0) return undefined;
        // Undo moves index back
        const prevIndex = Math.max(0, index - 1);
        const item = buffer[prevIndex];
        if (item) {
            setIndex(prevIndex);
            return item;
        }
    }, [buffer, index]);

    const clearBuffer = useCallback(() => {
        setBuffer([]);
        setIndex(-1);
        lastPushTime.current = 0;
        if (timeoutRef.current) clearTimeout(timeoutRef.current);
    }, []);

    const canUndo = index > 0 && buffer.length > 1;

    return {
        pushToBuffer,
        undo,
        canUndo,
        clearBuffer
    };
}

// Logic based on ns2 from chunk_511.ts
export function usePasteListener({ input, pastedContents, onInputChange, setCursorOffset, setPastedContents }: any) {
    const [hasProcessed, setHasProcessed] = useState(false);

    useEffect(() => {
        if (hasProcessed) return;
        if (input.length <= 10000) return; // v17 threshold

        // handleLargePaste logic (ls2) simplified
        // In real impl, it truncates and creates an attachment
        const TRUNCATE_THRESHOLD = 10000;
        const HEAD_TAIL_SIZE = 500; // cs2 / 2

        const pasteIds = Object.keys(pastedContents).map(Number);
        const nextId = pasteIds.length > 0 ? Math.max(...pasteIds) + 1 : 1;

        const head = input.slice(0, HEAD_TAIL_SIZE);
        const tail = input.slice(-HEAD_TAIL_SIZE);
        const middle = input.slice(HEAD_TAIL_SIZE, -HEAD_TAIL_SIZE);

        // Count lines in middle
        const lines = middle.split('\n').length;
        const placeholder = `[...Truncated text #${nextId} +${lines} lines...]`;

        const newInput = head + placeholder + tail;

        onInputChange(newInput);
        setCursorOffset(newInput.length);
        setPastedContents({
            ...pastedContents,
            [nextId]: {
                id: nextId,
                type: "text",
                content: middle
            }
        });
        setHasProcessed(true);

    }, [input, hasProcessed, pastedContents, onInputChange, setCursorOffset, setPastedContents]);

    useEffect(() => {
        if (input === "") setHasProcessed(false);
    }, [input]);
}

export function usePromptSubmissionStatus(args: { input: string; submitCount: number }): string {
    return args.input.length > 0 ? "Ready" : 'Try "create a util logging.py that..."';
}

// Logic based on Ac2 from chunk_493.ts
export function useHistoryNavigation(
    onSelect: (input: string, mode: string, pastedContents: Record<string, any>) => void,
    input: string,
    pastedContents: Record<string, any>,
    setCursorOffset: (value: number) => void
) {
    const [historyIndex, setHistoryIndex] = useState(0);
    // tempInput stores the input state before user started navigating history
    const [tempInput, setTempInput] = useState<{ display: string, pastedContents: any } | undefined>(undefined);
    const hasShownHint = useRef(false);
    const { addNotification, removeNotification } = useNotifications();
    const historyBuffer = useRef<any[]>([]);
    const historyPointer = useRef(0);

    // History loader state (Lo5 logic)
    const historyLoaderPromise = useRef<Promise<any[]> | null>(null);
    const historyLoaderTarget = useRef(0);

    const loadHistory = useCallback(async (count: number) => {
        const BATCH_SIZE = 10;
        const targetCount = Math.ceil(count / BATCH_SIZE) * BATCH_SIZE;

        // If a request covering this is already in progress, return it
        if (historyLoaderPromise.current && historyLoaderTarget.current >= targetCount) {
            return historyLoaderPromise.current;
        }

        // Wait for any current request to finish (sequential loading)
        if (historyLoaderPromise.current) {
            await historyLoaderPromise.current;
        }

        historyLoaderTarget.current = targetCount;

        const loader = (async () => {
            const buffer: any[] = [];
            let loaded = 0;
            const iterator = readHistoryStream();
            for await (const item of iterator) {
                buffer.push(item);
                loaded++;
                // Stop when we have enough items
                if (loaded >= targetCount) break;
            }
            return buffer;
        })();

        historyLoaderPromise.current = loader;

        try {
            return await loader;
        } finally {
            // Reset loader state
            historyLoaderPromise.current = null;
            historyLoaderTarget.current = 0;
        }
    }, []);

    const handleSelect = useCallback((item: any, isUp: boolean) => {
        if (!item || !item.display) return;

        const rawMode = detectInputModeFromPrefix(item.display);

        // Strip prefix if necessary, based on F in chunk_493.ts
        // "bash" or "background" (mapped to bash here?) -> remove prefix
        const text = (rawMode === "bash" && item.display.startsWith("!"))
            ? item.display.slice(1)
            : item.display;

        onSelect(text, rawMode, item.pastedContents || {});

        // Cursor positioning: 0 if going up (backwards in history), end if going down
        setCursorOffset(isUp ? 0 : text.length);
    }, [onSelect, setCursorOffset]);

    const showSearchHint = useCallback(() => {
        addNotification({
            key: "search-history-hint",
            jsx: (
                <Text>
                    <Text color="gray">Tip: </Text>
                    <Text bold>Ctrl+R</Text>
                    <Text color="gray"> to search history</Text>
                </Text>
            ),
            priority: "immediate",
            timeoutMs: 5000
        });
    }, [addNotification]);

    const onHistoryUp = useCallback(() => {
        const currentPtr = historyPointer.current;
        historyPointer.current++;

        (async () => {
            if (currentPtr === 0) {
                const hasInput = input.trim() !== "";
                setTempInput(hasInput ? { display: input, pastedContents } : undefined);
            }

            const neededIndex = currentPtr + 1; // 1-based count for loadHistory

            // Check if we need to load more history
            if (historyBuffer.current.length < neededIndex) {
                const newHistory = await loadHistory(neededIndex);
                if (newHistory.length > historyBuffer.current.length) {
                    historyBuffer.current = newHistory;
                }
            }

            if (currentPtr >= historyBuffer.current.length) {
                // End of history
                historyPointer.current--;
                return;
            }

            const item = historyBuffer.current[currentPtr];
            setHistoryIndex(currentPtr + 1);
            handleSelect(item, true); // true = isUp

            if (currentPtr + 1 >= 2 && !hasShownHint.current) {
                hasShownHint.current = true;
                showSearchHint();
            }
        })();
    }, [input, pastedContents, handleSelect, showSearchHint, loadHistory]);

    const onHistoryDown = useCallback(() => {
        const currentPtr = historyPointer.current;
        if (currentPtr > 1) {
            historyPointer.current--;
            setHistoryIndex(currentPtr - 1);
            const item = historyBuffer.current[currentPtr - 2];
            handleSelect(item, false); // false = isDown (cursor at end)
        } else if (currentPtr === 1) {
            historyPointer.current = 0;
            setHistoryIndex(0);
            if (tempInput) {
                // Restore temp input
                onSelect(tempInput.display, detectInputModeFromPrefix(tempInput.display), tempInput.pastedContents);
                setCursorOffset(tempInput.display.length);
            } else {
                onSelect("", "prompt", {});
                setCursorOffset(0);
            }
        }
        return currentPtr <= 0;
    }, [tempInput, handleSelect, onSelect, setCursorOffset]);

    const resetHistory = useCallback(() => {
        setTempInput(undefined);
        setHistoryIndex(0);
        historyPointer.current = 0;
        removeNotification("search-history-hint");
        historyBuffer.current = [];
    }, [removeNotification]);

    const dismissSearchHint = useCallback(() => {
        removeNotification("search-history-hint");
    }, [removeNotification]);

    return {
        resetHistory,
        onHistoryUp,
        onHistoryDown,
        dismissSearchHint,
        historyIndex
    };
}

// Logic based on cl2 from chunk_501.ts
export function useMcpAtMentionListener(clients: any, handler: (mention: any) => void) {
    const lastClientRef = useRef<any>(undefined);

    useEffect(() => {
        if (!Array.isArray(clients)) return;

        // Helper to find client with 'ide' name
        const ideClient = clients.find((c: any) => c.name === 'ide' && c.type === 'connected');

        if (lastClientRef.current !== ideClient) {
            lastClientRef.current = ideClient;
        }

        if (ideClient && ideClient.client) {
            const client = ideClient.client;

            // Subscribe to notifications if supported
            if (typeof client.setNotificationHandler === 'function') {
                client.setNotificationHandler({ method: 'at_mentioned' }, (notification: any) => {
                    if (lastClientRef.current !== ideClient) return;

                    try {
                        const { params } = notification;
                        if (!params) return;

                        // Original logic adds 1 to line numbers (0-based to 1-based?)
                        const lineStart = params.lineStart !== undefined ? params.lineStart + 1 : undefined;
                        const lineEnd = params.lineEnd !== undefined ? params.lineEnd + 1 : undefined;

                        handler({
                            filePath: params.filePath,
                            lineStart,
                            lineEnd
                        });
                    } catch (error) {
                        console.error("Error handling at_mentioned notification", error);
                    }
                });
            }
        }
    }, [clients, handler]);
}

export function useInputKeypress(handler: (input: string, key: any) => void) {
    useInput((input, key) => {
        handler(input, key);
    });
}

export function useTerminalSize() {
    return { columns: process.stdout.columns || 80, rows: process.stdout.rows || 24 };
}

// Hotkey checkers
export const modeCycleHotkey = { check: (input: string, key: any) => key.ctrl && input === 'm', displayText: "Ctrl+M" };
export const modelPickerHotkey = { check: (input: string, key: any) => key.ctrl && input === 'p', displayText: "Ctrl+P" };
export const thinkingPickerHotkey = { check: (input: string, key: any) => key.ctrl && input === 't', displayText: "Ctrl+T" };
export const imagePasteHotkey = { displayText: "Ctrl+V" }; // Mock
