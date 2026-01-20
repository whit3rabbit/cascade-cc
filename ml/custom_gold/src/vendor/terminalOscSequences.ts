/**
 * Terminal Operating System Command (OSC) sequences.
 * Deobfuscated from Tl1 in chunk_203.ts.
 */

export const OSC = "\x1b]";
export const BEL = "\x07";
export const ST = "\x1b\\"; // String Terminator

export const OSC_COMMANDS = {
    SET_TITLE_AND_ICON: 0,
    SET_ICON: 1,
    SET_TITLE: 2,
    SET_COLOR: 4,
    SET_CWD: 7,
    HYPERLINK: 8,
    ITERM2: 9,
    SET_FG_COLOR: 10,
    SET_BG_COLOR: 11,
    SET_CURSOR_COLOR: 12,
    CLIPBOARD: 52,
    RESET_COLOR: 104,
    RESET_FG_COLOR: 110,
    RESET_BG_COLOR: 111,
    RESET_CURSOR_COLOR: 112,
    SEMANTIC_PROMPT: 133
};

/**
 * Creates an OSC sequence.
 * Deobfuscated from A0A in chunk_203.ts.
 */
export function createOscSequence(command: number, ...params: (string | number)[]): string {
    return `${OSC}${command};${params.join(";")}${BEL}`;
}

/**
 * Generates a terminal hyperlink sequence.
 * Deobfuscated from dWB in chunk_203.ts.
 */
export function createHyperlinkSequence(url: string, params?: Record<string, string>): string {
    const paramString = params ? Object.entries(params).map(([k, v]) => `${k}=${v}`).join(":") : "";
    return createOscSequence(OSC_COMMANDS.HYPERLINK, paramString, url);
}

/**
 * Resets a terminal hyperlink.
 */
export const RESET_HYPERLINK = createOscSequence(OSC_COMMANDS.HYPERLINK, "", "");

export const ITERM2_ACTIONS = {
    NOTIFY: 0,
    BADGE: 2,
    PROGRESS: 4
};

export const PROGRESS_STATES = {
    CLEAR: 0,
    SET: 1,
    ERROR: 2,
    INDETERMINATE: 3
};

/**
 * Generates an ITerm2-specific progress bar sequence.
 * Deobfuscated from mt8 in chunk_203.ts.
 */
export function getITerm2ProgressSequence(state: "completed" | "error" | "indeterminate" | "running", percentage?: number): string {
    const pct = Math.max(0, Math.min(100, Math.round(percentage ?? 0)));
    switch (state) {
        case "completed":
            return createOscSequence(OSC_COMMANDS.ITERM2, ITERM2_ACTIONS.PROGRESS, PROGRESS_STATES.CLEAR, "");
        case "error":
            return createOscSequence(OSC_COMMANDS.ITERM2, ITERM2_ACTIONS.PROGRESS, PROGRESS_STATES.ERROR, pct);
        case "indeterminate":
            return createOscSequence(OSC_COMMANDS.ITERM2, ITERM2_ACTIONS.PROGRESS, PROGRESS_STATES.INDETERMINATE, "");
        case "running":
            return createOscSequence(OSC_COMMANDS.ITERM2, ITERM2_ACTIONS.PROGRESS, PROGRESS_STATES.SET, pct);
    }
}
