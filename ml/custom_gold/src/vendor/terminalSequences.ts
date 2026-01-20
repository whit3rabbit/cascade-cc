/**
 * Terminal CSI sequences.
 * Deobfuscated from oNA in chunk_202.ts.
 */

export const Sequences = {
    CURSOR_VISIBLE: 25,
    ALT_SCREEN: 47,
    ALT_SCREEN_CLEAR: 1049,
    MOUSE_NORMAL: 1000,
    MOUSE_BUTTON: 1002,
    MOUSE_ANY: 1003,
    FOCUS_EVENTS: 1004,
    BRACKETED_PASTE: 2004,
    SYNCHRONIZED_UPDATE: 2026
};

export function sendCsiH(code: number): string {
    return `\x1b[?${code}h`;
}

export function sendCsiL(code: number): string {
    return `\x1b[?${code}l`;
}

export const SHOW_CURSOR = sendCsiH(Sequences.CURSOR_VISIBLE);
export const HIDE_CURSOR = sendCsiL(Sequences.CURSOR_VISIBLE);
export const ENTER_ALT_SCREEN = sendCsiH(Sequences.ALT_SCREEN_CLEAR);
export const EXIT_ALT_SCREEN = sendCsiL(Sequences.ALT_SCREEN_CLEAR);
export const ENABLE_BRACKETED_PASTE = sendCsiH(Sequences.BRACKETED_PASTE);
export const DISABLE_BRACKETED_PASTE = sendCsiL(Sequences.BRACKETED_PASTE);
