import { TerminalEvent } from "../utils/shared/eventEmitter.js";

export interface Key {
    name: string;
    ctrl: boolean;
    meta: boolean;
    shift: boolean;
    option: boolean;
    fn: boolean;
    sequence: string;
    raw: string;
    isPasted: boolean;
    code?: string;
}

/**
 * Keypress event object.
 * Deobfuscated from JeA in chunk_202.ts.
 */
export class KeypressEvent extends TerminalEvent {
    public key: any;
    public input: string;

    constructor(public keypress: Key) {
        super();
        const [normalizedKey, input] = normalizeKey(keypress);
        this.key = normalizedKey;
        this.input = input;
    }
}

/**
 * Main entry point for keypress parsing.
 * Deobfuscated from TWB in chunk_202.ts.
 */
export function parseTerminalInput(state: any, data: string | Buffer): [Key[], any] {
    const input = Buffer.isBuffer(data) ? data.toString() : data;
    const keys: Key[] = [];

    // Very simplified version of the state machine in chunk_202.ts
    for (let i = 0; i < input.length; i++) {
        keys.push(parseKeypress(input[i]));
    }

    return [keys, { ...state, incomplete: "" }];
}

/**
 * Maps a single character/sequence to a key object.
 * Deobfuscated from _WB in chunk_202.ts.
 */
export function parseKeypress(sequence: string): Key {
    const key: Key = {
        name: "",
        ctrl: false,
        meta: false,
        shift: false,
        option: false,
        fn: false,
        sequence,
        raw: sequence,
        isPasted: false
    };

    if (sequence === "\r") {
        key.name = "return";
    } else if (sequence === "\n") {
        key.name = "enter";
    } else if (sequence === "\t") {
        key.name = "tab";
    } else if (sequence === "\x7f") {
        key.name = "backspace";
    } else if (sequence === "\x1b") {
        key.name = "escape";
    } else if (sequence.length === 1 && sequence.charCodeAt(0) <= 26) {
        key.name = String.fromCharCode(sequence.charCodeAt(0) + 96);
        key.ctrl = true;
    } else {
        key.name = sequence.toLowerCase();
        key.shift = sequence !== key.name;
    }

    return key;
}

/**
 * Normalizes key attributes.
 * Deobfuscated from Pt8 in chunk_202.ts.
 */
export function normalizeKey(keypress: Key): [any, string] {
    const meta = ({
        upArrow: keypress.name === "up",
        downArrow: keypress.name === "down",
        leftArrow: keypress.name === "left",
        rightArrow: keypress.name === "right",
        return: keypress.name === "return",
        escape: keypress.name === "escape",
        ctrl: keypress.ctrl,
        shift: keypress.shift,
        tab: keypress.name === "tab",
        backspace: keypress.name === "backspace",
        meta: keypress.meta || keypress.option
    });

    let input = keypress.ctrl ? keypress.name : keypress.sequence;
    if (input.startsWith("\x1b")) input = input.slice(1);

    return [meta, input || ""];
}
