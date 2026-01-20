/**
 * Global kill-ring implementation (Emacs-style clipboard history).
 * Deobfuscated from ReA and related functions in chunk_206.ts.
 */

const MAX_HISTORY = 10;
let killRingBuffer: string[] = [];
let isRotationActive = false;
let isYanking = false;
let rotationIndex = 0;
let currentAppendStart = 0;
let currentAppendLength = 0;

/**
 * Adds text to the global kill-ring.
 * Deobfuscated from ReA in chunk_206.ts.
 */
export function addToKillRing(text: string, mode: "append" | "prepend" = "append") {
    if (text.length === 0) return;

    if (isRotationActive && killRingBuffer.length > 0) {
        if (mode === "prepend") {
            killRingBuffer[0] = text + killRingBuffer[0];
        } else {
            killRingBuffer[0] = killRingBuffer[0] + text;
        }
    } else {
        killRingBuffer.unshift(text);
        if (killRingBuffer.length > MAX_HISTORY) {
            killRingBuffer.pop();
        }
        isRotationActive = true;
        isYanking = false;
    }
}

/**
 * Gets the most recent entry from the kill-ring.
 */
export function getRecentKillRingEntry(): string {
    return killRingBuffer[0] ?? "";
}

/**
 * Ends a sequence of "kill" actions.
 */
export function endKillSequence() {
    isRotationActive = false;
}

/**
 * Initializes history rotation for yank-pop.
 */
export function startKillRingRotation(start: number, length: number) {
    currentAppendStart = start;
    currentAppendLength = length;
    isYanking = true;
    rotationIndex = 0;
}

/**
 * Cycles through kill-ring entries.
 * Deobfuscated from aKB in chunk_206.ts.
 */
export function rotateKillRing() {
    if (!isYanking || killRingBuffer.length <= 1) return null;

    rotationIndex = (rotationIndex + 1) % killRingBuffer.length;
    return {
        text: killRingBuffer[rotationIndex] ?? "",
        start: currentAppendStart,
        length: currentAppendLength
    };
}

/**
 * Updates the recorded length of the last yanked item.
 */
export function updateLastYankLength(length: number) {
    currentAppendLength = length;
}

/**
 * Marks the end of yanking.
 */
export function endYank() {
    isYanking = false;
}
