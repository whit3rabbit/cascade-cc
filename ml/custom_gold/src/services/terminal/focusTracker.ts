/**
 * Handles terminal focus tracking sequences (\x1B[I for focus in, \x1B[O for focus out).
 */

let isFocused = true;
const focusChangeListeners = new Set<(focused: boolean) => void>();

function handleTerminalFocusData(data: Buffer | string) {
    const input = data.toString();
    if (input.includes("\x1B[I")) {
        isFocused = true;
        focusChangeListeners.forEach(listener => listener(true));
    }
    if (input.includes("\x1B[O")) {
        isFocused = false;
        focusChangeListeners.forEach(listener => listener(false));
    }
}

/**
 * Initializes focus tracking by listening to stdin and enabling focus reporting.
 */
export function initFocusTracking() {
    const cleanup = () => {
        if (focusChangeListeners.size === 0) return;
        process.stdin.off("data", handleTerminalFocusData);
        process.stdout.write("\x1B[?1004l"); // Disable focus reporting
    };

    process.on("exit", cleanup);

    // Enable focus reporting and start listening
    process.stdout.write("\x1B[?1004h");
    process.stdin.on("data", handleTerminalFocusData);
}

/**
 * Subscribes to focus change events.
 */
export function onFocusChange(callback: (focused: boolean) => void): () => void {
    focusChangeListeners.add(callback);
    return () => focusChangeListeners.delete(callback);
}

/**
 * Returns whether the terminal is currently focused.
 */
export function isTerminalFocused(): boolean {
    return isFocused;
}
