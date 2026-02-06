/**
 * File: src/utils/terminal/iterm2ProgressBar.ts
 * Role: Provides utilities for managing the iTerm2 progress bar.
 */

import { EnvService } from "../../services/config/EnvService.js";

/**
 * Sets the iTerm2 progress bar value.
 * @param value - Progress percentage (0-100), or -1 to clear.
 */
export function setIterm2Progress(value: number): void {
    if (EnvService.get("TERM_PROGRAM") !== "iTerm.app") return;

    if (value < 0) {
        clearIterm2Progress();
        return;
    }

    // OSC 9;4;N ST
    // N=0: clear, N=1: busy, N=2: set progress
    // Some versions use OSC 9;4;progress ST where progress is 0-100
    process.stdout.write(`\x1b]9;4;${Math.min(100, Math.max(0, value))}\x07`);
}

/**
 * Sets the iTerm2 progress bar to "busy" (indeterminate) mode.
 */
export function setIterm2Busy(busy: boolean): void {
    if (EnvService.get("TERM_PROGRAM") !== "iTerm.app") return;

    if (busy) {
        process.stdout.write('\x1b]9;4;1\x07');
    } else {
        clearIterm2Progress();
    }
}

/**
 * Clears the iTerm2 progress bar.
 */
export function clearIterm2Progress(): void {
    if (EnvService.get("TERM_PROGRAM") !== "iTerm.app") return;

    process.stdout.write('\x1b]9;4;0\x07');
}
