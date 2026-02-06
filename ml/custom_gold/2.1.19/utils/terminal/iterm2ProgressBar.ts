/**
 * File: src/utils/terminal/iterm2ProgressBar.ts
 * Role: Provides utilities for managing the iTerm2 progress bar.
 */

import semver from "semver";

function supportsIterm2Progress(): boolean {
    if (!process.stdout.isTTY) return false;
    if (process.env.WT_SESSION) return false;
    if (process.env.ConEmuANSI || process.env.ConEmuPID || process.env.ConEmuTask) return true;

    const version = semver.coerce(process.env.TERM_PROGRAM_VERSION || "");
    if (!version) return false;

    if (process.env.TERM_PROGRAM === "ghostty") {
        return semver.gte(version, "1.2.0");
    }

    if (process.env.TERM_PROGRAM === "iTerm.app") {
        return semver.gte(version, "3.6.6");
    }

    return false;
}

/**
 * Sets the iTerm2 progress bar value.
 * @param value - Progress percentage (0-100), or -1 to clear.
 */
export function setIterm2Progress(value: number): void {
    if (!supportsIterm2Progress()) return;

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
    if (!supportsIterm2Progress()) return;

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
    if (!supportsIterm2Progress()) return;

    process.stdout.write('\x1b]9;4;0\x07');
}
