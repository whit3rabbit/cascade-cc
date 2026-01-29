/**
 * File: src/utils/shared/uiUtils.ts
 * Role: Low-level terminal and TTY interaction helpers.
 */

import { ReadStream as TtyReadStream } from "node:tty";
import { openSync } from "node:fs";

/**
 * Attempts to open and return a TTY ReadStream.
 */
export function getTTY(): TtyReadStream | null {
    try {
        return new TtyReadStream(openSync("/dev/tty", "r"));
    } catch (error) {
        console.warn("[UIUtils] Failed to get TTY:", error);
        return null;
    }
}

/**
 * Normalizes interactive options for terminal tools.
 */
export function getInteractiveOptions(options?: { stdin?: any; stdout?: any }): { stdin: any; stdout: any } {
    return {
        stdin: options?.stdin ?? process.stdin,
        stdout: options?.stdout ?? process.stdout,
    };
}
