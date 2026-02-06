/**
 * File: src/utils/shared/bashUtils.ts
 * Role: Logic for cleaning, stripping, and executing bash commands.
 */

import { exec, execFile, ExecOptions, ExecFileOptions } from 'node:child_process';

// --- Regex Constants ---
const CWD_RESET_REGEX = /(?:^|\n)(Shell cwd was reset to .+)$/;
const SANDBOX_VIOLATIONS_REGEX = /<sandbox_violations>[\s\S]*?<\/sandbox_violations>/g;
const ANSI_UNDERLINE_REGEX = /\u001b\[([0-9]+;)*4(;[0-9]+)*m|\u001b\[4(;[0-9]+)*m|\u001b\[([0-9]+;)*4m/g;

/**
 * Strips sandbox violation tags from a string.
 */
export function stripSandboxViolations(text: string): string {
    if (typeof text !== 'string') return text;
    return text.replace(SANDBOX_VIOLATIONS_REGEX, "");
}

/**
 * Strips ANSI underline escape codes.
 */
export function stripUnderline(text: string): string {
    if (typeof text !== 'string') return text;
    return text.replace(ANSI_UNDERLINE_REGEX, "");
}

/**
 * Cleans stderr by removing sandbox violations and trimming.
 */
export function cleanSandboxViolations(stderr?: string): { cleanedStderr: string } {
    if (!stderr) return { cleanedStderr: "" };
    const cleanedStderr = stripSandboxViolations(stderr).trim();
    return { cleanedStderr };
}

/**
 * Detects and removes shell CWD reset warnings from stderr.
 */
export function cleanCwdResetWarning(stderr?: string): { cleanedStderr: string, cwdResetWarning: string | null } {
    if (!stderr) return { cleanedStderr: "", cwdResetWarning: null };

    const match = CWD_RESET_REGEX.exec(stderr);
    if (!match) {
        return { cleanedStderr: stderr, cwdResetWarning: null };
    }

    const cwdResetWarning = match[1] || null;
    const cleanedStderr = stderr.replace(CWD_RESET_REGEX, "").trim();

    return { cleanedStderr, cwdResetWarning };
}

export interface BashResult {
    exitCode: number;
    stdout: string;
    stderr: string;
}

/**
 * Executes a bash command and returns the result as a promise.
 */
export async function executeBashCommand(command: string, options: ExecOptions = {}, onProcess?: (pid: number) => void): Promise<BashResult> {
    return new Promise((resolve) => {
        const child = exec(command, options, (error: any, stdout: any, stderr: any) => {
            resolve({
                exitCode: error?.code || 0,
                stdout: stdout?.toString() || "",
                stderr: stderr?.toString() || (error ? error.message : "")
            });
        });

        if (child.pid && onProcess) {
            onProcess(child.pid);
        }
    });
}

/**
 * Spawns a bash command (via execFile) with a callback.
 */
export function spawnBashCommand(
    file: string,
    args: string[],
    options: ExecFileOptions,
    callback: (error: any, stdout: string | Buffer, stderr: string | Buffer) => void
) {
    return execFile(file, args, options, callback as any);
}


// --- Aliases for deobfuscation compatibility ---
export {
    executeBashCommand as YY,
    spawnBashCommand as FX2
};
