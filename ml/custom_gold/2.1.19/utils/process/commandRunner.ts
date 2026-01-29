/**
 * File: src/utils/process/commandRunner.ts
 * Role: Unified interface for running system commands using Node.js child_process.
 */

import { spawn, SpawnOptions } from 'node:child_process';

export interface CommandResult {
    code: number;
    stdout: string;
    stderr: string;
}

/**
 * Runs a shell command and returns its exit code and output streams.
 * 
 * @param command - The command to execute.
 * @param args - Array of arguments for the command.
 * @param options - Options to pass to spawn.
 * @returns {Promise<CommandResult>}
 */
export function runCommand(
    command: string,
    args: string[] = [],
    options: SpawnOptions = {}
): Promise<CommandResult> {
    return new Promise((resolve, reject) => {
        const child = spawn(command, args, {
            stdio: ['ignore', 'pipe', 'pipe'],
            ...options
        });

        let stdout = '';
        let stderr = '';

        child.stdout?.on('data', (data: Buffer | string) => {
            stdout += data.toString();
        });

        child.stderr?.on('data', (data: Buffer | string) => {
            stderr += data.toString();
        });

        child.on('close', (code: number | null) => {
            resolve({
                code: code ?? 0,
                stdout: stdout.trim(),
                stderr: stderr.trim()
            });
        });

        child.on('error', (err: Error) => {
            reject(err);
        });
    });
}
