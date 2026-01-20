import { spawn } from 'node:child_process';

export interface RunProcessResult {
    stdout: string;
    stderr: string;
    code: number | null;
}

export function runProcess(command: string, args: string[], options?: { cwd?: string, env?: NodeJS.ProcessEnv }): Promise<RunProcessResult> {
    return new Promise((resolve, reject) => {
        const proc = spawn(command, args, {
            cwd: options?.cwd,
            env: options?.env || process.env,
            stdio: ['ignore', 'pipe', 'pipe']
        });

        let stdout = '';
        let stderr = '';

        proc.stdout?.on('data', (data) => stdout += data.toString());
        proc.stderr?.on('data', (data) => stderr += data.toString());

        proc.on('error', (err) => reject(err));

        proc.on('close', (code) => {
            resolve({
                stdout,
                stderr,
                code
            });
        });
    });
}
