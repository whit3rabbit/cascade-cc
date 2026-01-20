import { execa } from "execa";
import { logError } from "../../services/logger/loggerService.js";

export interface ShellResult {
    stdout: string;
    stderr: string;
    code: number;
    error?: string;
}

export interface ShellOptions {
    abortSignal?: AbortSignal;
    timeout?: number;
    preserveOutputOnError?: boolean;
    cwd?: string;
    env?: NodeJS.ProcessEnv;
    stdin?: any;
    maxBuffer?: number;
    shell?: boolean;
}

const DEFAULT_TIMEOUT = 10 * 60 * 1000; // 10 minutes

/**
 * Executes a shell command and returns the result.
 * Deobfuscated from WQ/g6 in chunk_19.ts.
 */
export async function execShellCommand(
    command: string,
    args: string[],
    options: ShellOptions = {}
): Promise<ShellResult> {
    const {
        abortSignal,
        timeout = DEFAULT_TIMEOUT,
        preserveOutputOnError = true,
        cwd,
        env,
        maxBuffer = 1e6,
        shell = false,
        stdin
    } = options;

    try {
        const result = await execa(command, args, {
            maxBuffer,
            signal: abortSignal,
            timeout,
            cwd,
            env,
            shell,
            stdin,
            reject: false
        });

        if (result.failed) {
            if (preserveOutputOnError) {
                const code = result.exitCode ?? 1;
                return {
                    stdout: result.stdout || "",
                    stderr: result.stderr || "",
                    code,
                    error: typeof result.signal === "string" ? result.signal : String(code)
                };
            } else {
                return {
                    stdout: "",
                    stderr: "",
                    code: result.exitCode ?? 1
                };
            }
        }

        return {
            stdout: result.stdout,
            stderr: result.stderr,
            code: 0
        };
    } catch (err: any) {
        logError("shell", err);
        return {
            stdout: "",
            stderr: "",
            code: 1,
            error: err.message
        };
    }
}
