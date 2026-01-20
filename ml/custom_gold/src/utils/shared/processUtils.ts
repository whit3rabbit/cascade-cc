import { ChildProcess, exec, spawn } from "node:child_process";
import { PassThrough } from "node:stream";

/**
 * Simple generic ring buffer.
 * Deobfuscated from VNA in chunk_188.ts.
 */
export class RingBuffer<T> {
    private buffer: T[];
    private head = 0;
    private size = 0;

    constructor(private capacity: number) {
        this.buffer = Array(capacity);
    }

    add(item: T): void {
        this.buffer[this.head] = item;
        this.head = (this.head + 1) % this.capacity;
        if (this.size < this.capacity) this.size++;
    }

    addAll(items: T[]): void {
        for (const item of items) this.add(item);
    }

    getRecent(count: number): T[] {
        const recent: T[] = [];
        const actualCount = Math.min(count, this.size);
        const startIdx = this.size < this.capacity ? 0 : this.head;

        for (let i = 0; i < actualCount; i++) {
            const idx = (startIdx + this.size - actualCount + i) % this.capacity;
            recent.push(this.buffer[idx]);
        }
        return recent;
    }

    toArray(): T[] {
        const result: T[] = [];
        const startIdx = this.size < this.capacity ? 0 : this.head;
        for (let i = 0; i < this.size; i++) {
            result.push(this.buffer[(startIdx + i) % this.capacity]);
        }
        return result;
    }

    clear(): void {
        this.head = 0;
        this.size = 0;
    }

    length(): number {
        return this.size;
    }
}

/**
 * Circular buffer for process output with truncation support.
 * Deobfuscated from rZA in chunk_188.ts.
 */
export class OutputBuffer {
    private content = "";
    private isTruncated = false;
    private totalBytesReceived = 0;

    constructor(private maxSize = 67108864) { } // Default 64MB

    append(data: string | Buffer): void {
        const str = typeof data === "string" ? data : data.toString();
        this.totalBytesReceived += str.length;

        if (this.isTruncated && this.content.length >= this.maxSize) return;

        if (this.content.length + str.length > this.maxSize) {
            const remainingSpace = this.maxSize - this.content.length;
            if (remainingSpace > 0) {
                this.content += str.slice(0, remainingSpace);
            }
            this.isTruncated = true;
        } else {
            this.content += str;
        }
    }

    toString(): string {
        if (!this.isTruncated) return this.content;
        const truncatedAmountKb = Math.round((this.totalBytesReceived - this.maxSize) / 1024);
        return `${this.content}\n... [output truncated - ${truncatedAmountKb}KB removed]`;
    }

    clear(): void {
        this.content = "";
        this.isTruncated = false;
        this.totalBytesReceived = 0;
    }

    get length(): number { return this.content.length; }
    get truncated(): boolean { return this.isTruncated; }
    get totalBytes(): number { return this.totalBytesReceived; }
}

/**
 * Runs a process with timeouts and output capture.
 * Deobfuscated from rsA in chunk_188.ts.
 */
export function runProcessWithOutput(
    process: ChildProcess,
    abortSignal: AbortSignal,
    timeoutMs: number,
    onProgress?: (recent: string, all: string, count: number) => void
) {
    let status: "running" | "killed" | "completed" | "backgrounded" = "running";
    const stdoutBuffer = new OutputBuffer();
    const stderrBuffer = new OutputBuffer();
    const ringBuffer = new RingBuffer<string>(1000);
    let totalLines = 0;

    const handleData = (data: Buffer) => {
        const lines = data.toString().split("\n").filter(l => l.trim());
        ringBuffer.addAll(lines);
        totalLines += lines.length;

        if (onProgress) {
            onProgress(
                ringBuffer.getRecent(5).join("\n"),
                ringBuffer.getRecent(100).join("\n"),
                totalLines
            );
        }
    };

    process.stdout?.on("data", (data) => {
        stdoutBuffer.append(data);
        handleData(data);
    });

    process.stderr?.on("data", (data) => {
        stderrBuffer.append(data);
        handleData(data);
    });

    const kill = () => {
        if (status === "running") {
            status = "killed";
            if (process.pid) {
                terminateChildProcess(process.pid, "SIGKILL");
            } else {
                process.kill("SIGKILL");
            }
        }
    };

    const abortHandler = () => kill();
    abortSignal.addEventListener("abort", abortHandler, { once: true });

    let onTimeoutHandler: ((background: (taskId: string) => any) => void) | undefined;

    const result = new Promise<{
        code: number | null;
        stdout: string;
        stderr: string;
        interrupted: boolean;
        backgroundTaskId?: string;
    }>((resolve) => {
        let backgroundTaskId: string | undefined;

        const background = (taskId: string) => {
            if (status === "running") {
                backgroundTaskId = taskId;
                status = "backgrounded";
                cleanup();
                return {
                    stdoutStream: new PassThrough().end(stdoutBuffer.toString()),
                    stderrStream: new PassThrough().end(stderrBuffer.toString())
                };
            }
            return null;
        };

        const cleanup = () => {
            if (timeout) clearTimeout(timeout);
            abortSignal.removeEventListener("abort", abortHandler);
        };

        const timeout = setTimeout(() => {
            if (status === "running") {
                if (onTimeoutHandler) {
                    onTimeoutHandler(background);
                } else {
                    kill();
                }
            }
        }, timeoutMs);

        process.on("close", (code, signal) => {
            cleanup();
            if (status === "running" || status === "backgrounded") {
                status = "completed";
            }
            const exitCode = code !== null ? code : (signal === "SIGTERM" ? 144 : 1);
            resolve({
                code: exitCode,
                stdout: stdoutBuffer.toString(),
                stderr: stderrBuffer.toString(),
                interrupted: exitCode === 137 || status === "killed",
                backgroundTaskId
            });
        });

        process.on("error", (err) => {
            cleanup();
            resolve({
                code: 1,
                stdout: stdoutBuffer.toString(),
                stderr: (err instanceof Error ? err.message : String(err)),
                interrupted: false,
                backgroundTaskId
            });
        });
    });

    return {
        get status() { return status; },
        background: (taskId: string) => {
            if (status === "running") {
                status = "backgrounded";
                // In a real implementation we might need to detach or something
                return {
                    stdoutStream: new PassThrough().end(stdoutBuffer.toString()),
                    stderrStream: new PassThrough().end(stderrBuffer.toString())
                };
            }
            return null;
        },
        kill,
        onTimeout: (handler: (background: (taskId: string) => any) => void) => {
            onTimeoutHandler = handler;
        },
        result
    };
}

/**
 * Terminates a process and all its children.
 * Deobfuscated from vYB in chunk_188.ts.
 */
export function terminateChildProcess(pid: number, signal: string | number = "SIGTERM", callback?: (err?: Error) => void) {
    if (process.platform === "win32") {
        exec(`taskkill /pid ${pid} /T /F`, (err) => {
            if (callback) callback(err || undefined);
        });
        return;
    }

    const children: Record<number, number[]> = { [pid]: [] };
    const pidsToProcess: Record<number, number> = { [pid]: 1 };

    const getChildren = (parentPid: number) => {
        const cmd = process.platform === "darwin" ? "pgrep" : "ps";
        const args = process.platform === "darwin" ? ["-P", String(parentPid)] : ["-o", "pid", "--no-headers", "--ppid", String(parentPid)];

        const child = spawn(cmd, args);
        let stdout = "";
        child.stdout.on("data", (data) => stdout += data.toString());

        child.on("close", (code) => {
            delete pidsToProcess[parentPid];
            if (code === 0) {
                const foundPids = stdout.match(/\d+/g);
                if (foundPids) {
                    foundPids.forEach(p => {
                        const childPid = parseInt(p, 10);
                        children[parentPid].push(childPid);
                        children[childPid] = [];
                        pidsToProcess[childPid] = 1;
                        getChildren(childPid);
                    });
                }
            }
            if (Object.keys(pidsToProcess).length === 0) {
                killAll();
            }
        });
    };

    const killAll = () => {
        const killed: Record<number, boolean> = {};
        try {
            Object.keys(children).forEach(parent => {
                const p = parseInt(parent, 10);
                children[p].forEach(child => {
                    if (!killed[child]) {
                        try { process.kill(child, signal); } catch (e: any) { if (e.code !== "ESRCH") throw e; }
                        killed[child] = true;
                    }
                });
                if (!killed[p]) {
                    try { process.kill(p, signal); } catch (e: any) { if (e.code !== "ESRCH") throw e; }
                    killed[p] = true;
                }
            });
            if (callback) callback();
        } catch (err: any) {
            if (callback) callback(err);
            else throw err;
        }
    };

    getChildren(pid);
}
