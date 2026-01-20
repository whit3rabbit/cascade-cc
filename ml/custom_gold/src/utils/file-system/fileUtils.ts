import * as fs from "fs";
import { stat as statAsync, open as openAsync } from "fs/promises";

const SLOW_OPERATION_THRESHOLD_MS = 5;

/**
 * Measures the duration of a file system operation and logs if it's slow.
 */
export function measureFsOperation<T>(name: string, operation: () => T): T {
    const start = performance.now();
    try {
        return operation();
    } finally {
        const duration = performance.now() - start;
        if (duration > SLOW_OPERATION_THRESHOLD_MS) {
            console.warn(`[SLOW OPERATION DETECTED] fs.${name} (${duration.toFixed(1)}ms)`);
        }
    }
}

/**
 * Resolves a symlink and returns the resolved path and whether it was a symlink.
 */
export function resolveSymlink(fileSystem: any, path: string): { resolvedPath: string; isSymlink: boolean } {
    if (!fileSystem.existsSync(path)) return {
        resolvedPath: path,
        isSymlink: false
    };
    try {
        const resolvedPath = fileSystem.realpathSync(path);
        return {
            resolvedPath,
            isSymlink: resolvedPath !== path
        };
    } catch (e) {
        return {
            resolvedPath: path,
            isSymlink: false
        };
    }
}

/**
 * Checks if a resolved symlink path is already in the provided set.
 */
export function isSymlinkInSet(fileSystem: any, path: string, seenPaths: Set<string>): boolean {
    const { resolvedPath } = resolveSymlink(fileSystem, path);
    if (seenPaths.has(resolvedPath)) return true;
    seenPaths.add(resolvedPath);
    return false;
}

/**
 * Gets all resolved paths for a given path (including symlink target if applicable).
 */
export function getResolvedPaths(path: string): string[] {
    const paths: string[] = [];
    const fs = getFileSystem();
    paths.push(path);
    const { resolvedPath, isSymlink } = resolveSymlink(fs, path);
    if (isSymlink && resolvedPath !== path) {
        paths.push(resolvedPath);
    }
    return paths;
}

/**
 * Returns the current file system implementation.
 */
export function getFileSystem() {
    return fileSystem;
}

/**
 * Reads lines from a file in reverse order.
 */
export async function* readLinesReversed(path: string): AsyncGenerator<string> {
    const fileHandle = await openAsync(path, "r");
    try {
        let fileSize = (await fileHandle.stat()).size;
        let remainder = "";
        const buffer = Buffer.alloc(4096);
        while (fileSize > 0) {
            const bytesToRead = Math.min(4096, fileSize);
            fileSize -= bytesToRead;
            await fileHandle.read(buffer, 0, bytesToRead, fileSize);
            const lines = (buffer.toString("utf8", 0, bytesToRead) + remainder).split(/\r?\n/);
            remainder = lines[0] || "";
            for (let i = lines.length - 1; i >= 1; i--) {
                const line = lines[i];
                if (line) yield line;
            }
        }
        if (remainder) yield remainder;
    } finally {
        await fileHandle.close();
    }
}

export const nodeFileSystem = {
    cwd() {
        return process.cwd();
    },
    existsSync(path: string) {
        return measureFsOperation("existsSync", () => fs.existsSync(path));
    },
    async stat(path: string) {
        return statAsync(path);
    },
    statSync(path: string) {
        return measureFsOperation("statSync", () => fs.statSync(path));
    },
    lstatSync(path: string) {
        return measureFsOperation("lstatSync", () => fs.lstatSync(path));
    },
    readFileSync(path: string, options: { encoding: BufferEncoding }) {
        return measureFsOperation("readFileSync", () => fs.readFileSync(path, {
            encoding: options.encoding
        }));
    },
    readFileBytesSync(path: string) {
        return measureFsOperation("readFileBytesSync", () => fs.readFileSync(path));
    },
    readSync(path: string, options: { length: number }) {
        return measureFsOperation("readSync", () => {
            let fd: number | undefined = undefined;
            try {
                fd = fs.openSync(path, "r");
                const buffer = Buffer.alloc(options.length);
                const bytesRead = fs.readSync(fd, buffer, 0, options.length, 0);
                return {
                    buffer,
                    bytesRead
                };
            } finally {
                if (fd !== undefined) fs.closeSync(fd);
            }
        });
    },
    writeFileSync(path: string, data: string | Buffer, options: { encoding?: BufferEncoding; mode?: number; flush?: boolean }) {
        return measureFsOperation("writeFileSync", () => {
            const exists = fs.existsSync(path);
            if (!options.flush) {
                const writeOptions: any = {
                    encoding: options.encoding
                };
                if (!exists) writeOptions.mode = options.mode ?? 0o600;
                else if (options.mode !== undefined) writeOptions.mode = options.mode;
                fs.writeFileSync(path, data, writeOptions);
                return;
            }
            let fd: number | undefined = undefined;
            try {
                const mode = !exists ? options.mode ?? 0o600 : options.mode;
                fd = fs.openSync(path, "w", mode);
                fs.writeFileSync(fd, data, {
                    encoding: options.encoding
                });
                fs.fsyncSync(fd);
            } finally {
                if (fd !== undefined) fs.closeSync(fd);
            }
        });
    },
    appendFileSync(path: string, data: string | Buffer, options?: { mode?: number }) {
        return measureFsOperation("appendFileSync", () => {
            if (!fs.existsSync(path)) {
                const mode = options?.mode ?? 0o600;
                const fd = fs.openSync(path, "a", mode);
                try {
                    fs.appendFileSync(fd, data);
                } finally {
                    fs.closeSync(fd);
                }
            } else {
                fs.appendFileSync(path, data);
            }
        });
    },
    copyFileSync(src: string, dest: string) {
        return measureFsOperation("copyFileSync", () => fs.copyFileSync(src, dest));
    },
    unlinkSync(path: string) {
        return measureFsOperation("unlinkSync", () => fs.unlinkSync(path));
    },
    renameSync(oldPath: string, newPath: string) {
        return measureFsOperation("renameSync", () => fs.renameSync(oldPath, newPath));
    },
    linkSync(target: string, path: string) {
        return measureFsOperation("linkSync", () => fs.linkSync(target, path));
    },
    symlinkSync(target: string, path: string) {
        return measureFsOperation("symlinkSync", () => fs.symlinkSync(target, path));
    },
    readlinkSync(path: string) {
        return measureFsOperation("readlinkSync", () => fs.readlinkSync(path));
    },
    realpathSync(path: string) {
        return measureFsOperation("realpathSync", () => fs.realpathSync(path));
    },
    mkdirSync(path: string) {
        return measureFsOperation("mkdirSync", () => {
            if (!fs.existsSync(path)) fs.mkdirSync(path, {
                recursive: true,
                mode: 0o700
            });
        });
    },
    readdirSync(path: string) {
        return measureFsOperation("readdirSync", () => fs.readdirSync(path, {
            withFileTypes: true
        }));
    },
    readdirStringSync(path: string) {
        return measureFsOperation("readdirStringSync", () => fs.readdirSync(path));
    },
    isDirEmptySync(path: string) {
        return measureFsOperation("isDirEmptySync", () => {
            return (this as any).readdirSync(path).length === 0;
        });
    },
    rmdirSync(path: string) {
        return measureFsOperation("rmdirSync", () => fs.rmdirSync(path));
    },
    rmSync(path: string, options?: fs.RmOptions) {
        return measureFsOperation("rmSync", () => fs.rmSync(path, options));
    },
    createWriteStream(path: string) {
        return fs.createWriteStream(path);
    }
};

export let fileSystem: any = nodeFileSystem;

export function setFileSystem(fs: any) {
    fileSystem = fs;
}
