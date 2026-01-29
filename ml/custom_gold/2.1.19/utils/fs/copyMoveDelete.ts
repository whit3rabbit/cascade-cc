/**
 * File: src/utils/fs/copyMoveDelete.ts
 * Role: File system operations (copy, move, delete, ensure) using native Node.js methods.
 */

import * as fs from 'node:fs';
import * as fsPromises from 'node:fs/promises';
import * as path from 'node:path';

interface JsonOptions {
    spaces?: number | string;
}

const stringify = (obj: any, options?: JsonOptions) => JSON.stringify(obj, null, options?.spaces || 2);

/**
 * Copies a file or directory.
 */
export const copy = (src: string, dest: string, options?: any, callback?: (err: Error | null) => void) => {
    if (typeof options === 'function') {
        callback = options;
        options = {};
    }
    fs.cp(src, dest, { recursive: true, ...options }, callback!);
};

export const copySync = (src: string, dest: string, options?: any) => {
    fs.cpSync(src, dest, { recursive: true, ...options });
};

/**
 * Removes a file or directory.
 */
export const remove = (dir: string, callback: (err: Error | null) => void) => {
    fs.rm(dir, { recursive: true, force: true }, callback);
};

export const removeSync = (dir: string) => {
    fs.rmSync(dir, { recursive: true, force: true });
};

/**
 * Vacuums a directory (removes and recreates).
 */
export const emptyDir = async (dir: string) => {
    await fsPromises.rm(dir, { recursive: true, force: true });
    await fsPromises.mkdir(dir, { recursive: true });
};

export const emptyDirSync = (dir: string) => {
    fs.rmSync(dir, { recursive: true, force: true });
    fs.mkdirSync(dir, { recursive: true });
};

/**
 * Ensures a file exists, creating parent directories if necessary.
 */
export const ensureFile = (file: string, callback: (err: Error | null) => void) => {
    const dir = path.dirname(file);
    fs.mkdir(dir, { recursive: true }, (err) => {
        if (err && (err as any).code !== 'EEXIST') return callback(err);
        fs.writeFile(file, '', { flag: 'a' }, callback);
    });
};

export const ensureFileSync = (file: string) => {
    const dir = path.dirname(file);
    fs.mkdirSync(dir, { recursive: true });
    fs.writeFileSync(file, '', { flag: 'a' });
};

/**
 * JSON utilities.
 */
export const readJson = async (file: string) => {
    const content = await fsPromises.readFile(file, 'utf8');
    return JSON.parse(content);
};

export const readJsonSync = (file: string) => {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
};

export const writeJson = async (file: string, obj: any, options?: JsonOptions) => {
    await fsPromises.writeFile(file, stringify(obj, options), 'utf8');
};

export const writeJsonSync = (file: string, obj: any, options?: JsonOptions) => {
    fs.writeFileSync(file, stringify(obj, options), 'utf8');
};

export const outputJson = async (file: string, obj: any, options?: JsonOptions) => {
    const dir = path.dirname(file);
    await fsPromises.mkdir(dir, { recursive: true });
    await writeJson(file, obj, options);
};

export const outputJsonSync = (file: string, obj: any, options?: JsonOptions) => {
    const dir = path.dirname(file);
    fs.mkdirSync(dir, { recursive: true });
    writeJsonSync(file, obj, options);
};

/**
 * Moves a file or directory.
 */
export const move = (src: string, dest: string, options?: any, callback?: (err: Error | null) => void) => {
    if (typeof options === 'function') {
        callback = options;
        options = {};
    }
    const overwrite = options?.overwrite || false;
    fs.mkdir(path.dirname(dest), { recursive: true }, (err) => {
        if (err && (err as any).code !== 'EEXIST') return callback!(err);
        fs.rename(src, dest, (err) => {
            if (err) {
                if ((err as any).code === 'EXDEV') {
                    // Cross-device move: copy and unlink
                    fs.cp(src, dest, { recursive: true, force: overwrite }, (err) => {
                        if (err) return callback!(err);
                        fs.rm(src, { recursive: true, force: true }, callback!);
                    });
                    return;
                }
                callback!(err);
                return;
            }
            callback!(null);
        });

    });
};

export const moveSync = (src: string, dest: string, options?: any) => {
    const overwrite = options?.overwrite || false;
    fs.mkdirSync(path.dirname(dest), { recursive: true });
    try {
        fs.renameSync(src, dest);
    } catch (err: any) {
        if (err.code === 'EXDEV') {
            fs.cpSync(src, dest, { recursive: true, force: overwrite });
            fs.rmSync(src, { recursive: true, force: true });
        } else {
            throw err;
        }
    }
};

// --- Aliases for better readability or compatibility ---
export {
    ensureFile as createFile,
    ensureFileSync as createFileSync,
    move as rename,
    moveSync as renameSync
};
