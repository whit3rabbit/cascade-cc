
import { dirname, join, parse as parsePath, sep } from 'path';

/**
 * Common utilities for file path manipulation and git detection.
 */

export function getParentDirectories(files: string[]): string[] {
    const parentDirs = new Set<string>();
    files.forEach((file) => {
        let dir = dirname(file);
        while (dir !== '.' && dir !== parsePath(dir).root) {
            parentDirs.add(dir + sep);
            dir = dirname(dir);
        }
    });
    return Array.from(parentDirs);
}

export function getCommonPrefix(a: string, b: string): string {
    const minLength = Math.min(a.length, b.length);
    let i = 0;
    while (i < minLength && a[i] === b[i]) {
        i++;
    }
    return a.substring(0, i);
}

export function getLongestCommonPrefix(strings: string[]): string {
    if (strings.length === 0) return "";
    let prefix = strings[0];
    for (let i = 1; i < strings.length; i++) {
        prefix = getCommonPrefix(prefix, strings[i]);
        if (prefix === "") return "";
    }
    return prefix;
}
