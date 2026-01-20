import * as cp from "node:child_process";
import * as fs from "node:fs";

/**
 * Libc detection utilities.
 * Deobfuscated from xeA in chunk_207.ts.
 */

const LDD_PATH = "/usr/bin/ldd";
const GLIBC = "glibc";
const MUSL = "musl";

let cachedFamily: string | null = null;
let cachedVersion: string | null = null;

function isLinux(): boolean {
    return process.platform === "linux";
}

function getReport(): any {
    // @ts-ignore
    if (isLinux() && process.report) {
        // @ts-ignore
        const excludeNetwork = process.report.excludeNetwork;
        // @ts-ignore
        process.report.excludeNetwork = true;
        // @ts-ignore
        const report = process.report.getReport();
        // @ts-ignore
        process.report.excludeNetwork = excludeNetwork;
        return report;
    }
    return {};
}

function getConfOutput(): string {
    try {
        return cp.execSync("getconf GNU_LIBC_VERSION 2>&1 || true; ldd --version 2>&1 || true", {
            encoding: "utf8"
        });
    } catch (e) {
        return "";
    }
}

export function familySync(): string | null {
    if (!isLinux()) return null;
    if (cachedFamily) return cachedFamily;

    const report = getReport();
    if (report.header && report.header.glibcVersionRuntime) {
        return (cachedFamily = GLIBC);
    }

    if (Array.isArray(report.sharedObjects)) {
        if (report.sharedObjects.some((so: string) => so.includes("libc.musl-") || so.includes("ld-musl-"))) {
            return (cachedFamily = MUSL);
        }
    }

    try {
        const lddContent = fs.readFileSync(LDD_PATH, "utf8");
        if (lddContent.includes("musl")) return (cachedFamily = MUSL);
        if (lddContent.includes("GNU C Library")) return (cachedFamily = GLIBC);
    } catch (e) { }

    const output = getConfOutput();
    if (output.includes(GLIBC)) return (cachedFamily = GLIBC);
    if (output.includes(MUSL)) return (cachedFamily = MUSL);

    return null;
}

export function versionSync(): string | null {
    if (!isLinux()) return null;
    if (cachedVersion) return cachedVersion;

    const report = getReport();
    if (report.header && report.header.glibcVersionRuntime) {
        return (cachedVersion = report.header.glibcVersionRuntime);
    }

    try {
        const lddContent = fs.readFileSync(LDD_PATH, "utf8");
        const match = lddContent.match(/LIBC[a-z0-9 \-).]*?(\d+\.\d+)/i);
        if (match) return (cachedVersion = match[1]);
    } catch (e) { }

    const output = getConfOutput();
    const parts = output.trim().split(/\s+/);
    if (output.includes(GLIBC)) {
        return (cachedVersion = parts[1] || null);
    }

    return null;
}

export function isNonGlibcLinuxSync(): boolean {
    return isLinux() && familySync() !== GLIBC;
}
