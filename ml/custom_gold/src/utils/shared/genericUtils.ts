
// Logic from chunk_58.ts (Generic Utilities & Crypto)

/**
 * Simple string hashing function (FNV-1a inspired).
 */
export function hashString(str: string): number {
    let hash = 2166136261;
    for (let i = 0; i < str.length; i++) {
        hash ^= str.charCodeAt(i);
        hash += (hash << 1) + (hash << 4) + (hash << 7) + (hash << 8) + (hash << 24);
    }
    return hash >>> 0;
}

/**
 * Salted random value generator for experimentation buckets.
 */
export function getDeterministicRandom(salt: string, input: string, mode: number): number | null {
    if (mode === 2) return (hashString(hashString(salt + input) + "") % 10000) / 10000;
    if (mode === 1) return (hashString(input + salt) % 1000) / 1000;
    return null;
}

/**
 * Decrypts data using AES-CBC.
 */
export async function decryptAesCbc(encryptedData: string, keyStr: string, cryptoImpl?: any): Promise<string> {
    const crypto = cryptoImpl || globalThis.crypto?.subtle;
    if (!crypto) throw new Error("No SubtleCrypto implementation found");

    const encoder = new TextEncoder();
    const key = await crypto.importKey(
        "raw",
        encoder.encode(keyStr),
        { name: "AES-CBC", length: 128 },
        true,
        ["encrypt", "decrypt"]
    );

    const [ivHex, dataHex] = encryptedData.split(".");
    const iv = Buffer.from(ivHex, "hex");
    const data = Buffer.from(dataHex, "hex");

    const decrypted = await crypto.decrypt(
        { name: "AES-CBC", iv },
        key,
        data
    );

    return new TextDecoder().decode(decrypted);
}

/**
 * Simple wildcard pattern matcher (_____ maps to .*).
 */
export function matchPattern(str: string, pattern: string, isPath: boolean = false): boolean {
    try {
        let regexStr = pattern.replace(/[*.+?^${}()|[\]\\]/g, "\\$&").replace(/_____/g, ".*");
        if (isPath) regexStr = "\\/?" + regexStr.replace(/(^\/|\/$)/g, "") + "\\/?";
        return new RegExp("^" + regexStr + "$", "i").test(str);
    } catch {
        return false;
    }
}

/**
 * Normalizes version strings for comparison.
 */
export function normalizeVersion(version: string | number): string {
    let v = String(version).replace(/(^v|\+.*$)/g, "").split(/[-.]/);
    if (v.length === 3) v.push("~");
    return v.map(p => p.match(/^[0-9]+$/) ? p.padStart(5, " ") : p).join("-");
}
