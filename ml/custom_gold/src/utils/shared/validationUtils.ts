const BASE64_RGX = /^[a-zA-Z0-9+/]*={0,2}$/;

function checkBase64(str: string): boolean {
    if (str === "") return true;
    if (str.length % 4 !== 0) return false;
    try {
        atob(str);
        return true;
    } catch {
        return false;
    }
}

export function isValidBase64(str: string): boolean {
    if (!BASE64_RGX.test(str)) return false;
    const normalized = str.replace(/[-_]/g, (m) => m === "-" ? "+" : "/");
    const padded = normalized.padEnd(Math.ceil(normalized.length / 4) * 4, "=");
    return checkBase64(padded);
}

export function isValidJWT(token: string, algorithm: string | null = null): boolean {
    try {
        const parts = token.split(".");
        if (parts.length !== 3) return false;
        const [header] = parts;
        if (!header) return false;

        const decodedHeader = JSON.parse(atob(header));
        if ("typ" in decodedHeader && decodedHeader?.typ !== "JWT") return false;
        if (!decodedHeader.alg) return false;
        if (algorithm && (!("alg" in decodedHeader) || decodedHeader.alg !== algorithm)) return false;

        return true;
    } catch {
        return false;
    }
}
