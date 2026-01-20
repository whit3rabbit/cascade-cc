
export async function generatePkceChallenge(length = 43): Promise<{ code_verifier: string; code_challenge: string }> {
    if (length < 43 || length > 128) {
        throw new Error(`Expected a length between 43 and 128. Received ${length}.`);
    }

    const verifier = await generateRandomString(length);
    const challenge = await generateCodeChallenge(verifier);

    return {
        code_verifier: verifier,
        code_challenge: challenge
    };
}

async function generateRandomString(length: number): Promise<string> {
    const buffer = new Uint8Array(length);
    // Using Web Crypto API if available, or Node's crypto
    const crypto = globalThis.crypto ?? (await import("node:crypto")).webcrypto;
    crypto.getRandomValues(buffer);

    let result = "";
    const chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~";
    for (let i = 0; i < length; i++) {
        result += chars[buffer[i] % chars.length];
    }
    return result;
}

async function generateCodeChallenge(verifier: string): Promise<string> {
    const encoder = new TextEncoder();
    const data = encoder.encode(verifier);

    const crypto = globalThis.crypto ?? (await import("node:crypto")).webcrypto;
    const hashBuffer = await crypto.subtle.digest("SHA-256", data);

    // Convert buffer to Base64 URL encoded string
    const hashArray = Array.from(new Uint8Array(hashBuffer));
    const hashString = String.fromCharCode(...hashArray);
    const base64 = btoa(hashString);

    return base64.replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
