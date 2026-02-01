/**
 * File: src/utils/shared/crypto.ts
 * Role: Unified logic for common hashing and cryptographic operations.
 */

import { createHash, randomBytes, BinaryLike } from "node:crypto";

/**
 * Computes a SHA-256 hash of the provided data and returns the raw digest Buffer.
 * 
 * @param data - The data to hash (string or Buffer).
 * @returns {Buffer} The binary hash digest.
 */
export function sha256(data: BinaryLike): Buffer {
    return createHash('sha256').update(data).digest();
}

/**
 * Computes a SHA-256 hash of the provided data and returns the hex-encoded string.
 * 
 * @param data - The data to hash.
 * @returns {string} The hex-encoded hash string.
 */
export function sha256Hex(data: BinaryLike): string {
    return createHash('sha256').update(data).digest('hex');
}

/**
 * Generates a cryptographically strong random string.
 * 
 * @param length - The length of the string to generate.
 * @returns {string} The generated random string (base64url encoded).
 */
export function generateRandomString(length: number): string {
    return randomBytes(length).toString('base64url').slice(0, length);
}

/**
 * Generates a PKCE code challenge from a code verifier.
 * 
 * @param codeVerifier - The code verifier string.
 * @returns {string} The code challenge (base64url encoded SHA-256 hash).
 */
export function pkceChallenge(codeVerifier: string): string {
    const hash = sha256(codeVerifier);
    return hash.toString('base64')
        .replace(/\+/g, '-')
        .replace(/\//g, '_')
        .replace(/=/g, '');
}
