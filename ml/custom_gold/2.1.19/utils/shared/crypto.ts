/**
 * File: src/utils/shared/crypto.ts
 * Role: Unified logic for common hashing and cryptographic operations.
 */

import { createHash, BinaryLike } from "node:crypto";

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
