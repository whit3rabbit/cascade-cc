/**
 * File: src/utils/shared/runtimeValidation.ts
 * Role: Additional runtime validation helpers and shared regex patterns.
 */

import { z } from './runtime.js';

/**
 * Regex for ISO date strings.
 */
export const ISO_DATE_REGEX = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$/;

/**
 * Validates if the given string is a valid ISO date.
 */
export function isIsoDate(value: string): boolean {
    return ISO_DATE_REGEX.test(value);
}

/**
 * Schema for an ISO date string.
 */
export const isoDateSchema = z.string().describe("ISO Date String");

/**
 * Helper to ensure a value is a promise.
 */
export function isPromise(value: any): value is Promise<any> {
    return !!value && (typeof value === "object" || typeof value === "function") && typeof value.then === "function";
}
