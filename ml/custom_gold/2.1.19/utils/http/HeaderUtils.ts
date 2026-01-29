/**
 * File: src/utils/http/HeaderUtils.ts
 * Role: Utilities for manipulating HTTP headers, including support for nullable values.
 */

export const PRIVATE_NULLABLE_HEADERS = Symbol.for("brand.privateNullableHeaders");

export interface NullableHeaders {
    [PRIVATE_NULLABLE_HEADERS]: boolean;
    values: Headers;
    nulls: Set<string>;
}

export type HeadersInput = Headers | [string, string | null][] | Record<string, string | string[] | null | undefined> | NullableHeaders;

/**
 * Generator function to iterate through header entries reliably.
 */
export function* iterateHeaders(headers: HeadersInput): Generator<[string, string | null]> {
    if (!headers) {
        return;
    }

    if (typeof headers === 'object' && PRIVATE_NULLABLE_HEADERS in headers) {
        const nullableHeaders = headers as NullableHeaders;
        for (const [key, value] of nullableHeaders.values.entries()) {
            yield [key, value];
        }
        for (const nullKey of nullableHeaders.nulls) {
            yield [nullKey, null];
        }
        return;
    }

    if (headers instanceof Headers) {
        yield* headers.entries() as IterableIterator<[string, string]>;
    } else if (Array.isArray(headers)) {
        yield* headers;
    } else {
        for (const [key, value] of Object.entries(headers)) {
            const vals = Array.isArray(value) ? value : [value];
            for (const val of vals) {
                if (val === undefined) {
                    continue;
                }
                yield [key, val as string | null];
            }
        }
    }
}

/**
 * Normalizes headers, handling potential null values and case-insensitivity.
 */
export function normalizeHeaders(headersInput: HeadersInput): NullableHeaders {
    const normalizedHeaders = new Headers();
    const nullHeaderKeys = new Set<string>();

    for (const [key, value] of iterateHeaders(headersInput)) {
        const lowerKey = key.toLowerCase();

        // Default behavior is to replace, so we delete any existing same-case key first
        if (!nullHeaderKeys.has(lowerKey)) {
            normalizedHeaders.delete(key);
            nullHeaderKeys.add(lowerKey);
        }

        if (value === null) {
            normalizedHeaders.delete(key);
            nullHeaderKeys.add(lowerKey);
        } else {
            normalizedHeaders.append(key, value);
            nullHeaderKeys.delete(lowerKey);
        }
    }

    return {
        [PRIVATE_NULLABLE_HEADERS]: true,
        values: normalizedHeaders,
        nulls: nullHeaderKeys,
    };
}

/**
 * Safely encodes a header value.
 */
export function encodeHeaderValue(value: string): string {
    return value.replace(/[^A-Za-z0-9\-._~!$&'()*+,;=:@]+/g, encodeURIComponent);
}
