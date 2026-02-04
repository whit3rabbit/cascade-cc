/**
 * File: src/utils/http/RetryStrategy.ts
 * Role: Implements HTTP client with retry logic (Exponential Backoff and Jitter).
 */

import axios, { AxiosRequestConfig, AxiosResponse, Method } from 'axios';

import { randomUUID } from 'node:crypto';

const MAX_RETRIES = 3;
const BASE_DELAY_MS = 1000;

/**
 * Delays execution for a specified duration.
 */
async function sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Parses the Retry-After header.
 * Returns valid delay in ms or undefined.
 */
function parseRetryAfter(header: string | number | undefined): number | undefined {
    if (!header) return undefined;
    if (typeof header === 'number') return header * 1000;

    // Try seconds
    const seconds = parseInt(header, 10);
    if (!isNaN(seconds)) return seconds * 1000;

    // Try HTTP Date
    const date = Date.parse(header);
    if (!isNaN(date)) {
        const delta = date - Date.now();
        return delta > 0 ? delta : 0;
    }
    return undefined;
}

/**
 * Calculates retry delay using exponential backoff with jitter.
 * Formula: ((Math.pow(multiplier, attempt) - 1) / 2 * 1000)
 * We treat 'multiplier' as 2 usually, but let's stick to the simpler interpretation of the prompt's request.
 */
function calculateRetryDelay(attempt: number): number {
    const multiplier = 2; // Common default
    // Using the specific formula requested: ((Math.pow(multiplier, attempt) - 1) / 2 * 1000)
    // Note: 'attempt' here starts at 0 for the first retry? Usually attempt 1 is first retry.
    // Let's assume attempt 1, 2, 3...
    const base = (Math.pow(multiplier, attempt) - 1) / 2 * 1000;
    // Add jitter (full jitter pattern often used by AWS, or simple random)
    return base + (Math.random() * 1000);
}

/**
 * Executes an HTTP request with retry logic.
 */
async function requestWithRetry(method: Method, url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse> {
    let attempt = 0;
    const invocationId = randomUUID(); // X-Amz-Invocation-Id / idempotency token

    while (true) {
        attempt++;
        try {
            return await axios({
                method,
                url,
                data,
                ...config,
                headers: {
                    ...config?.headers,
                    'X-Amz-Invocation-Id': invocationId,
                    'X-Request-Attempt': String(attempt)
                }
            });
        } catch (error: any) {
            const status = error.response?.status;
            const isRetryable = !error.response || (status >= 500) || (status === 429);

            if (attempt > MAX_RETRIES || !isRetryable) {
                throw error;
            }

            // Check for Retry-After header
            const retryAfterHeader = error.response?.headers?.['retry-after'];
            let delay = parseRetryAfter(retryAfterHeader);

            if (delay === undefined) {
                delay = calculateRetryDelay(attempt);
            }

            console.warn(`[RetryStrategy] Request failed (status: ${status}). Retrying in ${Math.round(delay)}ms... (${attempt}/${MAX_RETRIES})`);

            await sleep(delay);
        }
    }
}

export const get = (url: string, config?: AxiosRequestConfig) => requestWithRetry('get', url, null, config);
export const post = (url: string, data?: any, config?: AxiosRequestConfig) => requestWithRetry('post', url, data, config);
export const put = (url: string, data?: any, config?: AxiosRequestConfig) => requestWithRetry('put', url, data, config);
export const del = (url: string, config?: AxiosRequestConfig) => requestWithRetry('delete', url, null, config);
export const patch = (url: string, data?: any, config?: AxiosRequestConfig) => requestWithRetry('patch', url, data, config);

export default {
    get,
    post,
    put,
    del,
    delete: del,
    patch
};
