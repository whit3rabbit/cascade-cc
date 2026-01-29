/**
 * File: src/utils/http/RetryStrategy.ts
 * Role: Implements HTTP client with retry logic (Exponential Backoff and Jitter).
 */

import axios, { AxiosRequestConfig, AxiosResponse, Method } from 'axios';

const MAX_RETRIES = 3;
const BASE_DELAY_MS = 1000;

/**
 * Delays execution for a specified duration.
 */
async function sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
}

/**
 * Executes an HTTP request with retry logic.
 */
async function requestWithRetry(method: Method, url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse> {
    let retries = 0;

    while (true) {
        try {
            return await axios({
                method,
                url,
                data,
                ...config
            });
        } catch (error: any) {
            const isRetryable = !error.response || (error.response.status >= 500) || (error.response.status === 429);

            if (retries >= MAX_RETRIES || !isRetryable) {
                throw error;
            }

            // Exponential backoff with jitter
            const delay = BASE_DELAY_MS * Math.pow(2, retries) + Math.random() * 100;
            console.warn(`[RetryStrategy] Request failed (status: ${error.response?.status}). Retrying in ${Math.round(delay)}ms... (${retries + 1}/${MAX_RETRIES})`);

            await sleep(delay);
            retries++;
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
