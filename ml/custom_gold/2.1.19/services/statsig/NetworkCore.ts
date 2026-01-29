/**
 * File: src/services/statsig/NetworkCore.ts
 * Role: Handles low-level network requests for Statsig, including retries and compression.
 */

// import { Log } from '../statsig/StatsigService.js'; // Assuming this exists or will be converted

export interface NetworkOptions {
    networkTimeoutMilliseconds?: number;
    apiEndpoint?: string;
    [key: string]: any;
}

export interface NetworkCoreOptions {
    networkOptions?: NetworkOptions;
    [key: string]: any;
}

export interface PostRequestArgs {
    endpoint: string;
    body: any;
    sdkKey: string;
    headers?: Record<string, string>;
}

export interface PostResponse {
    body: string;
    code: number;
}

export class NetworkCore {
    private options: NetworkCoreOptions;
    private networkConfig: NetworkOptions;
    private timeout: number;

    constructor(options: NetworkCoreOptions = {}) {
        this.options = options;
        this.networkConfig = options.networkOptions || {};
        this.timeout = this.networkConfig.networkTimeoutMilliseconds || 10000;
    }

    /**
     * Sends a POST request to Statsig.
     */
    async post(requestArgs: PostRequestArgs): Promise<PostResponse | null> {
        const { endpoint, body, sdkKey } = requestArgs;
        const url = `${this.networkConfig.apiEndpoint || "https://statsig.anthropic.com/v1/"}${endpoint}`;

        try {
            const response = await fetch(url, {
                method: "POST",
                body: JSON.stringify(body),
                headers: {
                    "Content-Type": "application/json",
                    "statsig-api-key": sdkKey,
                    ...requestArgs.headers
                },
                signal: AbortSignal.timeout(this.timeout)
            });

            if (!response.ok) {
                throw new Error(`Statsig request failed: ${response.statusText}`);
            }

            return {
                body: await response.text(),
                code: response.status
            };
        } catch (error: any) {
            console.error(`Network error during Statsig POST to ${endpoint}:`, error);
            return null;
        }
    }
}
