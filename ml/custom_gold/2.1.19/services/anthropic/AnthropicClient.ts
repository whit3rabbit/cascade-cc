/**
 * File: src/services/anthropic/AnthropicClient.ts
 * Role: Anthropic API Client for Files, Models, and Messages
 */

import { normalizeHeaders } from '../../utils/http/HeaderUtils.js';
import { path } from '../../utils/http/PathBuilder.js';

interface RequestOptions {
    headers?: Record<string, string>;
    signal?: AbortSignal;
    [key: string]: any;
}

interface ApiParams {
    betas?: string[];
    [key: string]: any;
}

export class AnthropicFilesAPI {
    private _client: any;

    constructor(client: any) {
        this._client = client;
    }

    list(params: ApiParams = {}, options: RequestOptions = {}) {
        const { betas, ...query } = params;
        return this._client.get("/v1/files", {
            query,
            ...options,
            headers: normalizeHeaders({
                "anthropic-beta": [...(betas ?? []), "files-api-2025-04-14"].join(","),
                ...(options?.headers ?? {})
            })
        });
    }

    delete(fileId: string, options: RequestOptions = {}) {
        const { betas } = options as any;
        return this._client.delete(path`/v1/files/${fileId}`, {
            headers: normalizeHeaders({
                "anthropic-beta": [...(betas ?? []), "files-api-2025-04-14"].join(","),
                ...(options?.headers ?? {})
            })
        });
    }

    download(fileId: string, options: RequestOptions = {}) {
        const { betas } = options as any;
        return this._client.get(path`/v1/files/${fileId}/content`, {
            headers: normalizeHeaders({
                "anthropic-beta": [...(betas ?? []), "files-api-2025-04-14"].join(","),
                "Accept": "application/binary",
                ...(options?.headers ?? {})
            }),
            __binaryResponse: true
        });
    }

    upload(params: ApiParams, options: RequestOptions = {}) {
        const { betas, ...body } = params;
        return this._client.post("/v1/files", {
            body,
            ...options,
            headers: normalizeHeaders({
                "anthropic-beta": [...(betas ?? []), "files-api-2025-04-14"].join(","),
                ...(options?.headers ?? {})
            })
        });
    }
}

export class AnthropicModelsAPI {
    private _client: any;

    constructor(client: any) {
        this._client = client;
    }

    retrieve(modelId: string, options: RequestOptions = {}) {
        const { betas } = options as any;
        return this._client.get(path`/v1/models/${modelId}?beta=true`, {
            headers: normalizeHeaders({
                ...(betas ? { "anthropic-beta": betas.join(",") } : {}),
                ...(options?.headers ?? {})
            })
        });
    }

    list(params: ApiParams = {}, options: RequestOptions = {}) {
        const { betas, ...query } = params;
        return this._client.get("/v1/models?beta=true", {
            query,
            ...options,
            headers: normalizeHeaders({
                ...(betas ? { "anthropic-beta": betas.join(",") } : {}),
                ...(options?.headers ?? {})
            })
        });
    }
}

export class AnthropicMessagesAPI {
    private _client: any;

    constructor(client: any) {
        this._client = client;
    }

    create(params: ApiParams, options: RequestOptions = {}) {
        const { betas, ...body } = params;
        return this._client.post("/v1/messages", {
            body,
            ...options,
            headers: normalizeHeaders({
                ...(betas ? { "anthropic-beta": betas.join(",") } : {}),
                ...(options?.headers ?? {})
            }),
            stream: params.stream ?? false
        });
    }

    countTokens(params: ApiParams, options: RequestOptions = {}) {
        const { betas, ...body } = params;
        return this._client.post("/v1/messages/count_tokens", {
            body,
            ...options,
            headers: normalizeHeaders({
                ...(betas ? { "anthropic-beta": betas.join(",") } : {}),
                ...(options?.headers ?? {})
            })
        });
    }
}

import { ProxyAgent, Dispatcher } from 'undici';

// Helper to get proxy dispatcher
function getDispatcher(): Dispatcher | undefined {
    const proxyUrl = process.env.HTTPS_PROXY || process.env.HTTP_PROXY;
    const noProxy = process.env.NO_PROXY;

    if (proxyUrl) {
        // Basic ProxyAgent usage. Check undici docs if more complex config needed.
        // undici ProxyAgent supports token/auth in url.
        // noProxy support might depend on undici version or manual check?
        // undici@6 ProxyAgent doesn't seem to take "no_proxy" list directly in constructor options universally?
        // BUT standard practice is often just passing the URL.
        // Let's assume standard behavior for now.
        return new ProxyAgent(proxyUrl);
    }
    return undefined;
}

// Helper for custom headers
function getCustomHeaders(): Record<string, string> {
    const customHeadersStr = process.env.ANTHROPIC_CUSTOM_HEADERS;
    if (!customHeadersStr) return {};

    return customHeadersStr.split('\n').reduce((acc, line) => {
        const parts = line.split(':');
        if (parts.length >= 2) {
            const key = parts[0].trim();
            const val = parts.slice(1).join(':').trim();
            if (key && val) acc[key] = val;
        }
        return acc;
    }, {} as Record<string, string>);
}

const dispatcher = getDispatcher();

export class Anthropic {
    public files: AnthropicFilesAPI;
    public models: AnthropicModelsAPI;
    public messages: AnthropicMessagesAPI;
    private _client: any;

    constructor(options: { apiKey?: string; accessToken?: string; baseUrl?: string } = {}) {
        // Simple client wrapper around fetch
        this._client = {
            post: async (url: string, data: any) => {
                const { getAuthHeaders } = await import('../auth/AuthService.js');
                const { EnvService } = await import('../config/EnvService.js');
                const authHeaders = await getAuthHeaders();

                // Handle Disable Prompt Caching
                if (url.includes('/messages')) {
                    const disableGlobal = EnvService.get('DISABLE_PROMPT_CACHING');
                    const model = data.body?.model || '';
                    const disableHaiku = EnvService.get('DISABLE_PROMPT_CACHING_HAIKU') && model.includes('haiku');
                    const disableSonnet = EnvService.get('DISABLE_PROMPT_CACHING_SONNET') && model.includes('sonnet');
                    const disableOpus = EnvService.get('DISABLE_PROMPT_CACHING_OPUS') && model.includes('opus');

                    if (disableGlobal || disableHaiku || disableSonnet || disableOpus) {
                        // Strip caching headers/params if possible, or just don't add them. 
                        // Since we are passing 'data.body' directly, we might need to mutate it.
                        // Assuming current impl relies on 'cache_control' appearing in messages.
                        if (data.body && Array.isArray(data.body.messages)) {
                            // Deep clone to avoid mutating original if needed, but for now duplicate
                            data.body.messages = data.body.messages.map((msg: any) => {
                                if (Array.isArray(msg.content)) {
                                    return {
                                        ...msg,
                                        content: msg.content.map((block: any) => {
                                            const { cache_control, ...rest } = block;
                                            return rest;
                                        })
                                    };
                                }
                                return msg;
                            });
                        }
                        // Also strip top-level system if it has cache_control (new API)
                        if (data.body && Array.isArray(data.body.system)) {
                            data.body.system = data.body.system.map((block: any) => {
                                const { cache_control, ...rest } = block;
                                return rest;
                            });
                        }
                    }
                }

                const headers = {
                    ...authHeaders,
                    ...getCustomHeaders(),
                    ...data.headers
                };
                const fetchOptions: any = {
                    method: 'POST',
                    headers,
                    body: JSON.stringify(data.body),
                    dispatcher
                };
                if (data.signal) fetchOptions.signal = data.signal;

                const response = await fetch((options.baseUrl || 'https://api.anthropic.com') + url, fetchOptions);
                if (!response.ok) {
                    const error = await response.json().catch(() => ({}));
                    throw new Error(error.error?.message || `API error: ${response.status}`);
                }

                if (data.stream) {
                    return response; // Return full response for streaming
                }
                return { data: await response.json() };
            },
            get: async (url: string, data: any) => {
                const { getAuthHeaders } = await import('../auth/AuthService.js');
                const authHeaders = await getAuthHeaders();
                const headers = {
                    ...authHeaders,
                    ...getCustomHeaders(),
                    ...data.headers
                };
                const fetchOptions: any = {
                    headers,
                    dispatcher
                };

                const response = await fetch((options.baseUrl || 'https://api.anthropic.com') + url + (data.query ? '?' + new URLSearchParams(data.query).toString() : ''), fetchOptions);
                return { data: await response.json() };
            },
            delete: async (url: string, data: any) => {
                const { getAuthHeaders } = await import('../auth/AuthService.js');
                const authHeaders = await getAuthHeaders();
                const headers = {
                    ...authHeaders,
                    ...getCustomHeaders(),
                    ...data.headers
                };
                const fetchOptions: any = {
                    method: 'DELETE',
                    headers,
                    dispatcher
                };
                const response = await fetch((options.baseUrl || 'https://api.anthropic.com') + url, fetchOptions);
                return { data: await response.json() };
            }
        };

        this.files = new AnthropicFilesAPI(this._client);
        this.models = new AnthropicModelsAPI(this._client);
        this.messages = new AnthropicMessagesAPI(this._client);
    }
}
