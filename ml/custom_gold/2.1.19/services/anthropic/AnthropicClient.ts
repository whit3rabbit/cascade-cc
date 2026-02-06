/**
 * File: src/services/anthropic/AnthropicClient.ts
 * Role: Anthropic API Client for Files, Models, and Messages
 */

import { normalizeHeaders } from '../../utils/http/HeaderUtils.js';
import { path } from '../../utils/http/PathBuilder.js';
import { BugReportService } from '../bugreport/BugReportService.js';

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

    if (proxyUrl) {
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

function getUserAgent(): string {
    const version = "2.1.19";
    const entrypoint = process.env.CLAUDE_CODE_ENTRYPOINT || "cli";
    const agentSdk = process.env.CLAUDE_AGENT_SDK_VERSION ? `, agent-sdk/${process.env.CLAUDE_AGENT_SDK_VERSION}` : "";
    return `claude-cli/${version} (external, ${entrypoint}${agentSdk})`;
}

function getBillingHeader(): string {
    const version = "2.1.19";
    const entrypoint = process.env.CLAUDE_CODE_ENTRYPOINT || "cli";
    const sessionType = "session"; // or turn? original uses turning point
    return `cc_version=${version}.${sessionType}; cc_entrypoint=${entrypoint}`;
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
                const { iterateHeaders } = await import('../../utils/http/HeaderUtils.js');
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
                        if (data.body && Array.isArray(data.body.messages)) {
                            data.body.messages = data.body.messages.map((msg: any) => {
                                if (Array.isArray(msg.content)) {
                                    return {
                                        ...msg,
                                        content: msg.content.map((block: any) => {
                                            const { cache_control: _cache_control, ...rest } = block;
                                            return rest;
                                        })
                                    };
                                }
                                return msg;
                            });
                        }
                        if (data.body && Array.isArray(data.body.system)) {
                            data.body.system = data.body.system.map((block: any) => {
                                const { cache_control: _cache_control, ...rest } = block;
                                return rest;
                            });
                        }
                    }
                }

                const finalHeaders: Record<string, string> = {
                    "User-Agent": getUserAgent(),
                    "x-anthropic-billing-header": getBillingHeader(),
                    "anthropic-version": "2023-06-01"
                };
                const sources = [authHeaders, getCustomHeaders(), data.headers];
                for (const source of sources) {
                    if (!source) continue;
                    for (const [key, value] of iterateHeaders(source)) {
                        if (value === null) {
                            delete finalHeaders[key.toLowerCase()];
                        } else {
                            finalHeaders[key.toLowerCase()] = value;
                        }
                    }
                }

                const fetchOptions: any = {
                    method: 'POST',
                    headers: finalHeaders,
                    body: JSON.stringify(data.body),
                    dispatcher
                };
                if (data.signal) fetchOptions.signal = data.signal;

                BugReportService.setLastApiRequest({
                    url: (options.baseUrl || 'https://api.anthropic.com') + url,
                    method: 'POST',
                    headers: finalHeaders,
                    body: data.body
                });

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
                const { iterateHeaders } = await import('../../utils/http/HeaderUtils.js');
                const { getAuthHeaders } = await import('../auth/AuthService.js');
                const authHeaders = await getAuthHeaders();

                const finalHeaders: Record<string, string> = {
                    "User-Agent": getUserAgent(),
                    "x-anthropic-billing-header": getBillingHeader(),
                    "anthropic-version": "2023-06-01"
                };
                const sources = [authHeaders, getCustomHeaders(), data.headers];
                for (const source of sources) {
                    if (!source) continue;
                    for (const [key, value] of iterateHeaders(source)) {
                        if (value === null) {
                            delete finalHeaders[key.toLowerCase()];
                        } else {
                            finalHeaders[key.toLowerCase()] = value;
                        }
                    }
                }

                const fetchOptions: any = {
                    headers: finalHeaders,
                    dispatcher
                };

                BugReportService.setLastApiRequest({
                    url: (options.baseUrl || 'https://api.anthropic.com') + url + (data.query ? '?' + new URLSearchParams(data.query).toString() : ''),
                    method: 'GET',
                    headers: finalHeaders
                });

                const response = await fetch((options.baseUrl || 'https://api.anthropic.com') + url + (data.query ? '?' + new URLSearchParams(data.query).toString() : ''), fetchOptions);
                return { data: await response.json() };
            },
            delete: async (url: string, data: any) => {
                const { iterateHeaders } = await import('../../utils/http/HeaderUtils.js');
                const { getAuthHeaders } = await import('../auth/AuthService.js');
                const authHeaders = await getAuthHeaders();

                const finalHeaders: Record<string, string> = {
                    "User-Agent": getUserAgent(),
                    "x-anthropic-billing-header": getBillingHeader(),
                    "anthropic-version": "2023-06-01"
                };
                const sources = [authHeaders, getCustomHeaders(), data.headers];
                for (const source of sources) {
                    if (!source) continue;
                    for (const [key, value] of iterateHeaders(source)) {
                        if (value === null) {
                            delete finalHeaders[key.toLowerCase()];
                        } else {
                            finalHeaders[key.toLowerCase()] = value;
                        }
                    }
                }

                const fetchOptions: any = {
                    method: 'DELETE',
                    headers: finalHeaders,
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
