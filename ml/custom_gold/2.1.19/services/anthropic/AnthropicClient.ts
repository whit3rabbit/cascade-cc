/**
 * File: src/services/anthropic/AnthropicClient.ts
 * Role: Anthropic API Client for Files and Models
 */

import { normalizeHeaders } from '../../utils/http/HeaderUtils.js';
import { path } from '../../utils/http/PathBuilder.js';

interface RequestOptions {
    headers?: Record<string, string>;
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
