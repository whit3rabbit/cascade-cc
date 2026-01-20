
import React, { useMemo } from "react";
import { Box, Text } from "ink";
import { z } from "zod";
import axios from "axios";
import { LRUCache } from "lru-cache";
import {
    isDomainAllowed,
    isValidUrl,
    performFetch,
    createTurndownService,
    ALLOWED_DOMAINS
} from "../../services/fetch/TurndownService.js";
import { getModelClient } from "../../services/model/ModelClient.js";
import { formatBytes } from "../../utils/shared/formatUtils.js";

// Logic from chunk_476.ts

const FETCH_CACHE_SIZE = 50 * 1024 * 1024; // 50MB
const FETCH_TTL = 15 * 60 * 1000; // 15 mins
const CONTENT_TRUNCATION_LIMIT = 100000;

const fetchCache = new LRUCache<string, any>({
    maxSize: FETCH_CACHE_SIZE,
    sizeCalculation: (value) => Buffer.byteLength(value.content),
    ttl: FETCH_TTL,
});

export class DomainBlockedError extends Error {
    constructor(domain: string) {
        super(`Claude Code is unable to fetch from ${domain}`);
        this.name = "DomainBlockedError";
    }
}

export class DomainCheckFailedError extends Error {
    constructor(domain: string) {
        super(`Unable to verify if domain ${domain} is safe to fetch. This may be due to network restrictions or enterprise security policies blocking claude.ai.`);
        this.name = "DomainCheckFailedError";
    }
}

// Dummy checkDomain function - in original it calls an internal API
async function checkDomain(domain: string): Promise<{ status: string }> {
    // This is a placeholder for the Ci5 call
    return { status: "allowed" };
}

async function internalFetch(url: string, signal?: AbortSignal) {
    if (!isValidUrl(url)) throw new Error("Invalid URL");

    const cached = fetchCache.get(url);
    if (cached) return cached;

    let targetUrl = url;
    try {
        const parsed = new URL(url);
        // Preference for HTTPS
        if (parsed.protocol === "http:") {
            parsed.protocol = "https:";
            targetUrl = parsed.toString();
        }

        const hostname = parsed.hostname;
        // Preflight check
        const { status } = await checkDomain(hostname);
        if (status === "blocked") throw new DomainBlockedError(hostname);
        if (status === "check_failed") throw new DomainCheckFailedError(hostname);
    } catch (e) {
        if (e instanceof DomainBlockedError || e instanceof DomainCheckFailedError) throw e;
    }

    const response: any = await performFetch(targetUrl, signal);

    // If it's a redirect result from performFetch (manual handling)
    if (response.type === "redirect") return response;

    const rawContent = Buffer.from(response.data).toString("utf-8");
    const contentType = response.headers["content-type"] ?? "";
    const bytes = Buffer.byteLength(rawContent);

    let processedContent: string;
    if (contentType.includes("text/html")) {
        const td = createTurndownService();
        processedContent = td.turndown(rawContent);
    } else {
        processedContent = rawContent;
    }

    if (processedContent.length > CONTENT_TRUNCATION_LIMIT) {
        processedContent = processedContent.substring(0, CONTENT_TRUNCATION_LIMIT) + "...[content truncated]";
    }

    const result = {
        bytes,
        code: response.status,
        codeText: response.statusText,
        content: processedContent,
        contentType
    };

    fetchCache.set(url, result);
    return result;
}

// Rg2: Apply prompt to content
async function applyPromptToContent(prompt: string, content: string, signal?: AbortSignal, isNonInteractive: boolean = false) {
    // This uses the model to filter the content based on the prompt
    const client = getModelClient();
    const systemPrompt = "You are a web content processor. Filter and summarize the provided content based on the user's request.";
    const userPrompt = `Request: ${prompt}\n\nContent:\n${content}`;

    const response = await client.complete({
        messages: [{ role: "user", content: userPrompt }],
        system: systemPrompt,
        signal
    });

    return response.text ?? "No response from model";
}

export const FetchTool = {
    name: "WebFetch",
    userFacingName: "Fetch",
    description: "Fetches content from a URL and optionally processes it with a prompt.",

    inputSchema: z.object({
        url: z.string().url().describe("The URL to fetch content from"),
        prompt: z.string().describe("The prompt to run on the fetched content")
    }),

    async call({ url, prompt }: { url: string, prompt?: string }, { abortController, options }: any) {
        const startTime = Date.now();
        const fetchResult = await internalFetch(url, abortController?.signal);

        if (fetchResult.type === "redirect") {
            const redirectMsg = `REDIRECT DETECTED: The URL redirects to a different host.\n\nOriginal URL: ${fetchResult.originalUrl}\nRedirect URL: ${fetchResult.redirectUrl}\nStatus: ${fetchResult.statusCode}\n\nPlease use WebFetch again with the new URL.`;
            return {
                data: {
                    bytes: Buffer.byteLength(redirectMsg),
                    code: fetchResult.statusCode,
                    codeText: "Redirect",
                    result: redirectMsg,
                    durationMs: Date.now() - startTime,
                    url
                }
            };
        }

        let finalResult: string;
        if (prompt && fetchResult.content) {
            finalResult = await applyPromptToContent(
                prompt,
                fetchResult.content,
                abortController?.signal,
                options.isNonInteractiveSession
            );
        } else {
            finalResult = fetchResult.content;
        }

        return {
            data: {
                bytes: fetchResult.bytes,
                code: fetchResult.code,
                codeText: fetchResult.codeText,
                result: finalResult,
                durationMs: Date.now() - startTime,
                url
            }
        };
    },

    renderToolUseMessage({ url, prompt }: any) {
        return `Fetching ${url}${prompt ? ` with prompt: "${prompt}"` : ""}`;
    },

    renderToolResultMessage({ bytes, code, codeText, result }: any, { verbose }: any) {
        const size = formatBytes(bytes);
        return (
            <Box flexDirection="column">
                <Box height={1}>
                    <Text>Received <Text bold>{size}</Text> ({code} {codeText})</Text>
                </Box>
                {verbose && (
                    <Box flexDirection="column">
                        <Text>{result}</Text>
                    </Box>
                )}
            </Box>
        );
    }
};
