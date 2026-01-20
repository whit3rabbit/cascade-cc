
import { getAuthHeaders, getCliUserAgent, getInternalUserAgent } from "./anthropicApiClient.js";
import { getActiveModel, getModelPricing, calculateUsageCost } from "./claudeUtils.js";
import { log } from "../logger/loggerService.js";
import { randomUUID } from "node:crypto";
import axios from "axios";

const logger = log("claudeApi");

export interface Message {
    role: "user" | "assistant";
    content: any;
}

export interface AnthropicRequest {
    model: string;
    messages: Message[];
    system?: string | any[];
    max_tokens?: number;
    tools?: any[];
    tool_choice?: any;
    stream?: boolean;
    temperature?: number;
    metadata?: any;
}

/**
 * Calls the Anthropic API in a one-shot manner. (Cd in chunk_580)
 */
let lastRequest: AnthropicRequest | null = null;

export function getLastApiRequest(): AnthropicRequest | null {
    return lastRequest;
}

/**
 * Calls the Anthropic API in a one-shot manner. (Cd in chunk_580)
 */
export async function callClaude(params: any): Promise<any> {
    const { messages, systemPrompt, tools, model, max_tokens, temperature, signal } = params;

    const { headers, error } = getAuthHeaders();
    if (error) throw new Error(error);

    const data: AnthropicRequest = {
        model: model || getActiveModel(),
        messages,
        system: systemPrompt,
        max_tokens: max_tokens || 4096,
        tools: tools,
        stream: false,
        temperature: temperature ?? 1
    };

    lastRequest = data;

    try {
        const response = await axios.post("https://api.anthropic.com/v1/messages", data, {
            headers: {
                ...headers,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                "User-Agent": getCliUserAgent()
            },
            signal
        });
        return response.data;
    } catch (err: any) {
        logger.error(`API call failed: ${err.message}`);
        if (err.response) {
            logger.error(`Response data: ${JSON.stringify(err.response.data)}`);
        }
        throw err;
    }
}

/**
 * Calls the Anthropic API with streaming. (zHA in chunk_580)
 */
export async function* getStreamingContent(params: any): AsyncGenerator<any> {
    const { messages, systemPrompt, tools, model, max_tokens, temperature, signal } = params;

    const { headers, error } = getAuthHeaders();
    if (error) throw new Error(error);

    const data: AnthropicRequest = {
        model: model || getActiveModel(),
        messages,
        system: systemPrompt,
        max_tokens: max_tokens || 4096,
        tools: tools,
        stream: true,
        temperature: temperature ?? 1
    };

    lastRequest = data;

    try {
        const response = await fetch("https://api.anthropic.com/v1/messages", {
            method: "POST",
            headers: {
                ...headers,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
                "User-Agent": getCliUserAgent()
            },
            body: JSON.stringify(data),
            signal
        });

        if (!response.ok) {
            const errBody = await response.text();
            throw new Error(`API error: ${response.status} ${errBody}`);
        }

        const reader = response.body?.getReader();
        if (!reader) throw new Error("No response body");

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
                if (line.startsWith("data: ")) {
                    const jsonStr = line.substring(6);
                    if (jsonStr === "[DONE]") break;
                    try {
                        const event = JSON.parse(jsonStr);
                        yield event;
                    } catch (err) {
                        logger.error(`Failed to parse stream event: ${err}`);
                    }
                }
            }
        }
    } catch (err: any) {
        logger.error(`Streaming failed: ${err.message}`);
        throw err;
    }
}
