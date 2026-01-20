import { EventEmitter } from "node:events";

/**
 * Event-driven wrapper for Anthropic Messages streaming API.
 * Deobfuscated from _0A in chunk_226.ts.
 */
export class AnthropicStream extends EventEmitter {
    messages: any[] = [];
    receivedMessages: any[] = [];
    controller: AbortController = new AbortController();
    private _messageInProgress: any = null;
    private _responsePromise: Promise<any>;
    private _resolveResponse!: (val: any) => void;
    private _rejectResponse!: (err: any) => void;
    private _donePromise: Promise<void>;
    private _resolveDone!: () => void;
    private _rejectDone!: (err: any) => void;
    private _ended: boolean = false;
    private _errored: boolean = false;
    private _aborted: boolean = false;
    private _requestParams: any;

    constructor(params: any) {
        super();
        this._requestParams = params;

        this._responsePromise = new Promise((resolve, reject) => {
            this._resolveResponse = resolve;
            this._rejectResponse = reject;
        });

        this._donePromise = new Promise((resolve, reject) => {
            this._resolveDone = resolve;
            this._rejectDone = reject;
        });

        // Suppress unhandled rejections for internal promises
        this._responsePromise.catch(() => { });
        this._donePromise.catch(() => { });
    }

    get response() {
        return this._responsePromise;
    }

    async withResponse() {
        const resp = await this._responsePromise;
        return {
            data: this,
            response: resp,
            request_id: resp.headers.get("request-id")
        };
    }

    static createMessage(client: any, params: any, options?: any) {
        const stream = new AnthropicStream(params);
        // Add existing messages to history
        for (const msg of params.messages) {
            stream.messages.push(msg);
        }

        const streamParams = { ...params, stream: true };
        const streamOptions = {
            ...options,
            headers: {
                ...options?.headers,
                "X-Stainless-Helper-Method": "stream"
            }
        };

        stream._run(async () => {
            const signal = options?.signal;
            let onAbort;
            if (signal) {
                if (signal.aborted) stream.abort();
                onAbort = () => stream.abort();
                signal.addEventListener("abort", onAbort);
            }

            try {
                const { response, data } = await client.messages.create(streamParams, {
                    ...streamOptions,
                    signal: stream.controller.signal
                }).withResponse();

                stream._connected(response);

                for await (const event of data) {
                    stream._handleEvent(event);
                }
            } finally {
                if (signal && onAbort) signal.removeEventListener("abort", onAbort);
            }
        });

        return stream;
    }

    private _run(fn: () => Promise<void>) {
        fn().then(() => {
            this._emitFinal();
            this.emit("end");
            this._resolveDone();
        }).catch((err) => {
            this._handleError(err);
        });
    }

    private _connected(response: any) {
        if (this._ended) return;
        this._resolveResponse(response);
        this.emit("connect");
    }

    private _handleEvent(event: any) {
        if (this._ended) return;

        this.emit("streamEvent", event);

        switch (event.type) {
            case "message_start":
                this._messageInProgress = event.message;
                break;
            case "content_block_start":
                this._messageInProgress.content.push(event.content_block);
                break;
            case "content_block_delta":
                this._handleDelta(event);
                break;
            case "content_block_stop":
                this.emit("contentBlock", this._messageInProgress.content[event.index]);
                break;
            case "message_delta":
                this._handleMessageDelta(event);
                break;
            case "message_stop":
                this.receivedMessages.push(this._messageInProgress);
                this.emit("message", this._messageInProgress);
                break;
        }
    }

    private _handleDelta(event: any) {
        const block = this._messageInProgress.content[event.index];
        const delta = event.delta;

        switch (delta.type) {
            case "text_delta":
                if (block.type === "text") {
                    const prev = block.text || "";
                    block.text = prev + delta.text;
                    this.emit("text", delta.text, block.text);
                }
                break;
            case "thinking_delta":
                if (block.type === "thinking") {
                    block.thinking = (block.thinking || "") + delta.thinking;
                    this.emit("thinking", delta.thinking, block.thinking);
                }
                break;
            case "signature_delta":
                if (block.type === "thinking") {
                    block.signature = delta.signature;
                    this.emit("signature", delta.signature);
                }
                break;
            case "input_json_delta":
                if (block.type === "tool_use") {
                    block.partial_json = (block.partial_json || "") + delta.partial_json;
                    this.emit("inputJson", delta.partial_json, block.partial_json);
                }
                break;
        }
    }

    private _handleMessageDelta(event: any) {
        const delta = event.delta;
        if (delta.stop_reason) this._messageInProgress.stop_reason = delta.stop_reason;
        if (delta.stop_sequence) this._messageInProgress.stop_sequence = delta.stop_sequence;
        if (event.usage) {
            this._messageInProgress.usage.output_tokens = event.usage.output_tokens;
        }
    }

    private _handleError(err: any) {
        this._errored = true;
        this._ended = true;
        this._rejectResponse(err);
        this._rejectDone(err);
        this.emit("error", err);
    }

    private _emitFinal() {
        const last = this.receivedMessages[this.receivedMessages.length - 1];
        if (last) {
            this.emit("finalMessage", last);
        }
    }

    abort() {
        this._aborted = true;
        this.controller.abort();
    }

    async done() {
        return this._donePromise;
    }

    async finalMessage() {
        await this.done();
        return this.receivedMessages[this.receivedMessages.length - 1];
    }

    async finalText() {
        const msg = await this.finalMessage();
        return msg.content
            .filter((c: any) => c.type === "text")
            .map((c: any) => c.text)
            .join(" ");
    }
}

/**
 * High-level streaming API call generator.
 * Deobfuscated from zHA in chunk_580.ts.
 */
export async function* streamAnthropic(params: any): AsyncGenerator<any> {
    // This generator wraps the core executeAnthropicQuery logic (SW9 in chunk_580)
    // For now, we utilize the AnthropicStream class or implement the generator logic directly
    const { messages, systemPrompt, maxThinkingTokens, tools, signal, options } = params;

    // In a full implementation, this would call executeAnthropicQuery(messages, systemPrompt, ...)
    // which yields stream events.

    // Stubbed implementation using AnthropicStream for now
    const stream = new AnthropicStream(params);
    // ... (logic to yield events)
    yield { type: "stream_request_start" };
    // yield* executeAnthropicQuery(...)
}

/**
 * High-level non-streaming API call.
 * Deobfuscated from Cd in chunk_580.ts.
 */
export async function callAnthropic(params: any): Promise<any> {
    const stream = streamAnthropic(params);
    let lastAssistantMessage = null;
    for await (const event of stream) {
        if (event.type === "assistant") lastAssistantMessage = event;
    }
    if (!lastAssistantMessage) throw new Error("No assistant message found");
    return lastAssistantMessage;
}
