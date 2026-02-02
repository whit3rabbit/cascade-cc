/**
 * File: src/utils/http/StreamingResponse.ts
 * Role: Class for handling streaming responses from LLM APIs.
 */

export interface StreamingResponseResult {
    data: StreamingResponse;
    response: Response;
    request_id: string | null;
}

/**
 * Manages the state and control of a streaming response.
 */
export class StreamingResponse {
    public messages: any[] = [];
    public receivedMessages: any[] = [];
    public controller: AbortController = new AbortController();
    public clientOptions: any;

    private _responsePromise: Promise<Response>;
    private _resolveResponse!: (value: Response) => void;
    private _rejectResponse!: (reason: any) => void;

    private _donePromise: Promise<void>;
    private _resolveDone!: (value: void) => void;
    private _rejectDone!: (reason: any) => void;

    private _response: Response | null = null;

    constructor(clientOptions: any) {
        this.clientOptions = clientOptions;

        this._responsePromise = new Promise((resolve, reject) => {
            this._resolveResponse = resolve;
            this._rejectResponse = reject;
        });

        this._donePromise = new Promise((resolve, reject) => {
            this._resolveDone = resolve;
            this._rejectDone = reject;
        });

        // Suppress unhandled rejections as these are expected to be handled by withResponse() or elsewhere.
        this._responsePromise.catch(() => { });
        this._donePromise.catch(() => { });
    }

    /**
     * Returns the underlying Response object if resolved.
     */
    get response(): Response | null {
        return this._response;
    }

    /**
     * Waits for the initial Response object and returns it along with this context.
     */
    async withResponse(): Promise<StreamingResponseResult> {
        const response = await this._responsePromise;
        if (!response) {
            throw new Error("Could not resolve a `Response` object");
        }

        this._response = response;

        return {
            data: this,
            response,
            request_id: response.headers.get("request-id")
        };
    }

    /**
     * Helper to resolve the response promise from external callers (like providers).
     */
    setResponse(response: Response) {
        this._resolveResponse(response);
    }

    /**
     * Helper to signal completion.
     */
    done() {
        this._resolveDone();
    }

    /**
     * Helper to signal failure.
     */
    error(err: any) {
        this._rejectResponse(err);
        this._rejectDone(err);
    }

    /**
     * Creates a StreamingResponse from a Response with an SSE stream.
     * 
     * @param response - The response containing the SSE stream.
     * @param controller - An optional AbortController to signal completion or error.
     * @returns A new StreamingResponse instance.
     */
    static fromSSEResponse(response: Response, controller?: AbortController): StreamingResponse {
        const sr = new StreamingResponse({});
        sr.setResponse(response);

        if (!response.body) {
            sr.error(new Error("Response body is null"));
            return sr;
        }

        const stream = response.body;
        (async () => {
            const { parseSseEvents } = await import('./SseParser.js');
            try {
                for await (const event of parseSseEvents(stream)) {
                    if (event.data) {
                        try {
                            const parsed = JSON.parse(event.data);
                            sr.receivedMessages.push(parsed);
                            sr.messages.push(parsed);
                            // In a real implementation, we might emit events or update state here.
                        } catch (e) {
                            console.error("Failed to parse SSE data as JSON:", event.data);
                        }
                    }
                }
                sr.done();
            } catch (err) {
                sr.error(err);
            } finally {
                controller?.abort();
            }
        })();

        return sr;
    }

    /**
     * Creates a StreamingResponse from a ReadableStream (generic).
     * 
     * @param _stream - The stream to wrap.
     * @returns A new StreamingResponse instance.
     */
    static fromReadableStream(_stream: ReadableStream): StreamingResponse {
        return new StreamingResponse({});
    }
}
