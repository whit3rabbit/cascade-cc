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
     * Creates a StreamingResponse from a ReadableStream.
     * 
     * @param _stream - The stream to wrap.
     * @returns A new StreamingResponse instance.
     */
    static fromReadableStream(_stream: ReadableStream): StreamingResponse {
        // Placeholder for real stream parsing logic if needed.
        // In a full implementation, this would handle the chunked SSE parsing.
        return new StreamingResponse({});
    }
}
