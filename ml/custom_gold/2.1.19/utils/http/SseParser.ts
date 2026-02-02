/**
 * File: src/utils/http/SseParser.ts
 * Role: Parses Server-Sent Events (SSE) from a stream, matching gold reference dJ1.
 */

export interface SseEvent {
    id?: string;
    event?: string;
    data: string;
}

export interface SseParserCallbacks {
    onEvent?: (event: SseEvent) => void;
    onError?: (error: Error) => void;
    onRetry?: (delay: number) => void;
    onComment?: (comment: string) => void;
}

/**
 * Creates an SSE parser that handles field accumulation and line splitting.
 */
export function createSseParser(callbacks: SseParserCallbacks) {
    const { onEvent, onError, onRetry, onComment } = callbacks;
    let buffer = "";
    let isFirstChunk = true;
    let currentId: string | undefined;
    let currentEvent: string | undefined;
    let currentData = "";

    function processLine(line: string) {
        if (line === "") {
            dispatch();
            return;
        }

        if (line.startsWith(":")) {
            onComment?.(line.slice(line.startsWith(": ") ? 2 : 1));
            return;
        }

        const colonIndex = line.indexOf(":");
        if (colonIndex !== -1) {
            const field = line.slice(0, colonIndex);
            const value = line.slice(colonIndex + (line[colonIndex + 1] === " " ? 2 : 1));

            switch (field) {
                case "event":
                    currentEvent = value;
                    break;
                case "data":
                    currentData += (currentData ? "\n" : "") + value;
                    break;
                case "id":
                    currentId = value.includes("\0") ? undefined : value;
                    break;
                case "retry":
                    if (/^\d+$/.test(value)) {
                        onRetry?.(parseInt(value, 10));
                    } else {
                        onError?.(new Error(`Invalid retry value: ${value}`));
                    }
                    break;
                default:
                    // Ignore unknown fields or optionally log/warn
                    break;
            }
            return;
        }

        // Field name only (e.g. "foo")
        switch (line) {
            case "event": currentEvent = ""; break;
            case "data": currentData += (currentData ? "\n" : ""); break;
            case "id": currentId = ""; break;
            default: break;
        }
    }

    function dispatch() {
        if (currentData.length > 0) {
            onEvent?.({
                id: currentId,
                event: currentEvent,
                data: currentData
            });
        }
        currentId = undefined;
        currentEvent = undefined;
        currentData = "";
    }

    function feed(chunk: string) {
        let chunkToProcess = isFirstChunk ? chunk.replace(/^\xEF\xBB\xBF/, "") : chunk;
        isFirstChunk = false;

        buffer += chunkToProcess;
        const lines = buffer.split(/\r\n|\r|\n/);
        buffer = lines.pop() || "";

        for (const line of lines) {
            processLine(line);
        }
    }

    function reset() {
        buffer = "";
        isFirstChunk = true;
        currentId = undefined;
        currentEvent = undefined;
        currentData = "";
    }

    return { feed, reset };
}

/**
 * An async generator that yields SSE events from a stream.
 */
export async function* parseSseEvents(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseEvent> {
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    const events: SseEvent[] = [];

    // Using a simpler loop for the async generator to avoid complex Promise management
    // and match the logic while ensuring type safety.
    const parser = createSseParser({});
    let resolveNext: (() => void) | null = null;

    // Set callback to wake up the generator
    (parser as any).onEvent = (ev: SseEvent) => {
        events.push(ev);
        if (resolveNext) {
            resolveNext();
            resolveNext = null;
        }
    };

    let isStreamDone = false;
    const readLoop = (async () => {
        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                parser.feed(decoder.decode(value, { stream: true }));
            }
        } finally {
            isStreamDone = true;
            if (resolveNext) {
                (resolveNext as () => void)();
                resolveNext = null;
            }
            reader.releaseLock();
        }
    })();

    while (true) {
        if (events.length > 0) {
            yield events.shift()!;
            continue;
        }

        if (isStreamDone) break;

        await new Promise<void>(resolve => {
            resolveNext = resolve;
            // Check if it already finished while we were setting up
            if (events.length > 0 || isStreamDone) {
                if (resolveNext) {
                    (resolveNext as () => void)();
                    resolveNext = null;
                }
            }
        });
    }

    await readLoop;
}
