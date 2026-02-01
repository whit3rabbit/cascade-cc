/**
 * File: src/utils/http/SseParser.ts
 * Role: Parses Server-Sent Events (SSE) from a stream.
 */

export interface SseEvent {
    type: string;
    data: string;
}

/**
 * An async generator that yields SSE events from a stream.
 */
export async function* parseSseEvents(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseEvent> {
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            let currentEvent: string = "message";
            for (const line of lines) {
                if (line.startsWith("event:")) {
                    currentEvent = line.slice(6).trim();
                } else if (line.startsWith("data:")) {
                    yield { type: currentEvent, data: line.slice(5).trim() };
                    currentEvent = "message"; // reset for next data
                } else if (line === "") {
                    // end of event
                }
            }
        }
    } finally {
        reader.releaseLock();
    }
}
