
// Logic from d71, p71, k71

export class RetryError extends Error {
    public originalError: any;
    public retryContext: any;

    constructor(error: any, retryContext: any) {
        super(error instanceof Error ? error.message : String(error));
        this.originalError = error;
        this.retryContext = retryContext;
        this.name = "RetryError";
        if (error instanceof Error && error.stack) {
            this.stack = error.stack;
        }
    }
}

export class FallbackTriggeredError extends Error {
    constructor(public originalModel: string, public fallbackModel: string) {
        super(`Model fallback triggered: ${originalModel} -> ${fallbackModel}`);
        this.name = "FallbackTriggeredError";
    }
}

export function calculateBackoff(attempt: number, retryAfter: string | null | undefined): number {
    if (retryAfter) {
        const seconds = parseInt(retryAfter, 10);
        if (!isNaN(seconds)) return seconds * 1000;
    }
    const base = Math.min(500 * Math.pow(2, attempt - 1), 32000);
    const jitter = Math.random() * 0.25 * base;
    return base + jitter;
}

export async function sleep(ms: number, signal?: AbortSignal) {
    return new Promise<void>((resolve, reject) => {
        const timer = setTimeout(resolve, ms);
        if (signal) {
            const abortHandler = () => {
                clearTimeout(timer);
                reject(new Error("Aborted")); // Or custom AbortError
            };
            if (signal.aborted) {
                abortHandler();
                return;
            }
            signal.addEventListener("abort", abortHandler, { once: true });
            // cleanup listener
            setTimeout(() => signal.removeEventListener("abort", abortHandler), ms);
        }
    });
}

// Logic related to token usage extraction (jo, GSA, gK, FY2, EY2, c71, zY2)
export function getTotalTokens(usage: any): number {
    if (!usage) return 0;
    return (usage.input_tokens || 0) +
        (usage.cache_creation_input_tokens || 0) +
        (usage.cache_read_input_tokens || 0) +
        (usage.output_tokens || 0);
}
