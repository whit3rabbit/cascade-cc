/**
 * File: src/services/telemetry/TracingManager.ts
 * Role: Manages high-level OpenTelemetry spans for user interactions and LLM requests.
 */

// We'll create a mock for @opentelemetry/api since it might not be installed
// or we can try to use it if available. For this refinement, I'll assume we need to
// stub it or use a wrapper. The JS code used it directly.
// I'll create a stub interface here to avoid dependency issues during this phase.

interface Span {
    spanContext(): { spanId: string };
    setAttribute(key: string, value: any): void;
    setAttributes(attributes: Record<string, any>): void;
    end(): void;
}

interface Tracer {
    startSpan(name: string, options?: any): Span;
}

// Mock implementation to prevent runtime crashes if otel is missing
const mockSpan: Span = {
    spanContext: () => ({ spanId: `mock-${Date.now()}` }),
    setAttribute: () => { },
    setAttributes: () => { },
    end: () => { }
};

const mockTracer: Tracer = {
    startSpan: () => mockSpan
};

// Start of actual code logic with fallbacks
let trace: any;
let context: any;
let otelTrace: any;
let tracer: any = mockTracer;

try {
    const otel = require('@opentelemetry/api');
    trace = otel.trace;
    context = otel.context;
    otelTrace = otel.trace;
    tracer = otelTrace.getTracer("com.anthropic.claude_code.tracing", "1.0.0");
} catch (e) {
    // OpenTelemetry not found, using mocks
    trace = { getSpan: () => mockSpan, setSpan: (_ctx: any, span: any) => span };
    context = { active: () => ({}), with: (_ctx: any, fn: Function) => fn() };
}

let interactionSequence = 0;
const spanMap = new Map<string, { span: Span; startTime: number }>();

/**
 * Checks if enhanced telemetry is enabled via environment variables.
 */
function isEnhancedTelemetryEnabled(): boolean {
    return !!(process.env.CLAUDE_CODE_ENHANCED_TELEMETRY_BETA || process.env.ENABLE_ENHANCED_TELEMETRY_BETA);
}

/**
 * Starts a new interaction span representing a single user input-to-response cycle.
 */
export function startInteractionSpan(userPrompt?: string): Span {
    if (!isEnhancedTelemetryEnabled()) return trace.getSpan(context.active()) || mockSpan;

    interactionSequence++;
    const span = tracer.startSpan("claude_code.interaction", {
        attributes: {
            "span.type": "interaction",
            "interaction.sequence": interactionSequence,
            "user_prompt_length": userPrompt?.length || 0
        }
    });

    const spanId = span.spanContext().spanId;
    spanMap.set(spanId, { span, startTime: Date.now() });

    // In a real TS environment we'd need proper context management types
    // For now we assume the JS logic works via the mock or real lib
    return context.with(trace.setSpan(context.active(), span), () => span);
}

/**
 * Ends the currently active interaction span.
 */
export function endInteractionSpan(span?: Span): void {
    if (!span) return;
    const spanId = span.spanContext().spanId;
    const info = spanMap.get(spanId);

    if (info) {
        const duration = Date.now() - info.startTime;
        span.setAttribute("interaction.duration_ms", duration);
        span.end();
        spanMap.delete(spanId);
    }
}

/**
 * Starts a span for an LLM API request.
 */
export function startLLMRequestSpan(model: string, querySource: string): Span {
    if (!isEnhancedTelemetryEnabled()) return trace.getSpan(context.active()) || mockSpan;

    const span = tracer.startSpan("claude_code.llm_request", {
        attributes: {
            "span.type": "llm_request",
            "model": model,
            "query_source": querySource
        }
    });

    const spanId = span.spanContext().spanId;
    spanMap.set(spanId, { span, startTime: Date.now() });

    return span;
}

export interface LLMRequestStats {
    inputTokens?: number;
    outputTokens?: number;
    ttftMs?: number;
    success?: boolean;
}

/**
 * Ends an LLM request span with performance metrics.
 */
export function endLLMRequestSpan(span?: Span, stats: LLMRequestStats = {}): void {
    if (!span) return;
    const spanId = span.spanContext().spanId;
    const info = spanMap.get(spanId);

    if (info) {
        const duration = Date.now() - info.startTime;
        span.setAttributes({
            "duration_ms": duration,
            "input_tokens": stats.inputTokens,
            "output_tokens": stats.outputTokens,
            "ttft_ms": stats.ttftMs,
            "success": stats.success ?? true
        });
        span.end();
        spanMap.delete(spanId);
    }
}
