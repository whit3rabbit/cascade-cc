/**
 * File: src/services/telemetry/TracingManager.ts
 * Role: Manages high-level OpenTelemetry spans for user interactions and LLM requests.
 */

import { trace, context, Span, Tracer } from '@opentelemetry/api';
import { getSettings } from '../config/SettingsService.js';

const tracer: Tracer = trace.getTracer("com.anthropic.claude_code.tracing", "1.0.0");

let interactionSequence = 0;
const spanMap = new Map<string, { span: Span; startTime: number }>();

/**
 * Checks if enhanced telemetry is enabled via environment variables or settings.
 */
export function isEnhancedTelemetryEnabled(): boolean {
    if (process.env.CLAUDE_CODE_ENHANCED_TELEMETRY_BETA || process.env.ENABLE_ENHANCED_TELEMETRY_BETA) {
        return true;
    }
    return !!getSettings().telemetry?.enhanced;
}

/**
 * Starts a new interaction span representing a single user input-to-response cycle.
 */
export function startInteractionSpan(userPrompt?: string): Span {
    if (!isEnhancedTelemetryEnabled()) {
        const activeSpan = trace.getSpan(context.active());
        if (activeSpan) return activeSpan;
    }

    interactionSequence++;
    const span = tracer.startSpan("claude_code.interaction", {
        attributes: {
            "span.type": "interaction",
            "interaction.sequence": interactionSequence,
            "user_prompt_length": userPrompt?.length || 0
        }
    });

    const spanContext = span.spanContext();
    spanMap.set(spanContext.spanId, { span, startTime: Date.now() });

    return span;
}

/**
 * Ends the currently active interaction span.
 */
export function endInteractionSpan(span?: Span): void {
    if (!span) return;
    const spanContext = span.spanContext();
    const info = spanMap.get(spanContext.spanId);

    if (info) {
        const duration = Date.now() - info.startTime;
        span.setAttribute("interaction.duration_ms", duration);
        span.end();
        spanMap.delete(spanContext.spanId);
    }
}

/**
 * Starts a span for an LLM API request.
 */
export function startLLMRequestSpan(model: string, querySource: string): Span {
    if (!isEnhancedTelemetryEnabled()) {
        const activeSpan = trace.getSpan(context.active());
        if (activeSpan) return activeSpan;
    }

    const span = tracer.startSpan("claude_code.llm_request", {
        attributes: {
            "span.type": "llm_request",
            "model": model,
            "query_source": querySource
        }
    });

    const spanContext = span.spanContext();
    spanMap.set(spanContext.spanId, { span, startTime: Date.now() });

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
    const spanContext = span.spanContext();
    const info = spanMap.get(spanContext.spanId);

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
        spanMap.delete(spanContext.spanId);
    }
}
