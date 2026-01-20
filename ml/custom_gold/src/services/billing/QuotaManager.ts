
import { formatTimestamp } from "../../utils/shared/lodashLikeRuntimeAndEnv.js";
import { getActiveModel } from "../claude/claudeUtils.js";

// Logic related to Rate Limit and Overage handling (KX5, WX5, XY2, etc.)

export type RateLimitType =
    | "five_hour"
    | "seven_day"
    | "seven_day_opus"
    | "seven_day_sonnet"
    | "overage";

export type RateLimitStatus = "allowed" | "allowed_warning" | "rejected";

export interface RateLimitPlanContext {
    plan?: "default" | "pro" | "max" | "team" | "enterprise";
    hasExtraUsageEnabled?: boolean;
}

export interface QuotaStatus {
    status: RateLimitStatus;
    resetsAt?: number;
    unifiedRateLimitFallbackAvailable: boolean;
    rateLimitType?: RateLimitType;
    overageStatus?: string;
    overageResetsAt?: number;
    overageDisabledReason?: string;
    isUsingOverage: boolean;
    utilization?: number;
    surpassedThreshold?: number;
}

export interface HeaderReader {
    get(name: string): string | null;
}

export type QuotaListener = (quota: QuotaStatus) => void;

export interface QuotaRequestOptions {
    client: any;
    model?: string;
    betas?: string[];
    metadata?: Record<string, any>;
    signal?: AbortSignal;
}

export interface QuotaRefreshOptions {
    isQuotaEnabled: boolean;
    emitTelemetry?: (payload: any) => void;
}

const CLAIM_TO_RATE_LIMIT: Record<string, RateLimitType> = {
    "5h": "five_hour",
    "7d": "seven_day",
    overage: "overage"
};

const RATE_LIMIT_WARNING_RULES = [
    {
        rateLimitType: "five_hour" as const,
        claimAbbrev: "5h",
        windowSeconds: 18000,
        thresholds: [{ utilization: 0.9, timePct: 0.72 }]
    },
    {
        rateLimitType: "seven_day" as const,
        claimAbbrev: "7d",
        windowSeconds: 604800,
        thresholds: [
            { utilization: 0.75, timePct: 0.6 },
            { utilization: 0.5, timePct: 0.35 },
            { utilization: 0.25, timePct: 0.15 }
        ]
    }
];

function getHeader(headers: HeaderReader | Record<string, string> | undefined, name: string): string | null {
    if (!headers) return null;
    if (typeof (headers as HeaderReader).get === "function") {
        return (headers as HeaderReader).get(name);
    }
    const lowerName = name.toLowerCase();
    const record = headers as Record<string, string>;
    for (const [key, value] of Object.entries(record)) {
        if (key.toLowerCase() === lowerName) return value ?? null;
    }
    return null;
}

function getPlanContext(context?: RateLimitPlanContext): RateLimitPlanContext {
    return {
        plan: context?.plan ?? "default",
        hasExtraUsageEnabled: context?.hasExtraUsageEnabled ?? false
    };
}

function formatResetTime(timestampSeconds?: number): string | null {
    if (!timestampSeconds) return null;
    return formatTimestamp(timestampSeconds, true);
}

function formatHitLimit(limitName: string, resetSuffix: string): string {
    return `You've hit your ${limitName}${resetSuffix}`;
}

function formatOverageResetSuffix(quota: QuotaStatus, limitReset?: string, overageReset?: string): string {
    if (quota.resetsAt && quota.overageResetsAt) {
        return quota.resetsAt < quota.overageResetsAt
            ? ` · resets ${limitReset}`
            : ` · resets ${overageReset}`;
    }
    if (limitReset) return ` · resets ${limitReset}`;
    if (overageReset) return ` · resets ${overageReset}`;
    return "";
}

function calculateWindowProgress(resetAtSeconds: number, windowSeconds: number): number {
    const nowSeconds = Date.now() / 1000;
    const windowStart = resetAtSeconds - windowSeconds;
    const elapsed = nowSeconds - windowStart;
    return Math.max(0, Math.min(1, elapsed / windowSeconds));
}

function getRateLimitCallToAction(rateLimitType?: RateLimitType, context?: RateLimitPlanContext): string | null {
    if (!rateLimitType) return null;
    const { plan, hasExtraUsageEnabled } = getPlanContext(context);

    if (rateLimitType === "five_hour") {
        if (plan === "team" || plan === "enterprise") {
            if (!hasExtraUsageEnabled) return "/extra-usage to request more";
            return null;
        }
        if (plan === "pro" || plan === "max") {
            return "/upgrade to keep using Claude Code";
        }
    }

    if (rateLimitType === "overage" && (plan === "team" || plan === "enterprise") && !hasExtraUsageEnabled) {
        return "/extra-usage to request more";
    }

    return null;
}

function getLimitLabel(rateLimitType?: RateLimitType, context?: RateLimitPlanContext): string | null {
    if (!rateLimitType) return null;
    switch (rateLimitType) {
        case "seven_day":
            return "weekly limit";
        case "five_hour":
            return "session limit";
        case "seven_day_opus":
            return "Opus limit";
        case "seven_day_sonnet": {
            const { plan } = getPlanContext(context);
            return plan === "pro" || plan === "enterprise" ? "weekly limit" : "Sonnet limit";
        }
        case "overage":
            return "extra usage";
        default:
            return null;
    }
}

function getOverageSwitchMessage(quota: QuotaStatus, context?: RateLimitPlanContext): string {
    const resetTime = formatResetTime(quota.resetsAt);
    let limitLabel = "";

    if (quota.rateLimitType === "five_hour") limitLabel = "session limit";
    else if (quota.rateLimitType === "seven_day") limitLabel = "weekly limit";
    else if (quota.rateLimitType === "seven_day_opus") limitLabel = "Opus limit";
    else if (quota.rateLimitType === "seven_day_sonnet") {
        const { plan } = getPlanContext(context);
        limitLabel = plan === "pro" || plan === "enterprise" ? "weekly limit" : "Sonnet limit";
    }

    if (!limitLabel) return "Now using extra usage";
    return `You're now using extra usage${resetTime ? ` · Your ${limitLabel} resets ${resetTime}` : ""}`;
}

function checkSurpassedThreshold(headers: HeaderReader | Record<string, string>, fallbackAvailable: boolean): QuotaStatus | null {
    for (const [claim, rateLimitType] of Object.entries(CLAIM_TO_RATE_LIMIT)) {
        const thresholdValue = getHeader(headers, `anthropic-ratelimit-unified-${claim}-surpassed-threshold`);
        if (thresholdValue !== null) {
            const utilization = getHeader(headers, `anthropic-ratelimit-unified-${claim}-utilization`);
            const reset = getHeader(headers, `anthropic-ratelimit-unified-${claim}-reset`);
            return {
                status: "allowed_warning",
                resetsAt: reset ? Number(reset) : undefined,
                rateLimitType,
                utilization: utilization ? Number(utilization) : undefined,
                unifiedRateLimitFallbackAvailable: fallbackAvailable,
                isUsingOverage: false,
                surpassedThreshold: Number(thresholdValue)
            };
        }
    }
    return null;
}

function checkThresholdRules(
    headers: HeaderReader | Record<string, string>,
    rule: (typeof RATE_LIMIT_WARNING_RULES)[number],
    fallbackAvailable: boolean
): QuotaStatus | null {
    const utilizationValue = getHeader(headers, `anthropic-ratelimit-unified-${rule.claimAbbrev}-utilization`);
    const resetValue = getHeader(headers, `anthropic-ratelimit-unified-${rule.claimAbbrev}-reset`);
    if (utilizationValue === null || resetValue === null) return null;

    const utilization = Number(utilizationValue);
    const resetAt = Number(resetValue);
    const timePct = calculateWindowProgress(resetAt, rule.windowSeconds);

    if (!rule.thresholds.some((threshold) => utilization >= threshold.utilization && timePct <= threshold.timePct)) {
        return null;
    }

    return {
        status: "allowed_warning",
        resetsAt: resetAt,
        rateLimitType: rule.rateLimitType,
        utilization,
        unifiedRateLimitFallbackAvailable: fallbackAvailable,
        isUsingOverage: false
    };
}

function computeWarningStatus(
    headers: HeaderReader | Record<string, string>,
    fallbackAvailable: boolean
): QuotaStatus | null {
    const surpassed = checkSurpassedThreshold(headers, fallbackAvailable);
    if (surpassed) return surpassed;

    for (const rule of RATE_LIMIT_WARNING_RULES) {
        const warning = checkThresholdRules(headers, rule, fallbackAvailable);
        if (warning) return warning;
    }

    return null;
}

function normalizeRepresentativeClaim(value: string | null): RateLimitType | undefined {
    if (!value) return undefined;
    if (value in CLAIM_TO_RATE_LIMIT) return CLAIM_TO_RATE_LIMIT[value];
    if (
        value === "five_hour" ||
        value === "seven_day" ||
        value === "seven_day_opus" ||
        value === "seven_day_sonnet" ||
        value === "overage"
    ) {
        return value;
    }
    return undefined;
}

function areQuotaStatusesEqual(a: QuotaStatus, b: QuotaStatus): boolean {
    return (
        a.status === b.status &&
        a.resetsAt === b.resetsAt &&
        a.unifiedRateLimitFallbackAvailable === b.unifiedRateLimitFallbackAvailable &&
        a.rateLimitType === b.rateLimitType &&
        a.overageStatus === b.overageStatus &&
        a.overageResetsAt === b.overageResetsAt &&
        a.overageDisabledReason === b.overageDisabledReason &&
        a.isUsingOverage === b.isUsingOverage &&
        a.utilization === b.utilization &&
        a.surpassedThreshold === b.surpassedThreshold
    );
}

const DEFAULT_QUOTA_STATUS: QuotaStatus = {
    status: "allowed",
    unifiedRateLimitFallbackAvailable: false,
    isUsingOverage: false
};

export class QuotaManager {
    private static currentStatus: QuotaStatus = { ...DEFAULT_QUOTA_STATUS };
    private static listeners = new Set<QuotaListener>();

    static getCurrentStatus(): QuotaStatus {
        return { ...QuotaManager.currentStatus };
    }

    static subscribe(listener: QuotaListener): () => void {
        QuotaManager.listeners.add(listener);
        return () => QuotaManager.listeners.delete(listener);
    }

    static updateStatus(nextStatus: QuotaStatus, emitTelemetry?: (payload: any) => void): void {
        if (areQuotaStatusesEqual(QuotaManager.currentStatus, nextStatus)) return;
        QuotaManager.currentStatus = { ...nextStatus };
        for (const listener of QuotaManager.listeners) {
            listener({ ...nextStatus });
        }

        if (emitTelemetry) {
            const hoursTillReset = Math.round(
                (nextStatus.resetsAt ? nextStatus.resetsAt - Date.now() / 1000 : 0) / 3600
            );
            emitTelemetry({
                status: nextStatus.status,
                unifiedRateLimitFallbackAvailable: nextStatus.unifiedRateLimitFallbackAvailable,
                hoursTillReset
            });
        }
    }

    static formatRateLimitMessage(
        quota: QuotaStatus,
        _model: string,
        context?: RateLimitPlanContext
    ): string | null {
        const limitReset = formatResetTime(quota.resetsAt);
        const overageReset = formatResetTime(quota.overageResetsAt);
        const resetSuffix = limitReset ? ` · resets ${limitReset}` : "";

        if (quota.overageStatus === "rejected") {
            const overageSuffix = formatOverageResetSuffix(quota, limitReset ?? undefined, overageReset ?? undefined);
            if (quota.overageDisabledReason === "out_of_credits") return `You're out of extra usage${overageSuffix}`;
            return formatHitLimit("limit", overageSuffix);
        }

        if (quota.rateLimitType === "seven_day_sonnet") {
            const { plan } = getPlanContext(context);
            const limitLabel = plan === "pro" || plan === "enterprise" ? "weekly limit" : "Sonnet limit";
            return formatHitLimit(limitLabel, resetSuffix);
        }
        if (quota.rateLimitType === "seven_day_opus") return formatHitLimit("Opus limit", resetSuffix);
        if (quota.rateLimitType === "seven_day") return formatHitLimit("weekly limit", resetSuffix);
        if (quota.rateLimitType === "five_hour") return formatHitLimit("session limit", resetSuffix);

        return formatHitLimit("usage limit", resetSuffix);
    }

    static getWarningMessage(quota: QuotaStatus, context?: RateLimitPlanContext): string | null {
        if (quota.status !== "allowed_warning") return null;

        const limitLabel = getLimitLabel(quota.rateLimitType, context);
        if (!limitLabel) return null;

        const utilizationPercentage = quota.utilization ? Math.floor(quota.utilization * 100) : undefined;
        const resetTime = formatResetTime(quota.resetsAt);
        const callToAction = getRateLimitCallToAction(quota.rateLimitType, context);

        if (utilizationPercentage && resetTime) {
            const message = `You've used ${utilizationPercentage}% of your ${limitLabel} · resets ${resetTime}`;
            return callToAction ? `${message} · ${callToAction}` : message;
        }

        if (utilizationPercentage) {
            const message = `You've used ${utilizationPercentage}% of your ${limitLabel}`;
            return callToAction ? `${message} · ${callToAction}` : message;
        }

        const approachingLabel = quota.rateLimitType === "overage" ? `${limitLabel} limit` : limitLabel;
        if (resetTime) {
            const message = `Approaching ${approachingLabel} · resets ${resetTime}`;
            return callToAction ? `${message} · ${callToAction}` : message;
        }

        const message = `Approaching ${approachingLabel}`;
        return callToAction ? `${message} · ${callToAction}` : message;
    }

    static getRejectionMessage(quota: QuotaStatus, model: string, context?: RateLimitPlanContext): string | null {
        if (quota.status !== "rejected") return null;
        return QuotaManager.formatRateLimitMessage(quota, model, context);
    }

    static getOverageMessage(quota: QuotaStatus, context?: RateLimitPlanContext): string {
        return getOverageSwitchMessage(quota, context);
    }

    static parseQuotaHeaders(headers: HeaderReader | Record<string, string>): QuotaStatus {
        const status = (getHeader(headers, "anthropic-ratelimit-unified-status") ?? "allowed") as RateLimitStatus;
        const reset = getHeader(headers, "anthropic-ratelimit-unified-reset");
        const resetsAt = reset ? Number(reset) : undefined;
        const fallbackAvailable = getHeader(headers, "anthropic-ratelimit-unified-fallback") === "available";
        const representativeClaim = getHeader(headers, "anthropic-ratelimit-unified-representative-claim");
        const overageStatus = getHeader(headers, "anthropic-ratelimit-unified-overage-status") ?? undefined;
        const overageReset = getHeader(headers, "anthropic-ratelimit-unified-overage-reset");
        const overageResetsAt = overageReset ? Number(overageReset) : undefined;
        const overageDisabledReason =
            getHeader(headers, "anthropic-ratelimit-unified-overage-disabled-reason") ?? undefined;
        const isUsingOverage = status === "rejected" && (overageStatus === "allowed" || overageStatus === "allowed_warning");

        if (status === "allowed" || status === "allowed_warning") {
            const warning = computeWarningStatus(headers, fallbackAvailable);
            if (warning) return warning;
        }

        const rateLimitType = normalizeRepresentativeClaim(representativeClaim);

        return {
            status: status === "allowed_warning" ? "allowed" : status,
            resetsAt,
            unifiedRateLimitFallbackAvailable: fallbackAvailable,
            ...(rateLimitType ? { rateLimitType } : {}),
            ...(overageStatus ? { overageStatus } : {}),
            ...(overageResetsAt ? { overageResetsAt } : {}),
            ...(overageDisabledReason ? { overageDisabledReason } : {}),
            isUsingOverage
        };
    }

    static refreshFromHeaders(
        headers: HeaderReader | Record<string, string>,
        isQuotaEnabled: boolean,
        emitTelemetry?: (payload: any) => void
    ): void {
        if (!isQuotaEnabled) {
            if (QuotaManager.currentStatus.status !== "allowed" || QuotaManager.currentStatus.resetsAt) {
                QuotaManager.updateStatus({ ...DEFAULT_QUOTA_STATUS }, emitTelemetry);
            }
            return;
        }

        const nextStatus = QuotaManager.parseQuotaHeaders(headers);
        QuotaManager.updateStatus(nextStatus, emitTelemetry);
    }

    static async requestQuotaHeaders(options: QuotaRequestOptions): Promise<HeaderReader | Record<string, string> | null> {
        const {
            client,
            model = getActiveModel(),
            betas = [],
            metadata,
            signal
        } = options;

        if (!client?.beta?.messages?.create) return null;

        const response = await client.beta.messages.create(
            {
                model,
                max_tokens: 1,
                messages: [{ role: "user", content: "quota" }],
                ...(metadata ? { metadata } : {}),
                ...(betas.length > 0 ? { betas } : {})
            },
            signal ? { signal } : undefined
        );

        if (response?.headers) return response.headers;
        if (typeof response?.asResponse === "function") {
            const resp = await response.asResponse();
            return resp?.headers ?? null;
        }
        return null;
    }

    static async refreshFromApi(
        options: QuotaRequestOptions & QuotaRefreshOptions
    ): Promise<void> {
        if (!options.isQuotaEnabled) return;

        try {
            const headers = await QuotaManager.requestQuotaHeaders(options);
            if (headers) {
                QuotaManager.refreshFromHeaders(headers, options.isQuotaEnabled, options.emitTelemetry);
            }
        } catch (error) {
            QuotaManager.handleRateLimitError(error, options.isQuotaEnabled, options.emitTelemetry);
        }
    }

    static handleRateLimitError(
        error: any,
        isQuotaEnabled: boolean,
        emitTelemetry?: (payload: any) => void
    ): void {
        if (!isQuotaEnabled || error?.status !== 429) return;

        let nextStatus: QuotaStatus = { ...QuotaManager.currentStatus };
        if (error?.headers) {
            nextStatus = QuotaManager.parseQuotaHeaders(error.headers);
        }
        nextStatus = {
            ...nextStatus,
            status: "rejected"
        };

        QuotaManager.updateStatus(nextStatus, emitTelemetry);
    }
}

export function formatRelativeTime(timestamp: number): string {
    return formatTimestamp(timestamp, true);
}

// Logic from cY0 (Error handling/mapping)
export function mapApiErrorToMessage(error: any, model: string, context?: any): { content: string, error?: string } {
    if (error.message && error.message.includes("rate_limit")) {
        return { content: "Rate limit reached", error: "rate_limit" };
    }
    // ... logic from cY0
    return { content: `API Error: ${error.message || "Unknown error"}`, error: "unknown" };
}
