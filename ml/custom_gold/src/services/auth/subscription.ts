import axios from "axios";
import { getAuthHeaders, getCliUserAgent } from "../claude/anthropicApiClient.js";
import { isOauthActive } from "./oauthManager.js";
import { getGlobalState, setGlobalState } from "../session/globalState.js";

const ACCESS_CACHE_TTL = 3600000; // 1 hour

/**
 * Checks if the organization has access to Sonnet-1M models.
 */
export async function checkSonnet1mAccess(orgUuid: string): Promise<{
    hasAccess: boolean;
    hasAccessNotAsDefault: boolean;
    hasError: boolean;
}> {
    const auth = getAuthHeaders();
    if (auth.error) throw new Error(`Auth error: ${auth.error}`);

    const headers = {
        "Content-Type": "application/json",
        "User-Agent": getCliUserAgent(),
        ...auth.headers
    };

    try {
        const url = `https://api.anthropic.com/api/organization/${orgUuid}/claude_code_sonnet_1m_access`;
        const response = await axios.get(url, { headers, timeout: 5000 });
        const data = response.data;
        return {
            hasAccess: data.has_access,
            hasAccessNotAsDefault: data.has_access_not_as_default,
            hasError: false
        };
    } catch (err) {
        return {
            hasAccess: false,
            hasAccessNotAsDefault: false,
            hasError: true
        };
    }
}

/**
 * Returns the cached status of Sonnet-1M access.
 */
export function getSonnet1mAccessStatus(orgUuid: string): {
    hasAccess: boolean;
    wasPartOfDefaultRollout: boolean;
    needsRefresh: boolean;
} {
    const state = getGlobalState() as any;
    const cache = isOauthActive() ? state.s1mAccessCache : state.s1mNonSubscriberAccessCache;
    const entry = cache?.[orgUuid];
    const now = Date.now();

    if (!entry) return { hasAccess: false, wasPartOfDefaultRollout: false, needsRefresh: true };

    const isExpired = now - entry.timestamp > ACCESS_CACHE_TTL;
    return {
        hasAccess: entry.hasAccess || (entry.hasAccessNotAsDefault ?? false),
        wasPartOfDefaultRollout: entry.hasAccess,
        needsRefresh: isExpired
    };
}

/**
 * Refreshes and caches the Sonnet-1M access status.
 */
export async function refreshSonnet1mAccess(orgUuid: string): Promise<void> {
    try {
        const { hasAccess, hasAccessNotAsDefault } = await checkSonnet1mAccess(orgUuid);
        const state = getGlobalState() as any;
        const cacheKey = isOauthActive() ? "s1mAccessCache" : "s1mNonSubscriberAccessCache";
        setGlobalState({
            ...state,
            [cacheKey]: {
                ...state[cacheKey],
                [orgUuid]: {
                    hasAccess,
                    hasAccessNotAsDefault,
                    timestamp: Date.now()
                }
            }
        });
    } catch (err) {
        // Silently fail
    }
}

/**
 * Fetches the date of the first token used by the organization.
 */
export async function fetchFirstTokenDate(): Promise<void> {
    try {
        const state = getGlobalState() as any;
        if (state.claudeCodeFirstTokenDate !== undefined) return;

        const auth = getAuthHeaders();
        if (auth.error) return;

        const baseUrl = process.env.CLAUDE_BASE_API_URL || "https://api.anthropic.com";
        const url = `${baseUrl}/api/organization/claude_code_first_token_date`;
        const response = await axios.get(url, {
            headers: {
                ...auth.headers,
                "User-Agent": getCliUserAgent()
            }
        });

        const firstTokenDate = (response.data as any)?.first_token_date ?? null;
        if (firstTokenDate !== null) {
            const timestamp = new Date(firstTokenDate).getTime();
            if (isNaN(timestamp)) return;
        }

        setGlobalState({
            ...state,
            claudeCodeFirstTokenDate: firstTokenDate
        });
    } catch (err) {
        // Silently fail
    }
}
