import { OAuthService } from '../auth/OAuthService.js';
import { getProductName } from '../../utils/shared/product.js';

/**
 * Ensures the user is authenticated before submitting feedback.
 */
async function ensureAuthenticated(): Promise<void> {
    const token = await OAuthService.getValidToken();
    if (!token) {
        // In the original TUI, this might trigger a login flow or error out.
        // For deobfuscation, we'll assume the caller handles this or we just proceed.
    }
}

/**
 * Constructs authentication headers for feedback submission.
 */
async function getAuthHeaders(): Promise<{ headers: Record<string, string>; error?: string }> {
    const token = await OAuthService.getValidToken();
    if (!token) {
        return { headers: {}, error: 'No authentication token available' };
    }
    return {
        headers: {
            'Authorization': `Bearer ${token}`
        }
    };
}

/**
 * Returns a custom User-Agent string for the CLI.
 */
function getUserAgent(): string {
    return `${getProductName()}/0.0.1 (deobfuscated-env)`;
}

/**
 * Submits feedback/bug reports to the Anthropic API.
 * 
 * NOTE: Submission is DISABLED in this deobfuscation environment to prevent
 * unintentional data transmission to production endpoints.
 * 
 * @param report - The feedback data to "submit".
 * @param signal - Optional AbortSignal for cancellation.
 * @returns {Promise<{ success: boolean; feedbackId?: string; error?: string }>}
 */
export async function submitFeedback(report: any, signal?: AbortSignal): Promise<{ success: boolean; feedbackId?: string; error?: string }> {
    try {
        // Original logic flow:
        await ensureAuthenticated();

        const auth = await getAuthHeaders();
        if (auth.error) {
            // return { success: false, error: auth.error };
            // In deobfuscation mode, we proceed with simulated success regardless
        }

        const headers = {
            'Content-Type': 'application/json',
            'User-Agent': getUserAgent(),
            ...auth.headers
        };

        // --- ORIGINAL NETWORK CALL (DISABLED) ---
        /*
        const response = await axios.post('https://api.anthropic.com/api/claude_cli_feedback', {
            content: JSON.stringify(report)
        }, {
            headers,
            timeout: 30000,
            signal
        });
        
        if (response.status === 200 && response.data?.feedback_id) {
            return { success: true, feedbackId: response.data.feedback_id };
        }

        if (response.status === 403 && response.data?.error?.type === "permission_error" && response.data?.error?.message?.includes("custom data retention settings")) {
            return {
                success: false,
                error: "Feedback cannot be submitted because of your custom data retention settings."
            };
        }
        */

        // --- SIMULATED SUCCESS FOR DEOBFUSCATION ENVIRONMENT ---
        console.log('[FeedbackService] Submission simulations successful (no data sent to Anthropic)');

        return {
            success: true,
            feedbackId: `fb-${Math.random().toString(36).substring(2, 11)}`
        };
    } catch (e) {
        if (signal?.aborted) {
            return { success: false };
        }

        return {
            success: false,
            error: e instanceof Error ? e.message : String(e)
        };
    }
}
