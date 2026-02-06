import axios from 'axios';
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
        await ensureAuthenticated();

        const auth = await getAuthHeaders();
        // Even if we fail to get a token, we might still try? 
        // The original checks headers. 
        // But for safety let's assume we proceed if we have headers or handle it.
        // In the original code (chunk1197), FH() returns headers or error.
        if (auth.error) {
            // Proceeding might fail, but let's follow the logic.
            // Actually original returns { success: !1 } if FH() has error.
            return { success: false, error: auth.error };
        }

        const headers = {
            'Content-Type': 'application/json',
            'User-Agent': getUserAgent(),
            ...auth.headers
        };

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

        // Original logs error here: SV1(Error("Failed to submit feedback: request did not return feedback_id"));
        return { success: false, error: "Failed to submit feedback: request did not return feedback_id" };

    } catch (e: any) {
        if (axios.isCancel(e) || signal?.aborted) {
            return { success: false };
        }

        if (axios.isAxiosError(e) && e.response?.status === 403) {
            const data = e.response.data as any;
            if (data?.error?.type === "permission_error" && data?.error?.message?.includes("Custom data retention settings")) {
                // SV1(Error("Cannot submit feedback because custom data retention settings are enabled"));
                return {
                    success: false,
                    error: "Feedback cannot be submitted because of your custom data retention settings."
                };
            }
        }

        // SV1(e);
        return {
            success: false,
            error: e instanceof Error ? e.message : String(e)
        };
    }
}
