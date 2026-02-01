
import { PRODUCT_SYSTEM_PROMPT, PRODUCT_SYSTEM_PROMPT_SDK, AGENT_SYSTEM_PROMPT } from '../constants/product.js';

/**
 * Gets the appropriate system prompt based on the execution context.
 */
export function getSystemPrompt(options: {
    isNonInteractive?: boolean;
    hasAppendSystemPrompt?: boolean;
    isVertex?: boolean;
} = {}): string {
    if (options.isVertex) {
        return PRODUCT_SYSTEM_PROMPT;
    }
    if (options.isNonInteractive) {
        if (options.hasAppendSystemPrompt) {
            return PRODUCT_SYSTEM_PROMPT_SDK;
        }
        return AGENT_SYSTEM_PROMPT;
    }
    return PRODUCT_SYSTEM_PROMPT;
}

/**
 * Generates the attribution strings for commits and PRs.
 */
export function getAttribution(username: string, config: {
    attribution?: { commit?: string; pr?: string };
    includeCoAuthoredBy?: boolean;
    isRemote?: boolean;
    remoteSessionId?: string;
    ingressUrl?: string;
}) {
    if (config.isRemote) {
        const { remoteSessionId, ingressUrl } = config;
        if (remoteSessionId && ingressUrl && !ingressUrl.includes("localhost")) {
            const url = `${ingressUrl}/session/${remoteSessionId}`; // Assuming format based on uWA
            return {
                commit: url,
                pr: url
            };
        }
        return { commit: "", pr: "" };
    }

    const generatedWith = "🤖 Generated with [Claude Code](https://claude.com/claude-code)";
    const coAuthoredBy = `Co-Authored-By: ${username} <noreply@anthropic.com>`;

    if (config.attribution) {
        return {
            commit: config.attribution.commit ?? coAuthoredBy,
            pr: config.attribution.pr ?? generatedWith
        };
    }

    if (config.includeCoAuthoredBy === false) {
        return { commit: "", pr: "" };
    }

    return {
        commit: coAuthoredBy,
        pr: generatedWith
    };
}

/**
 * Selects a company announcement to display.
 */
export function getCompanyAnnouncement(
    announcements: string[] | undefined,
    numStartups: number
): string | undefined {
    if (!announcements || announcements.length === 0) {
        return undefined;
    }
    // If first startup, show the first announcement (deterministic)
    // Otherwise show a random one
    if (numStartups === 1) {
        return announcements[0];
    }
    return announcements[Math.floor(Math.random() * announcements.length)];
}
