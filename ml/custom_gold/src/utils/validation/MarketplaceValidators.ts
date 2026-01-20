
export const OFFICIAL_MARKETPLACES = new Set(["claude-code-marketplace", "claude-code-plugins", "claude-plugins-official", "anthropic-marketplace", "anthropic-plugins", "agent-skills", "life-sciences"]);
export const ANTHROPIC_ORG = "anthropics";

const RESERVED_NAMES_REGEX = /(?:official[^a-z0-9]*(anthropic|claude)|(?:anthropic|claude)[^a-z0-9]*official|^(?:anthropic|claude)[^a-z0-9]*(marketplace|plugins|official))/i;

export function isReservedName(name: string): boolean {
    if (OFFICIAL_MARKETPLACES.has(name.toLowerCase())) return false; // Allowed if it's strictly one of these (but logic in le3 returns false if in set?)
    // Wait, le3 returns !1 if in set. So it returns false if it IS an official name?
    // "le3" seems to mean "isImpersonating"?
    // If in official set, return false (not impersonating, it IS official).
    // If matches regex, return true (impersonating).

    if (OFFICIAL_MARKETPLACES.has(name.toLowerCase())) return false;
    if (RESERVED_NAMES_REGEX.test(name)) return true;
    return false;
}

export function validateMarketplaceName(name: string, source: { source: string, url?: string, repo?: string }): string | null {
    const lowerName = name.toLowerCase();

    // Logic from n02
    // If name is NOT in official set, return null (valid default).
    // Wait, n02 returns null if NOT in official set.
    // So n02 checks if you are TRYING to use an EXACT official name.

    if (!OFFICIAL_MARKETPLACES.has(lowerName)) return null;

    if (source.source === "github") {
        if (!(source.repo || "").toLowerCase().startsWith(`${ANTHROPIC_ORG}/`)) {
            return `The name '${name}' is reserved for official Anthropic marketplaces. Only repositories from 'github.com/${ANTHROPIC_ORG}/' can use this name.`;
        }
        return null;
    }

    if (source.source === "git" && source.url) {
        const url = source.url.toLowerCase();
        const isOfficial = url.includes("github.com/anthropics/") || url.includes("git@github.com:anthropics/");
        if (isOfficial) return null;
        return `The name '${name}' is reserved for official Anthropic marketplaces. Only repositories from 'github.com/${ANTHROPIC_ORG}/' can use this name.`;
    }

    return `The name '${name}' is reserved for official Anthropic marketplaces and can only be used with GitHub sources from the '${ANTHROPIC_ORG}' organization.`;
}
