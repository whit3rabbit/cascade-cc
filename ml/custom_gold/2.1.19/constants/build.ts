/**
 * Static information about the current build.
 */
export const BUILD_INFO = {
    VERSION: "2.1.19",
    BUILD_TIME: "2026-01-23T21:13:41Z",
    PACKAGE_NAME: "@anthropic-ai/claude-code",
    ISSUES_URL: "https://github.com/anthropics/claude-code/issues",
    README_URL: "https://code.claude.com/docs/en/overview"
} as const;

/**
 * Returns a standardized user-agent string for network requests.
 * 
 * @returns The User-Agent string.
 */
export function getUserAgent(): string {
    return `claude-cli/${BUILD_INFO.VERSION} (external, ${process.env.CLAUDE_CODE_ENTRYPOINT || 'native'})`;
}
