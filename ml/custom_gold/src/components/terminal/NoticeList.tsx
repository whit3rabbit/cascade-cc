
// Logic from chunk_509.ts (Notices, Changelog, Mascot)

import React, { useState, useEffect, useRef } from 'react';
import { Box, Text, useInput } from 'ink';
import { figures } from '../../vendor/terminalFigures.js';
import { useTheme } from '../../services/terminal/themeManager.js';
import { getApiKeyWithSource } from '../../services/auth/apiKeyManager.js';
import { getAuthTokenSource } from '../../services/auth/authSource.js';
import { formatCompactNumber } from '../../utils/shared/lodashLikeRuntimeAndEnv.js';

// --- Constants & Types ---

const LARGE_FILE_THRESHOLD_CHARS = 300000; // km
const ULTRA_CLAUDE_MD_THRESHOLD_CHARS = 50000; // cKA
const LARGE_AGENT_DESCRIPTION_THRESHOLD_TOKENS = 10000; // M4A
const REACT_VULN_THRESHOLD = 3; // D17

interface Notice {
    id: string;
    type: 'warning' | 'info';
    isActive: (context: NoticeContext) => boolean;
    render: (context: NoticeContext) => React.ReactNode;
}

interface NoticeContext {
    config?: any; // AppState/Config
    showSonnet1MNotice?: boolean;
    showOpus45Notice?: boolean;
    agentDefinitions?: any[]; // Agent definitions
}

// --- Helper Stubs (Mocking services not yet fully implemented) ---

function getLargeMemoryFiles(): { path: string; content: string }[] {
    // Stub: In real app, check FileHistoryManager for large files
    return [];
}

function getUltraClaudeMd(): { content: string } | null {
    // Stub
    return null;
}

function isOauthEnabled() {
    return process.env.CLAUDE_CODE_OAUTH_ENABLED === 'true';
}

function isJetBrainsPluginInstalled(ide?: string) {
    // Stub
    return false;
}

function getJetBrainsIde() {
    return null;
}

function isPluginInstalled(ide: string) {
    return false;
}

function checkReactVulnerability() {
    return { detected: false, packageManager: null, packageName: null, version: null, package: null };
}

function calculateAgentDescriptionsTokens(definitions: any[]) {
    // Stub calculation
    return definitions ? definitions.length * 100 : 0;
}

function logEvent(name: string, data?: any) {
    // Stub telemetry
}

function updateConfig(updater: (config: any) => any) {
    // Stub config update
}

// --- Notices Definitions (z17) ---

const notices: Notice[] = [
    {
        id: "react-vulnerability", // E17
        type: "warning",
        isActive: () => {
            // In real app: check vulnerability count and detected status
            const vuln = checkReactVulnerability();
            // if (count >= REACT_VULN_THRESHOLD) return false;
            return vuln.detected === true;
        },
        render: () => {
            const vuln = checkReactVulnerability();
            if (!vuln.detected || !vuln.packageName) return null;
            const updateCmd = vuln.packageManager === 'npm' ? `npm install ${vuln.packageName}@latest` : `yarn upgrade ${vuln.packageName}`;
            const cve = vuln.package === 'next' ? "CVE-2025-66478" : "CVE-2025-55182";
            const label = vuln.package === 'next' ? `Next.js ${vuln.version}` : `${vuln.packageName}@${vuln.version}`;
            return (
                <Box flexDirection="row" gap={1}>
                    <Text color="yellow">{figures.warning}</Text>
                    <Text color="yellow">
                        {label} has a critical vulnerability ({cve}) that could allow attackers to execute arbitrary code on your server. Run `{updateCmd}` to update.
                    </Text>
                </Box>
            );
        }
    },
    {
        id: "large-memory-files", // Z17
        type: "warning",
        isActive: () => getLargeMemoryFiles().length > 0,
        render: () => {
            const files = getLargeMemoryFiles();
            return (
                <>
                    {files.map(f => (
                        <Box key={f.path} flexDirection="row">
                            <Text color="yellow">{figures.warning}</Text>
                            <Text color="yellow">
                                Large <Text bold>{f.path}</Text> will impact performance ({formatCompactNumber(f.content.length)} chars &gt; {formatCompactNumber(LARGE_FILE_THRESHOLD_CHARS)})
                                <Text dimColor> • /memory to edit</Text>
                            </Text>
                        </Box>
                    ))}
                </>
            );
        }
    },
    {
        id: "ultra-claude-md", // Y17
        type: "warning",
        isActive: () => {
            const md = getUltraClaudeMd();
            return md !== null && md.content.length > ULTRA_CLAUDE_MD_THRESHOLD_CHARS;
        },
        render: () => {
            const md = getUltraClaudeMd();
            if (!md) return null;
            return (
                <Box flexDirection="row" gap={1}>
                    <Text color="yellow">{figures.warning}</Text>
                    <Text color="yellow">
                        CLAUDE.md entries marked as IMPORTANT exceed {formatCompactNumber(ULTRA_CLAUDE_MD_THRESHOLD_CHARS)} chars ({formatCompactNumber(md.content.length)} chars)
                        <Text dimColor> • /memory to edit</Text>
                    </Text>
                </Box>
            );
        }
    },
    {
        id: "large-agent-descriptions", // V17
        type: "warning",
        isActive: (ctx) => calculateAgentDescriptionsTokens(ctx.agentDefinitions || []) > LARGE_AGENT_DESCRIPTION_THRESHOLD_TOKENS,
        render: (ctx) => {
            const tokens = calculateAgentDescriptionsTokens(ctx.agentDefinitions || []);
            return (
                <Box flexDirection="row">
                    <Text color="yellow">{figures.warning}</Text>
                    <Text color="yellow">
                        Large cumulative agent descriptions will impact performance (~{formatCompactNumber(tokens)} tokens &gt; {formatCompactNumber(LARGE_AGENT_DESCRIPTION_THRESHOLD_TOKENS)})
                        <Text dimColor> • /agents to manage</Text>
                    </Text>
                </Box>
            )
        }
    },
    {
        id: "claude-ai-external-token", // J17
        type: "warning",
        isActive: () => {
            const { source } = getAuthTokenSource();
            return isOauthEnabled() && (source === "ANTHROPIC_AUTH_TOKEN" || source === "apiKeyHelper");
        },
        render: () => {
            const { source } = getAuthTokenSource();
            return (
                <Box flexDirection="row" marginTop={1}>
                    <Text color="yellow">{figures.warning}</Text>
                    <Text color="yellow">
                        Auth conflict: Using {source} instead of Claude account subscription token. Either unset {source}, or run `claude /logout`.
                    </Text>
                </Box>
            );
        }
    },
    {
        id: "api-key-conflict", // X17
        type: "warning",
        isActive: () => {
            const { source } = getApiKeyWithSource({ skipRetrievingKeyFromApiKeyHelper: true });
            // Check if OAuth is enabled/configured? kqA()
            const oauthConfigured = true; // Stub
            return oauthConfigured && (source === "ANTHROPIC_API_KEY" || source === "apiKeyHelper");
        },
        render: () => {
            const { source } = getApiKeyWithSource({ skipRetrievingKeyFromApiKeyHelper: true });
            return (
                <Box flexDirection="row" marginTop={1}>
                    <Text color="yellow">{figures.warning}</Text>
                    <Text color="yellow">
                        Auth conflict: Using {source} instead of Anthropic Console key. Either unset {source}, or run `claude /logout`.
                    </Text>
                </Box>
            );
        }
    },
    {
        id: "both-auth-methods", // I17
        type: "warning",
        isActive: () => {
            const { source: apiSource } = getApiKeyWithSource({ skipRetrievingKeyFromApiKeyHelper: true });
            const { source: tokenSource } = getAuthTokenSource();
            return apiSource !== "none" && tokenSource !== "none" && !(apiSource === "apiKeyHelper" && tokenSource === "apiKeyHelper");
        },
        render: () => {
            const { source: apiSource } = getApiKeyWithSource({ skipRetrievingKeyFromApiKeyHelper: true });
            const { source: tokenSource } = getAuthTokenSource();
            return (
                <Box flexDirection="column" marginTop={1}>
                    <Box flexDirection="row">
                        <Text color="yellow">{figures.warning}</Text>
                        <Text color="yellow">
                            Auth conflict: Both a token ({tokenSource}) and an API key ({apiSource}) are set. This may lead to unexpected behavior.
                        </Text>
                    </Box>
                    <Box flexDirection="column" marginLeft={3}>
                        <Text color="yellow">
                            • Trying to use {tokenSource === "claude.ai" ? "claude.ai" : tokenSource}? {apiSource === "ANTHROPIC_API_KEY" ? 'Unset the ANTHROPIC_API_KEY environment variable, or claude /logout then say "No" to the API key approval before login.' : apiSource === "apiKeyHelper" ? "Unset the apiKeyHelper setting." : "claude /logout"}
                        </Text>
                        <Text color="yellow">
                            • Trying to use {apiSource}? {tokenSource === "claude.ai" ? "claude /logout to sign out of claude.ai." : `Unset the ${tokenSource} environment variable.`}
                        </Text>
                    </Box>
                </Box>
            );
        }
    },
    {
        id: "sonnet-1m-welcome", // W17
        type: "info",
        isActive: (ctx) => ctx.showSonnet1MNotice === true,
        render: () => (
            <Box flexDirection="column" marginTop={1}>
                <Text bold>
                    You now have access to Sonnet 4.5 with 1M context (uses more rate limits than Sonnet on long requests) • Update in /model
                </Text>
            </Box>
        )
    },
    {
        id: "opus-4.5-available", // K17
        type: "info",
        isActive: (ctx) => ctx.showOpus45Notice === true,
        render: () => {
            // Simplified logic vs original complex permission check
            return (
                <Box marginLeft={1}>
                    <Text dimColor>/model to try Opus 4.5</Text>
                </Box>
            );
        }
    },
    {
        id: "jetbrains-plugin-install", // H17
        type: "info",
        isActive: (ctx) => {
            if (!isJetBrainsPluginInstalled()) return false;
            // Check config autoInstallIdeExtension...
            const ide = getJetBrainsIde();
            return ide !== null && !isPluginInstalled(ide);
        },
        render: () => {
            const ide = getJetBrainsIde() || "JetBrains";
            return (
                <Box flexDirection="row" gap={1} marginLeft={1}>
                    <Text color="magenta">{figures.arrowUp}</Text>
                    <Text>
                        Install the <Text color="magenta">{ide}</Text> plugin from the JetBrains Marketplace: <Text bold>https://docs.claude.com/s/claude-code-jetbrains</Text>
                    </Text>
                </Box>
            );
        }
    }
];

// --- Notice List Component (Sr2) ---

export function NoticeList({ agentDefinitions }: { agentDefinitions?: any[] } = {}) {
    const hasLogVulnerability = useRef(false);
    // const [config] = useConfig();
    const config = {}; // Stub config

    // Logic to determine notices to show
    // In real app, we check if they've seen it via config flags
    const showSonnet1MNotice = false; // Stub
    const showOpus45Notice = false; // Stub

    const context: NoticeContext = {
        config,
        showSonnet1MNotice,
        showOpus45Notice,
        agentDefinitions
    };

    // Filter active notices
    const activeNotices = notices.filter(n => n.isActive(context));

    useEffect(() => {
        // Telemetry & State updates for "Do not show again" logic
        activeNotices.forEach(n => {
            if (n.id === "sonnet-1m-welcome") logEvent("tengu_sonnet_1m_notice_shown", {});
            if (n.id === "opus-4.5-available") logEvent("tengu_opus_45_notice_shown", {});
            if (n.id === "react-vulnerability" && !hasLogVulnerability.current) {
                hasLogVulnerability.current = true;
                logEvent("tengu_react_vulnerability_notice_shown", {});
            }
        });
    }, [activeNotices]);

    if (activeNotices.length === 0) return null;

    return (
        <Box flexDirection="column" paddingLeft={1}>
            {activeNotices.map(n => (
                <React.Fragment key={n.id}>
                    {n.render(context)}
                </React.Fragment>
            ))}
        </Box>
    );
}

// --- Thinking Animation (gr2) ---
export function useThinkingFrame(isActive: boolean) {
    const [frame, setFrame] = useState(0);
    const [index, setIndex] = useState(-1);

    useInput((input, key) => {
        if (key.escape && isActive && index === -1) {
            setIndex(0);
        }
    });

    useEffect(() => {
        if (!isActive) {
            setIndex(-1);
            setFrame(0);
            return;
        }
    }, [isActive]);

    useEffect(() => {
        if (index === -1) return;
        const sequence = [1, 0, 1, 2, 2, 1, 0, 0, 0, 1, 2, 2, 1];
        if (index >= sequence.length) {
            setIndex(-1);
            setFrame(1);
            return;
        }
        setFrame(sequence[index]);
        const t = setTimeout(() => setIndex(i => i + 1), 60);
        return () => clearTimeout(t);
    }, [index]);

    return frame;
}

// --- Mascot Logo (nU0) ---
export function MascotLogo() {
    return (
        <Box flexDirection="column" paddingRight={1}>
            <Text>
                <Text color="white"> *</Text>
                <Text color="#a1e4f5"> ▐</Text>
                <Text color="#a1e4f5" backgroundColor="#D97757">▛███▜</Text>
                <Text color="#a1e4f5">▌</Text>
                <Text color="white"> *</Text>
            </Text>
            <Text>
                <Text color="white">*</Text>
                <Text color="#a1e4f5"> ▝▜</Text>
                <Text color="#a1e4f5" backgroundColor="#D97757">█████</Text>
                <Text color="#a1e4f5">▛▘</Text>
                <Text color="white"> *</Text>
            </Text>
            <Text>
                <Text color="white"> * </Text>
                <Text color="#a1e4f5">  ▘▘ ▝▝</Text>
                <Text color="white">*</Text>
            </Text>
        </Box>
    );
}

function AppleTerminalMascot() {
    return (
        <Box flexDirection="column" alignItems="center" paddingRight={1}>
            <Text>
                <Text color="white"> * </Text>
                <Text color="#a1e4f5">▗</Text>
                <Text color="#D97757" backgroundColor="#a1e4f5"> ▗   ▖ </Text>
                <Text color="#a1e4f5">▖</Text>
                <Text color="white"> *</Text>
            </Text>
            <Text>
                <Text color="white">*   </Text>
                <Text backgroundColor="#a1e4f5">       </Text>
                <Text color="white">   *</Text>
            </Text>
            <Text>
                <Text color="white"> * </Text>
                <Text color="#a1e4f5">  ▘▘ ▝▝  </Text>
                <Text color="white"> *</Text>
            </Text>
        </Box>
    );
}

// --- Changelog Parser (MK1) ---
export function parseChangelog(markdown: string) {
    if (!markdown) return {};
    try {
        const versions: Record<string, string[]> = {};
        // Split by H2 headers (## Version)
        const parts = markdown.split(/^## /gm).slice(1);
        for (const part of parts) {
            const lines = part.trim().split('\n');
            if (lines.length === 0) continue;

            // Extract version from first line (e.g., "0.2.14 - 2024-05-10")
            const header = lines[0];
            const version = header.split(" - ")[0]?.trim();
            if (!version) continue;

            // Extract bullet points
            const bullets = lines.slice(1)
                .filter(l => l.trim().startsWith("- "))
                .map(l => l.trim().substring(2).trim())
                .filter(Boolean);

            if (bullets.length > 0) {
                versions[version] = bullets;
            }
        }
        return versions;
    } catch (e) {
        console.error("Failed to parse changelog", e);
        return {};
    }
}

// --- Dashboard Utilities (PK1, nr2, OkA) ---

export interface DashboardHeaderInfo {
    version: string;
    cwd: string;
    modelDisplayName: string;
    billingType: string;
    agentName: string;
}

export function getHeaderInfo(): DashboardHeaderInfo {
    const version = process.env.npm_package_version || "2.0.76"; // Fallback to known version
    const cwd = process.cwd();
    // Stubbed logic for model/billing - normally from AppState or Config
    const modelDisplayName = process.env.CLAUDE_CODE_MODEL || "Claude 3.5 Sonnet";
    const billingType = "API Usage Billing";
    const agentName = "Claude";

    return {
        version,
        cwd: truncatePath(cwd, 40),
        modelDisplayName,
        billingType,
        agentName
    };
}


import { SessionPersistence } from "../../services/session/SessionHistory.js";

/**
 * Truncates a path for display (OkA).
 */
export function truncatePath(path: string, maxLength: number): string {
    if (path.length <= maxLength) return path;
    const separator = "/";
    const ellipsis = "…";
    const parts = path.split(separator);

    const first = parts[0] || "";
    const last = parts[parts.length - 1] || "";

    // If just filename is too long (or no separators found)
    if (parts.length === 1) {
        return path.substring(0, maxLength - ellipsis.length) + ellipsis;
    }

    // Try to handle cases where preserving first/last is hard
    if (first === "" && ellipsis.length + separator.length + last.length >= maxLength) {
        // Starts with separator (absolute path), but barely enough room
        return `${separator}${last.substring(0, maxLength - ellipsis.length - separator.length)}${ellipsis}`;
    }

    if (first !== "" && ellipsis.length * 2 + separator.length + last.length >= maxLength) {
        // Relative path, limited room
        return `${ellipsis}${separator}${last.substring(0, maxLength - ellipsis.length * 2 - separator.length)}${ellipsis}`;
    }

    if (parts.length === 2) {
        // only 2 parts, e.g. "dir/file"
        return `${first.substring(0, maxLength - ellipsis.length - separator.length - last.length)}${ellipsis}${separator}${last}`;
    }

    let remainingLength = maxLength - first.length - last.length - ellipsis.length - 2 * separator.length;

    // If we can't even fit first and last with ellipsis
    if (remainingLength <= 0) {
        return `${first.substring(0, Math.max(0, maxLength - last.length - ellipsis.length - 2 * separator.length))}${separator}${ellipsis}${separator}${last}`;
    }

    const middleParts: string[] = [];
    // Try to include end parts (folders closer to filename)
    for (let i = parts.length - 2; i > 0; i--) {
        const part = parts[i];
        if (part && part.length + separator.length <= remainingLength) {
            middleParts.unshift(part);
            remainingLength -= part.length + separator.length;
        } else {
            break;
        }
    }

    if (middleParts.length === 0) {
        return `${first}${separator}${ellipsis}${separator}${last}`;
    }

    return `${first}${separator}${ellipsis}${separator}${middleParts.join(separator)}${separator}${last}`;
}

let tasksCache: any[] | null = null;

export async function getRecentTasks(): Promise<any[]> {
    if (tasksCache) return tasksCache;

    // Check if we can get the current leaf UUID/Session ID to exclude it
    // Using a placeholder or environmental check if available, otherwise assume undefined (show all)
    // In chunk_509, h0() does this.
    // We'll read from process.env if available, or just ignore exclusion for now if service not fully hooked up.
    const currentLeafUuid = process.env.CLAUDE_CODE_SESSION_ID;

    try {
        const persistence = SessionPersistence.getInstance();
        const allSessions = await persistence.listLocalSessions() as any[];

        // Sorting usually happens in listLocalSessions, but let's ensure desc timestamp if needed
        // Assuming listLocalSessions returns recent first.

        tasksCache = allSessions
            .filter(session => {
                // Filter logic from chunk_509
                if (session.isSidechain) return false;
                if (currentLeafUuid && session.leafUuid === currentLeafUuid) return false;
                if (session.summary && session.summary.includes("I apologize")) return false;

                const hasSummary = session.summary && session.summary !== "No prompt";
                const hasPrompt = session.firstPrompt && session.firstPrompt !== "No prompt";

                return hasSummary || hasPrompt;
            })
            .slice(0, 3);

        return tasksCache;
    } catch (err) {
        console.error("Failed to fetch recent tasks", err);
        tasksCache = [];
        return tasksCache;
    }
}
