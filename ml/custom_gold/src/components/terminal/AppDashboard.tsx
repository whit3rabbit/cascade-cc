// Logic from chunk_510.ts (App Dashboard & Info Feeds)

import React, { useEffect, useMemo, useState } from "react";
import { Box, Text } from "ink";
import { MascotLogo, useThinkingFrame } from "./NoticeList.js";
import { figures } from "../../vendor/terminalFigures.js";

type FeedLine = {
    text: string;
    timestamp?: string;
};

type FeedConfig = {
    title: string;
    lines: FeedLine[];
    footer?: string;
    emptyMessage?: string;
    customContent?: {
        content: React.ReactNode;
        width: number;
    };
};

type TipInfo = {
    tip: string;
    color: "dim" | "warning" | "error";
};

type BuildInfo = {
    VERSION: string;
    ISSUES_EXPLAINER: string;
    PACKAGE_URL: string;
    README_URL: string;
    FEEDBACK_CHANNEL: string;
    BUILD_TIME: string;
};

const BUILD_INFO: BuildInfo = {
    ISSUES_EXPLAINER: "report the issue at https://github.com/anthropics/claude-code/issues",
    PACKAGE_URL: "@anthropic-ai/claude-code",
    README_URL: "https://code.claude.com/docs/en/overview",
    VERSION: "2.0.76",
    FEEDBACK_CHANNEL: "https://github.com/anthropics/claude-code/issues",
    BUILD_TIME: "2025-12-22T23:56:12Z"
};

const TOP_OF_FEED_TIP_KEY = "tengu-top-of-feed-tip";
const FEED_MAX_WIDTH = 50;
const PASS_ELIGIBILITY_CACHE_TTL_MS = 60 * 60 * 1000;

function wrapToWidth(text: string, width: number): string {
    if (!text) return "";
    if (width <= 0) return "";
    if (text.length <= width) return text;
    return text.slice(0, Math.max(width - 1, 0));
}

function truncatePath(path: string, width: number): string {
    if (path.length <= width) return path;
    if (width <= 1) return path.slice(0, width);
    const ellipsis = "…";
    const keep = Math.max(width - ellipsis.length, 0);
    return `${path.slice(0, keep)}${ellipsis}`;
}

function getTerminalColumns(): number {
    return process.stdout?.columns ?? 80;
}

function getTerminalInfo() {
    return {
        terminal: process.env.TERM_PROGRAM ?? "",
        isAppleTerminal: process.env.TERM_PROGRAM === "Apple_Terminal"
    };
}

function getTheme() {
    return "default";
}



const TIPS = [
    "Use /help to see all available commands.",
    "Use /compact to reduce conversation context size.",
    "Use /cost to track your token usage and costs.",
    "Use /doctor to diagnose installation issues.",
    "Use Ctrl+C to interrupt Claude's response.",
    "Use Ctrl+R to search your command history.",
    "Use /clear to clear the terminal screen."
];

function selectTip(_key: string, _fallback: TipInfo): TipInfo {
    const tip = TIPS[Math.floor(Math.random() * TIPS.length)];
    return { tip, color: "dim" };
}

function useInterval(callback: () => void, delay: number | null) {
    useEffect(() => {
        if (delay === null) return;
        const interval = setInterval(callback, delay);
        return () => clearInterval(interval);
    }, [callback, delay]);
}

function isDebugModeEnabled(): boolean {
    return Boolean(process.env.CLAUDE_CODE_DEBUG);
}

function getLogTarget(): string {
    return "stdout";
}

function isLogToStderr(): boolean {
    return false;
}

function isTruthyEnv(value?: string): boolean {
    if (!value) return false;
    return ["1", "true", "yes", "on"].includes(value.toLowerCase());
}

function isSandboxingEnabled(): boolean {
    return false;
}

function getSettings() {
    return {
        oauthAccount: {
            displayName: "User",
            organizationName: "Personal"
        },
        passesEligibilityCache: {} as Record<string, any>,
        passesUpsellSeenCount: 0,
        hasVisitedPasses: false,
        theme: "default",
        numStartups: 1,
        lastReleaseNotesSeen: ""
    };
}

function updateSettings(updater: (settings: ReturnType<typeof getSettings>) => any) {
    void updater;
}

function logInfo(message: string) {
    void message;
}

function logError(error: unknown) {
    void error;
}

function trackEvent(_name: string, _payload?: Record<string, any>) {
    void _name;
    void _payload;
}

function formatTimestamp(epochMs: number): string {
    const date = new Date(epochMs);
    return date.toLocaleDateString();
}

function getAppHeaderInfo() {
    return {
        version: BUILD_INFO.VERSION,
        cwd: process.cwd(),
        modelDisplayName: "Claude 3.5 Sonnet",
        billingType: "Free",
        agentName: ""
    };
}

function getWelcomeTitle(name: string | null): string {
    if (!name) return "Welcome to Claude Code";
    return `Welcome, ${name}`;
}

function getLayoutKind(columns: number): "compact" | "horizontal" | "vertical" {
    if (columns < 70) return "compact";
    if (columns < 120) return "vertical";
    return "horizontal";
}

function truncateModelAndBilling(model: string, billing: string, width: number) {
    const combined = `${model} · ${billing}`;
    if (combined.length <= width) {
        return { shouldSplit: false, truncatedModel: model, truncatedBilling: billing };
    }
    return {
        shouldSplit: true,
        truncatedModel: wrapToWidth(model, width),
        truncatedBilling: wrapToWidth(billing, width)
    };
}

function getHeaderMeasurements(title: string, subLine: string, metaLine: string) {
    return {
        titleWidth: title.length,
        subLineWidth: subLine.length,
        metaLineWidth: metaLine.length
    };
}

function computeLayoutWidths(columns: number, layout: string, header: ReturnType<typeof getHeaderMeasurements>) {
    if (layout === "horizontal") {
        const leftMin = Math.max(header.titleWidth, header.subLineWidth, 30);
        const leftWidth = Math.min(Math.floor(columns * 0.45), Math.max(30, leftMin));
        const rightWidth = Math.max(columns - leftWidth - 2, 20);
        return { leftWidth, rightWidth };
    }

    return { leftWidth: columns, rightWidth: columns };
}

// --- Width calculator (tr2) ---
export function calculateFeedWidth(config: FeedConfig): number {
    const { title, lines, footer, emptyMessage, customContent } = config;
    let width = title.length;
    if (customContent !== undefined) width = Math.max(width, customContent.width);
    else if (lines.length === 0 && emptyMessage) width = Math.max(width, emptyMessage.length);
    else {
        const timestampWidth = Math.max(0, ...lines.map((line) => (line.timestamp ? line.timestamp.length : 0)));
        for (const line of lines) {
            const timePadding = timestampWidth > 0 ? timestampWidth + 2 : 0;
            width = Math.max(width, line.text.length + timePadding);
        }
    }
    if (footer) width = Math.max(width, footer.length);
    return width;
}

// --- Info Feed (er2) ---
export function InfoFeedView({ config, actualWidth }: { config: FeedConfig; actualWidth: number }) {
    const { title, lines, footer, emptyMessage, customContent } = config;
    const padding = "  ";
    const timestampWidth = Math.max(0, ...lines.map((line) => (line.timestamp ? line.timestamp.length : 0)));

    return (
        <Box flexDirection="column" width={actualWidth}>
            <Text bold color="claude">
                {title}
            </Text>
            {customContent ? (
                <>
                    {customContent.content}
                    {footer && (
                        <Text dimColor italic>
                            {wrapToWidth(footer, actualWidth)}
                        </Text>
                    )}
                </>
            ) : lines.length === 0 && emptyMessage ? (
                <Text dimColor>{wrapToWidth(emptyMessage, actualWidth)}</Text>
            ) : (
                <>
                    {lines.map((line, index) => {
                        const available = Math.max(10, actualWidth - (timestampWidth > 0 ? timestampWidth + 2 : 0));
                        return (
                            <Text key={index}>
                                {timestampWidth > 0 && (
                                    <>
                                        <Text dimColor>{(line.timestamp || "").padEnd(timestampWidth)}</Text>
                                        {padding}
                                    </>
                                )}
                                <Text>{wrapToWidth(line.text, available)}</Text>
                            </Text>
                        );
                    })}
                    {footer && (
                        <Text dimColor italic>
                            {wrapToWidth(footer, actualWidth)}
                        </Text>
                    )}
                </>
            )}
        </Box>
    );
}

// --- Info Feed Grid (Qs2) ---
export function InfoFeedGrid({ feeds, maxWidth }: { feeds: FeedConfig[]; maxWidth: number }) {
    const widths = feeds.map((feed) => calculateFeedWidth(feed));
    const widest = Math.max(...widths);
    const actualWidth = Math.min(widest, maxWidth);

    return (
        <Box flexDirection="column">
            {feeds.map((feed, index) => (
                <React.Fragment key={index}>
                    <InfoFeedView config={feed} actualWidth={actualWidth} />
                    {index < feeds.length - 1 && (
                        <Box>
                            <Text color="claude">{"-".repeat(Math.max(1, actualWidth))}</Text>
                        </Box>
                    )}
                </React.Fragment>
            ))}
        </Box>
    );
}

// --- Feed Data (xK1) ---
export function createRecentActivityFeed(recentActivity: { summary?: string; firstPrompt?: string; modified: number }[]) {
    const lines = recentActivity.map((activity) => {
        const timestamp = formatTimestamp(activity.modified);
        return {
            text: (activity.summary && activity.summary !== "No prompt" ? activity.summary : activity.firstPrompt) || "",
            timestamp
        };
    });

    return {
        title: "Recent activity",
        lines,
        footer: lines.length > 0 ? "/resume for more" : undefined,
        emptyMessage: "No recent activity"
    } as FeedConfig;
}

// --- Feed Data (Gs2) ---
export function createNewFeaturesFeed(items: string[]) {
    const lines = items.map((text) => ({ text }));
    return {
        title: "What's new",
        lines,
        footer: lines.length > 0 ? "/release-notes for more" : undefined,
        emptyMessage: "Check the Claude Code changelog for updates"
    } as FeedConfig;
}

// --- Feed Data (Zs2) ---
export function createGettingStartedFeed(
    tips: { text: string; isEnabled: boolean; isComplete: boolean }[]
) {
    const lines = tips
        .filter(({ isEnabled }) => isEnabled)
        .sort((a, b) => Number(a.isComplete) - Number(b.isComplete))
        .map(({ text, isComplete }) => ({
            text: `${isComplete ? `${figures.tick} ` : ""}${text}`
        }));

    const shouldWarnHomeDir = process.cwd() === (process.env.HOME || "");
    const note = shouldWarnHomeDir
        ? "Note: You have launched claude in your home directory. For the best experience, launch it in a project directory instead."
        : undefined;
    if (note) lines.push({ text: note });

    return {
        title: "Tips for getting started",
        lines
    } as FeedConfig;
}

function createGuestPassFeed(): FeedConfig {
    return {
        title: "3 guest passes",
        lines: [],
        customContent: {
            content: (
                <>
                    <Box marginY={1}>
                        <Text color="claude">[✻] [✻] [✻]</Text>
                    </Box>
                    <Text dimColor>Share Claude Code with friends</Text>
                </>
            ),
            width: 30
        },
        footer: "/passes"
    };
}

async function fetchPassEligibility(campaign = "claude_code_guest_pass") {
    void campaign;
    return { eligible: false };
}

async function fetchPassRedemptions(campaign = "claude_code_guest_pass") {
    void campaign;
    return { redeemed: false };
}
void fetchPassRedemptions;

function canCheckPassEligibility(): boolean {
    return false;
}

function getCachedPassEligibility() {
    if (!canCheckPassEligibility()) {
        return { eligible: false, needsRefresh: false, hasCache: false };
    }
    const orgId = getSettings().oauthAccount?.organizationName;
    if (!orgId) {
        return { eligible: false, needsRefresh: false, hasCache: false };
    }
    const cached = getSettings().passesEligibilityCache?.[orgId];
    if (!cached) {
        return { eligible: false, needsRefresh: true, hasCache: false };
    }

    const { eligible, timestamp } = cached;
    const needsRefresh = Date.now() - timestamp > PASS_ELIGIBILITY_CACHE_TTL_MS;
    return { eligible, needsRefresh, hasCache: true };
}

let eligibilityFetchPromise: Promise<any> | null = null;
async function refreshPassEligibilityCache() {
    if (eligibilityFetchPromise) {
        logInfo("Passes: Reusing in-flight eligibility fetch");
        return eligibilityFetchPromise;
    }
    const orgId = getSettings().oauthAccount?.organizationName;
    if (!orgId) return null;

    eligibilityFetchPromise = (async () => {
        try {
            const result = await fetchPassEligibility();
            const cacheEntry = { ...result, timestamp: Date.now() };
            updateSettings((state) => ({
                ...state,
                passesEligibilityCache: {
                    ...state.passesEligibilityCache,
                    [orgId]: cacheEntry
                }
            }));
            logInfo(`Passes eligibility cached for org ${orgId}: ${result.eligible}`);
            return result;
        } catch (err) {
            logInfo("Failed to fetch and cache passes eligibility");
            logError(err);
            return null;
        } finally {
            eligibilityFetchPromise = null;
        }
    })();

    return eligibilityFetchPromise;
}

// --- Passes eligibility (RkA) ---
export async function checkPassEligibility() {
    if (!canCheckPassEligibility()) return null;
    const orgId = getSettings().oauthAccount?.organizationName;
    if (!orgId) return null;
    const cached = getSettings().passesEligibilityCache?.[orgId];
    const now = Date.now();
    if (!cached) {
        logInfo("Passes: No cache, fetching eligibility");
        return await refreshPassEligibilityCache();
    }
    if (now - cached.timestamp > PASS_ELIGIBILITY_CACHE_TTL_MS) {
        logInfo("Passes: Cache stale, returning cached data and refreshing in background");
        refreshPassEligibilityCache();
        const { timestamp: _timestamp, ...rest } = cached;
        return rest;
    }

    logInfo("Passes: Using fresh cached eligibility data");
    const { timestamp: _timestamp, ...rest } = cached;
    return rest;
}

async function primePassEligibilityCache() {
    void checkPassEligibility();
}

function shouldShowPassesUpsell() {
    const settings = getSettings();
    const { eligible, hasCache } = getCachedPassEligibility();
    if (!eligible || !hasCache) return false;
    if ((settings.passesUpsellSeenCount ?? 0) >= 3) return false;
    if (settings.hasVisitedPasses) return false;
    return true;
}

function usePassesUpsell() {
    const [visible] = useState(() => shouldShowPassesUpsell());
    return visible;
}

function trackPassesUpsellShown() {
    const nextCount = (getSettings().passesUpsellSeenCount ?? 0) + 1;
    updateSettings((state) => ({
        ...state,
        passesUpsellSeenCount: nextCount
    }));
    trackEvent("tengu_guest_passes_upsell_shown", { seen_count: nextCount });
}

function GuestPassesUpsellBanner() {
    return (
        <Text dimColor>
            <Text color="claude">[✻]</Text> <Text color="claude">[✻]</Text> <Text color="claude">[✻]</Text> · 3 guest
            passes at /passes
        </Text>
    );
}

function ClawdLogo() {
    return (
        <Box flexDirection="column" paddingRight={1}>
            <Text>
                <Text color="text"> *</Text>
                <Text color="ice_blue"> ▐</Text>
                <Text color="ice_blue" backgroundColor="clawd_background">
                    ▛███▜
                </Text>
                <Text color="ice_blue">▌</Text>
                <Text color="text"> *</Text>
            </Text>
            <Text>
                <Text color="text">*</Text>
                <Text color="ice_blue"> ▝▜</Text>
                <Text color="ice_blue" backgroundColor="clawd_background">
                    █████
                </Text>
                <Text color="ice_blue">▛▘</Text>
                <Text color="text"> *</Text>
            </Text>
            <Text>
                <Text color="text"> * </Text>
                <Text color="ice_blue"> ▘▘ ▝▝  </Text>
                <Text color="text">*</Text>
            </Text>
        </Box>
    );
}

// --- Minimal header (Es2) ---
export function MinimalAppHeader() {
    const columns = getTerminalColumns();
    const { version, cwd, modelDisplayName, billingType, agentName } = getAppHeaderInfo();
    const activeAgent = agentName;
    const showUpsell = usePassesUpsell();

    useEffect(() => {
        if (showUpsell) trackPassesUpsellShown();
    }, [showUpsell]);

    const width = Math.max(columns - 15, 20);
    const versionText = wrapToWidth(version, Math.max(width - "Claude Code v".length, 6));
    const { shouldSplit, truncatedModel, truncatedBilling } = truncateModelAndBilling(
        modelDisplayName,
        billingType,
        width
    );
    const separator = " · ";
    const remaining = activeAgent ? width - activeAgent.length - separator.length : width;
    const truncatedPath = truncatePath(cwd, Math.max(remaining, 10));

    return (
        <Box flexDirection="row" gap={3} paddingY={1}>
            <ClawdLogo />
            <Box flexDirection="column">
                <Text>
                    <Text bold>Claude Code</Text> <Text dimColor>v{versionText}</Text>
                </Text>
                {shouldSplit ? (
                    <>
                        <Text dimColor>{truncatedModel}</Text>
                        <Text dimColor>{truncatedBilling}</Text>
                    </>
                ) : (
                    <Text dimColor>
                        {truncatedModel} · {truncatedBilling}
                    </Text>
                )}
                <Text dimColor>{activeAgent ? `${activeAgent} · ${truncatedPath}` : truncatedPath}</Text>
                {showUpsell && <GuestPassesUpsellBanner />}
            </Box>
        </Box>
    );
}

function getTopOfFeedTip(): TipInfo {
    return selectTip(TOP_OF_FEED_TIP_KEY, { tip: "Use /help to see all available commands.", color: "dim" });
}

function TopOfFeedTip() {
    const tip = useMemo(() => getTopOfFeedTip(), []);

    useEffect(() => {
        if (tip.tip) {
            void tip;
        }
    }, [tip.tip]);

    if (!tip.tip) return null;

    return (
        <Box paddingLeft={2} flexDirection="column">
            <Text
                {...(tip.color === "warning"
                    ? { color: "warning" }
                    : tip.color === "error"
                        ? { color: "error" }
                        : { dimColor: true })}
            >
                {tip.tip}
            </Text>
        </Box>
    );
}

function shouldShowYearEndPromo() {
    return false;
}

function YearEndPromoView() {
    return (
        <Box flexDirection="column">
            <Text color="claude">A gift for you</Text>
            <Text dimColor>Your rate limits are 2x higher through 12/31. Enjoy the extra room to think!</Text>
        </Box>
    );
}

function getAnnouncements() {
    return {
        companyAnnouncements: [] as string[]
    };
}

function getReleaseNotesStatus(_lastSeen: string) {
    return { hasReleaseNotes: true };
}

function getRecentActivity(): { summary?: string; firstPrompt?: string; modified: number }[] {
    return [];
}

function getReleaseNotes(limit: number): string[] {
    void limit;
    return ["Added support for Claude 3.5 Sonnet", "Improved MCP tool handling", "Fixed various bugs"];
}

function getGettingStartedTips(): { text: string; isEnabled: boolean; isComplete: boolean }[] {
    return [
        { text: "Run claude in a git repository", isEnabled: true, isComplete: true },
        { text: "Use /init to set up your project", isEnabled: true, isComplete: false },
        { text: "Try asking 'Who are the top contributors?'", isEnabled: true, isComplete: false }
    ];
}

// --- App Dashboard (qs2) ---
export function AppDashboard({ isBeforeFirstMessage }: { isBeforeFirstMessage: boolean }) {
    const thinkingFrame = useThinkingFrame(isBeforeFirstMessage);
    const recentActivity = getRecentActivity();
    const displayName = getSettings().oauthAccount?.displayName ?? "";
    const releaseNotes = getReleaseNotes(3);
    const columns = getTerminalColumns();
    const showGettingStarted = isBeforeFirstMessage;
    const sandboxed = isSandboxingEnabled();
    const showPassesUpsell = usePassesUpsell();
    const showYearEnd = shouldShowYearEndPromo();
    const announcements = getAnnouncements();
    const settings = getSettings();
    const organizationName = settings.oauthAccount?.organizationName;
    const companyAnnouncements = announcements.companyAnnouncements;
    const [announcement] = useState(() =>
        companyAnnouncements && companyAnnouncements.length > 0
            ? settings.numStartups === 1
                ? companyAnnouncements[0]
                : companyAnnouncements[Math.floor(Math.random() * companyAnnouncements.length)]
            : undefined
    );
    const { hasReleaseNotes } = getReleaseNotesStatus(settings.lastReleaseNotesSeen);

    useEffect(() => {
        if (settings.lastReleaseNotesSeen === BUILD_INFO.VERSION) return;
        updateSettings((prev) => {
            if (prev.lastReleaseNotesSeen === BUILD_INFO.VERSION) return prev;
            return { ...prev, lastReleaseNotesSeen: BUILD_INFO.VERSION };
        });
        if (showGettingStarted) {
            primePassEligibilityCache();
        }
    }, [settings, showGettingStarted]);

    useEffect(() => {
        if (showPassesUpsell && !showGettingStarted) trackPassesUpsellShown();
    }, [showPassesUpsell, showGettingStarted]);

    const { cwd, modelDisplayName, billingType, agentName } = getAppHeaderInfo();
    const activeAgent = agentName;
    const truncatedModel = wrapToWidth(modelDisplayName, FEED_MAX_WIDTH - 20);

    if (!hasReleaseNotes && !showGettingStarted && !isTruthyEnv(process.env.CLAUDE_CODE_FORCE_FULL_LOGO)) {
        return (
            <>
                <Box />
                <MinimalAppHeader />
                {isDebugModeEnabled() && (
                    <Box paddingLeft={2} flexDirection="column">
                        <Text color="warning">Debug mode enabled</Text>
                        <Text dimColor>Logging to: {isLogToStderr() ? "stderr" : getLogTarget()}</Text>
                    </Box>
                )}
                <TopOfFeedTip />
                {showYearEnd && (
                    <Box paddingLeft={2}>
                        <YearEndPromoView />
                    </Box>
                )}
                {announcement && (
                    <Box paddingLeft={2} flexDirection="column">
                        {organizationName && <Text dimColor>Message from {organizationName}:</Text>}
                        <Text>{announcement}</Text>
                    </Box>
                )}
            </>
        );
    }

    const layout = getLayoutKind(columns);

    if (layout === "compact") {
        let title = getWelcomeTitle(displayName);
        if (title.length > columns - 4) title = getWelcomeTitle(null);
        const separator = " · ";
        const remaining = activeAgent ? columns - 4 - activeAgent.length - separator.length : columns - 4;
        const truncatedPath = truncatePath(cwd, Math.max(remaining, 10));

        return (
            <>
                <Box
                    flexDirection="column"
                    borderStyle="round"
                    borderColor="claude"
                    /* borderText={{
                        content: borderLabel,
                        position: "top",
                        align: "start",
                        offset: 1
                    }} */
                    paddingX={1}
                    paddingY={1}
                    alignItems="center"
                    width={columns}
                >
                    <Text bold>{title}</Text>
                    <Box marginY={1}>
                        <Box height={5} flexDirection="column" justifyContent="flex-end">
                            <Box marginBottom={thinkingFrame}>
                                <MascotLogo />
                            </Box>
                        </Box>
                    </Box>
                    <Text dimColor>{truncatedModel}</Text>
                    <Text dimColor>{billingType}</Text>
                    <Text dimColor>{activeAgent ? `${activeAgent} · ${truncatedPath}` : truncatedPath}</Text>
                </Box>
                {sandboxed && (
                    <Box marginTop={1} flexDirection="column">
                        <Text color="warning">Your bash commands will be sandboxed. Disable with /sandbox.</Text>
                    </Box>
                )}
            </>
        );
    }

    const welcomeTitle = getWelcomeTitle(displayName);
    const metaLine = organizationName ? `${truncatedModel} · ${billingType} · ${organizationName}` : `${truncatedModel} · ${billingType}`;
    const separator = " · ";
    const remaining = activeAgent ? FEED_MAX_WIDTH - activeAgent.length - separator.length : FEED_MAX_WIDTH;
    const truncatedPath = truncatePath(cwd, Math.max(remaining, 10));
    const agentLine = activeAgent ? `${activeAgent} · ${truncatedPath}` : truncatedPath;
    const headerMeasurements = getHeaderMeasurements(welcomeTitle, agentLine, metaLine);
    const { leftWidth, rightWidth } = computeLayoutWidths(columns, layout, headerMeasurements);

    return (
        <Box flexDirection="column" paddingX={1}>
            <MinimalAppHeader />
            {isDebugModeEnabled() && (
                <Box paddingLeft={2} flexDirection="column">
                    <Text color="warning">Debug mode enabled</Text>
                    <Text dimColor>Logging to: {isLogToStderr() ? "stderr" : getLogTarget()}</Text>
                </Box>
            )}
            <TopOfFeedTip />
            {showYearEnd && (
                <Box paddingLeft={2}>
                    <YearEndPromoView />
                </Box>
            )}
            {announcement && (
                <Box paddingLeft={2} flexDirection="column">
                    {organizationName && <Text dimColor>Message from {organizationName}:</Text>}
                    <Text>{announcement}</Text>
                </Box>
            )}
            {sandboxed && (
                <Box paddingLeft={2} flexDirection="column">
                    <Text color="warning">Your bash commands will be sandboxed. Disable with /sandbox.</Text>
                </Box>
            )}
        </Box>
    );
}

// --- Side Conversation (tU0) ---
export function SideConversationView({ question, response }: { question: string; response?: string }) {
    const [frame, setFrame] = useState(0);
    useInterval(() => setFrame((value) => value + 1), response ? null : 80);

    return (
        <Box flexDirection="column" paddingLeft={2} marginTop={1}>
            <Box>
                <Text color="warning" bold>
                    btw{" "}
                </Text>
                <Text dimColor>{question}</Text>
            </Box>
            <Box marginTop={1} marginLeft={2}>
                {response ? (
                    <Text>{response}</Text>
                ) : (
                    <Box>
                        <Spinner frame={frame} messageColor="warning" />
                        <Text color="warning">Answering...</Text>
                    </Box>
                )}
            </Box>
        </Box>
    );
}

function Spinner({ frame, messageColor }: { frame: number; messageColor: string }) {
    const frames = [".", "..", "..."];
    return (
        <Text color={messageColor}>
            {frames[frame % frames.length]}{" "}
        </Text>
    );
}

// --- Message Unresolved (T17) ---
// --- Message Unresolved (T17) ---
export function isMessageUnresolved(
    message: any,
    streamingIds: Set<string>,
    resolvedToolUseIds: Set<string>,
    inProgressToolUseIDs: Set<string>,
    resolvedIdsForMessage: Set<string>,
    view: string,
    _progressMap: any
) {
    if (view === "transcript") return true;
    switch (message.type) {
        case "attachment":
        case "user":
        case "assistant": {
            const toolUseId = message.uuid; // Was message.id, usually uuid for items
            if (!toolUseId) return true;
            // Simplified logic based on provided sets
            if (streamingIds.has(toolUseId)) return false;
            // Check if processed
            return !resolvedToolUseIds.has(toolUseId);
        }
        case "system":
            return message.subtype !== "api_error";
        case "grouped_tool_use":
            // Check if all tools in group are resolved
            return message.messages?.every((entry: any) => {
                const content = entry.message.content[0];
                return content?.type === "tool_use" && resolvedIdsForMessage.has(content.id);
            });
        case "collapsed_read_search":
            return false;
        default:
            return false;
    }
}
