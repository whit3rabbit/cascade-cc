import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Box, Text } from "ink";
import { PermissionSelect } from "../permissions/PermissionComponents.js";
import { useNotifications } from "../../services/terminal/NotificationService.js";
import { useAppState } from "../../contexts/AppStateContext.js";
import { sandboxService, isSandboxingEnabled } from "../../services/sandbox/sandboxService.js";
import { execFile } from "child_process";
import { promisify } from "util";
import { extname } from "path";
import { getSettings, updateSettings } from "../../services/terminal/settings.js";
import { installPlugin } from "../../services/terminal/PluginCliService.js";

type MarketplaceEntry = {
    name: string;
    description?: string;
    lspServers?: Record<string, any> | Array<Record<string, any> | string> | string;
    source?: string;
};

type MarketplaceIndexEntry = {
    entry: MarketplaceEntry;
    marketplaceName: string;
    extensions: Set<string>;
    command: string;
    isOfficial: boolean;
};

type LspRecommendation = {
    pluginId: string;
    pluginName: string;
    marketplaceName: string;
    description?: string;
    isOfficial: boolean;
    extensions: string[];
    command: string;
};

type LspPromptResponse = "yes" | "no" | "never" | "disable";

const execFileAsync = promisify(execFile);
const SETTINGS_ERROR_KEY = "settings-errors";
const LSP_RECOMMENDATION_TIMEOUT_MS = 30000;
const LSP_RECOMMENDATION_IGNORE_THRESHOLD = 5;
const LSP_RECOMMENDATION_TIMEOUT_IGNORE_MS = 28000;
const LSP_ERROR_POLL_INTERVAL_MS = 5000;

function logInfo(message: string) {
    console.log(message);
}

function logError(error: unknown) {
    console.error(error);
}

function formatTime(timestamp: Date) {
    return timestamp.toLocaleTimeString("en-US", {
        hour: "numeric",
        minute: "2-digit",
        second: "2-digit"
    });
}

function getSettingsValidationErrors(): { errors: Array<{ path?: string; message?: string }> } {
    return { errors: [] };
}

function subscribeToSettingsErrors(_callback: () => void): () => void {
    return () => { };
}

function getMarketplaceConfig(): Record<string, any> {
    return {};
}

async function loadMarketplace(_name: string): Promise<{ plugins: MarketplaceEntry[] }> {
    return { plugins: [] };
}

function isOfficialMarketplace(name: string) {
    return name === "official" || name === "anthropics/claude-code-plugins";
}

function isPluginInstalled(pluginId: string) {
    const settings = getSettings("userSettings") as any;
    return Boolean(settings?.enabledPlugins?.[pluginId]);
}

function shouldDisableLspRecommendations() {
    const settings = getSettings("userSettings") as any;
    const ignoredCount = settings?.lspRecommendationIgnoredCount ?? 0;
    return settings?.lspRecommendationDisabled === true || ignoredCount >= LSP_RECOMMENDATION_IGNORE_THRESHOLD;
}

function updateUserSettings(updater: (current: any) => any) {
    const current = (getSettings("userSettings") as any) ?? {};
    const next = updater(current);
    updateSettings("userSettings", next);
}

function addNeverSuggestPlugin(pluginId: string) {
    updateUserSettings((current) => {
        const existing = current.lspRecommendationNeverPlugins ?? [];
        if (existing.includes(pluginId)) return current;
        return {
            ...current,
            lspRecommendationNeverPlugins: [...existing, pluginId]
        };
    });
    logInfo(`[lspRecommendation] Added ${pluginId} to never suggest`);
}

function incrementIgnoredCount() {
    updateUserSettings((current) => ({
        ...current,
        lspRecommendationIgnoredCount: (current.lspRecommendationIgnoredCount ?? 0) + 1
    }));
    logInfo("[lspRecommendation] Incremented ignored count");
}

function disableLspRecommendations() {
    updateUserSettings((current) => ({
        ...current,
        lspRecommendationDisabled: true
    }));
}

// --- Sandbox Violation View (q29) ---
export function SandboxViolationView() {
    const [violations, setViolations] = useState<any[]>([]);
    const [totalCount, setTotalCount] = useState(0);

    useEffect(() => {
        const store = sandboxService.getViolationStore();
        const handleUpdate = () => {
            setViolations(store.getViolations(10));
            setTotalCount(store.getTotalCount());
        };
        store.on("update", handleUpdate);
        handleUpdate();
        return () => {
            store.off("update", handleUpdate);
        };
    }, []);

    if (!isSandboxingEnabled() || (process.platform as string) === "linux") return null;
    if (totalCount === 0) return null;

    return (
        <Box flexDirection="column" marginTop={1}>
            <Box marginLeft={0}>
                <Text color="permission">
                    ⧈ Sandbox blocked {totalCount} total {totalCount === 1 ? "operation" : "operations"}
                </Text>
            </Box>
            {violations.map((violation, index) => (
                <Box key={`${violation.timestamp?.getTime?.() ?? index}-${index}`} paddingLeft={2}>
                    <Text dimColor>
                        {formatTime(violation.timestamp)}{violation.command ? ` ${violation.command}:` : ""} {violation.line}
                    </Text>
                </Box>
            ))}
            <Box paddingLeft={2}>
                <Text dimColor>… showing last {Math.min(10, violations.length)} of {totalCount}</Text>
            </Box>
        </Box>
    );
}

// --- Settings Errors Notice (wV1) ---
export function useSettingsErrorsNotice() {
    const { addNotification, removeNotification } = useNotifications();
    const [errors, setErrors] = useState(() => getSettingsValidationErrors().errors);

    const refreshErrors = useCallback(() => {
        setErrors(getSettingsValidationErrors().errors);
    }, []);

    useEffect(() => {
        const unsubscribe = subscribeToSettingsErrors(refreshErrors);
        return () => unsubscribe();
    }, [refreshErrors]);

    useEffect(() => {
        if (errors.length > 0) {
            const text = `Found ${errors.length} invalid settings ${errors.length === 1 ? "file" : "files"} · /doctor for details`;
            addNotification({
                key: SETTINGS_ERROR_KEY,
                text,
                color: "warning",
                priority: "high",
                timeoutMs: 60000
            } as any);
        } else {
            removeNotification(SETTINGS_ERROR_KEY);
        }
    }, [errors, addNotification, removeNotification]);

    return errors;
}

// --- MCP Failure Notices (M29) ---
export function useMcpFailureNotice({ mcpClients = [] }: { mcpClients?: any[] }) {
    const { addNotification } = useNotifications();

    useEffect(() => {
        const failed = mcpClients.filter(
            (client) =>
                client.type === "failed" &&
                client.config?.type !== "sse-ide" &&
                client.config?.type !== "ws-ide" &&
                client.config?.type !== "claudeai-proxy"
        );
        const needsAuth = mcpClients.filter(
            (client) => client.type === "needs-auth" && client.config?.type !== "claudeai-proxy"
        );

        if (failed.length === 0 && needsAuth.length === 0) return;

        if (failed.length > 0) {
            addNotification({
                key: "mcp-failed",
                text: `${failed.length} MCP ${failed.length === 1 ? "server" : "servers"} failed · /mcp for info`,
                priority: "medium"
            } as any);
        }

        if (needsAuth.length > 0) {
            addNotification({
                key: "mcp-needs-auth",
                text: `${needsAuth.length} MCP ${needsAuth.length === 1 ? "server needs" : "servers need"} auth · /mcp for info`,
                priority: "medium"
            } as any);
        }
    }, [addNotification, mcpClients]);
}

// --- LSP Error Notice Hook (_29) ---
export function useLspErrorNotice() {
    const { addNotification } = useNotifications();
    const [, setAppState] = useAppState();
    const [shouldPoll, setShouldPoll] = useState(true);
    const knownErrors = useRef(new Set<string>());

    const reportError = useCallback(
        (source: string, error: string) => {
            const key = `${source}:${error}`;
            if (knownErrors.current.has(key)) return;
            knownErrors.current.add(key);
            logInfo(`LSP error: ${source} - ${error}`);

            setAppState((state) => {
                const existing = new Set(
                    state.plugins.errors.map((err: any) => {
                        if (err.type === "generic-error") return `generic-error:${err.source}:${err.error}`;
                        return `${err.type}:${err.source}`;
                    })
                );
                const genericKey = `generic-error:${source}:${error}`;
                if (existing.has(genericKey)) return state;
                return {
                    ...state,
                    plugins: {
                        ...state.plugins,
                        errors: [...state.plugins.errors, { type: "generic-error", source, error }]
                    }
                };
            });

            const displayName = source.startsWith("plugin:") ? source.split(":")[1] ?? source : source;
            addNotification({
                key: `lsp-error-${source}`,
                text: `LSP for ${displayName} failed · /plugin for details`,
                priority: "medium",
                timeoutMs: 8000
            } as any);
        },
        [addNotification, setAppState]
    );

    const checkStatus = useCallback(() => {
        const status = getLspManagerStatus();
        if (status.status === "failed") {
            reportError("lsp-manager", status.error?.message ?? "Unknown error");
            setShouldPoll(false);
            return;
        }
        if (status.status === "pending" || status.status === "not-started") return;

        const manager = getLspManager();
        if (manager) {
            const servers = manager.getAllServers();
            for (const [name, server] of servers) {
                if (server.state === "error" && server.lastError) {
                    reportError(name, server.lastError.message);
                }
            }
        }
    }, [reportError]);

    useInterval(checkStatus, shouldPoll ? LSP_ERROR_POLL_INTERVAL_MS : null);

    useEffect(() => {
        checkStatus();
    }, [checkStatus]);
}

// --- Binary Check (P29) ---
const binaryCheckCache = new Map<string, boolean>();

export async function checkBinaryExists(command: string): Promise<boolean> {
    if (!command || !command.trim()) {
        logInfo("[binaryCheck] Empty command provided, returning false");
        return false;
    }

    const trimmed = command.trim();
    const cached = binaryCheckCache.get(trimmed);
    if (cached !== undefined) {
        logInfo(`[binaryCheck] Cache hit for '${trimmed}': ${cached}`);
        return cached;
    }

    const tool = (process.platform as string) === "win32" ? "where" : "which";
    if ((process.platform as string) === "unknown") logInfo("[binaryCheck] Unknown platform, defaulting to 'which'");

    let found = false;
    try {
        await execFileAsync(tool, [trimmed], { timeout: 5000 });
        found = true;
    } catch (error: any) {
        found = false;
        logInfo(`[binaryCheck] Binary '${trimmed}' not found (exit code: ${error?.code ?? "unknown"})`);
    }
    binaryCheckCache.set(trimmed, found);
    if (found) logInfo(`[binaryCheck] Binary '${trimmed}' found`);
    return found;
}

// --- Marketplace Index (iQ7) ---
export async function indexMarketplacePlugins(): Promise<Map<string, MarketplaceIndexEntry>> {
    const index = new Map<string, MarketplaceIndexEntry>();
    try {
        const config = getMarketplaceConfig();
        for (const marketplaceName of Object.keys(config)) {
            try {
                const marketplace = await loadMarketplace(marketplaceName);
                const official = isOfficialMarketplace(marketplaceName);
                for (const plugin of marketplace.plugins) {
                    if (!plugin.lspServers) continue;
                    const lspInfo = extractLspServers(plugin.lspServers);
                    if (!lspInfo) continue;
                    const pluginId = `${plugin.name}@${marketplaceName}`;
                    index.set(pluginId, {
                        entry: plugin,
                        marketplaceName,
                        extensions: lspInfo.extensions,
                        command: lspInfo.command,
                        isOfficial: official
                    });
                }
            } catch (error) {
                logInfo(`[lspRecommendation] Failed to load marketplace ${marketplaceName}: ${error}`);
            }
        }
    } catch (error) {
        logInfo(`[lspRecommendation] Failed to load marketplaces config: ${error}`);
    }
    return index;
}

function isRecord(value: unknown): value is Record<string, any> {
    return typeof value === "object" && value !== null;
}

function extractLspServers(servers: MarketplaceEntry["lspServers"]) {
    if (!servers) return null;
    if (typeof servers === "string") {
        logInfo("[lspRecommendation] Skipping string path lspServers (not readable from marketplace)");
        return null;
    }
    if (Array.isArray(servers)) {
        for (const entry of servers) {
            if (typeof entry === "string") continue;
            const result = extractLspServersFromEntry(entry);
            if (result) return result;
        }
        return null;
    }
    return extractLspServersFromEntry(servers);
}

function extractLspServersFromEntry(entry: Record<string, any>) {
    const extensions = new Set<string>();
    let command: string | null = null;
    for (const value of Object.values(entry)) {
        if (!isRecord(value)) continue;
        if (!command && typeof value.command === "string") command = value.command;
        const extensionMap = value.extensionToLanguage;
        if (isRecord(extensionMap)) {
            for (const key of Object.keys(extensionMap)) {
                extensions.add(key.toLowerCase());
            }
        }
    }
    if (!command || extensions.size === 0) return null;
    return { extensions, command };
}

// --- LSP Recommendation Engine (v29) ---
export async function getLspPluginRecommendation(filePath: string): Promise<LspRecommendation[]> {
    if (shouldDisableLspRecommendations()) {
        logInfo("[lspRecommendation] Recommendations are disabled");
        return [];
    }

    const extension = extname(filePath).toLowerCase();
    if (!extension) {
        logInfo("[lspRecommendation] No file extension found");
        return [];
    }
    logInfo(`[lspRecommendation] Looking for LSP plugins for ${extension}`);

    const marketplaceIndex = await indexMarketplacePlugins();
    const settings = getSettings("userSettings") as any;
    const neverSuggest = settings?.lspRecommendationNeverPlugins ?? [];
    const matches: Array<{ info: MarketplaceIndexEntry; pluginId: string }> = [];

    for (const [pluginId, info] of marketplaceIndex) {
        if (!info.extensions.has(extension)) continue;
        if (neverSuggest.includes(pluginId)) {
            logInfo(`[lspRecommendation] Skipping ${pluginId} (in never suggest list)`);
            continue;
        }
        if (isPluginInstalled(pluginId)) {
            logInfo(`[lspRecommendation] Skipping ${pluginId} (already installed)`);
            continue;
        }
        matches.push({ info, pluginId });
    }

    const readyMatches: Array<{ info: MarketplaceIndexEntry; pluginId: string }> = [];
    for (const match of matches) {
        if (await checkBinaryExists(match.info.command)) {
            readyMatches.push(match);
            logInfo(`[lspRecommendation] Binary '${match.info.command}' found for ${match.pluginId}`);
        } else {
            logInfo(`[lspRecommendation] Skipping ${match.pluginId} (binary '${match.info.command}' not found)`);
        }
    }

    readyMatches.sort((left, right) => {
        if (left.info.isOfficial && !right.info.isOfficial) return -1;
        if (!left.info.isOfficial && right.info.isOfficial) return 1;
        return 0;
    });

    return readyMatches.map(({ info, pluginId }) => ({
        pluginId,
        pluginName: info.entry.name,
        marketplaceName: info.marketplaceName,
        description: info.entry.description,
        isOfficial: info.isOfficial,
        extensions: Array.from(info.extensions),
        command: info.command
    }));
}

// --- LSP Recommendation Hook (h29) ---
export function useLspPluginRecommendation() {
    const [state] = useAppState();
    const { addNotification } = useNotifications();
    const [recommendation, setRecommendation] = useState<{
        pluginId: string;
        pluginName: string;
        pluginDescription?: string;
        fileExtension: string;
        shownAt: number;
    } | null>(null);
    const seenFiles = useRef(new Set<string>());
    const isRunning = useRef(false);

    useEffect(() => {
        if (recommendation || isRunning.current) return;
        if (shouldDisableLspRecommendations()) return;
        const trackedFiles: string[] = state.fileHistory?.trackedFiles ?? [];
        const newFiles = trackedFiles.filter((file) => {
            if (seenFiles.current.has(file)) return false;
            seenFiles.current.add(file);
            return true;
        });
        if (newFiles.length === 0) return;

        isRunning.current = true;
        (async () => {
            for (const file of newFiles) {
                try {
                    const match = (await getLspPluginRecommendation(file))[0];
                    if (match) {
                        logInfo(`[useLspPluginRecommendation] Found match: ${match.pluginName} for ${file}`);
                        setRecommendation({
                            pluginId: match.pluginId,
                            pluginName: match.pluginName,
                            pluginDescription: match.description,
                            fileExtension: extname(file),
                            shownAt: Date.now()
                        });
                        return;
                    }
                } catch (error) {
                    logError(error);
                }
            }
        })().finally(() => {
            isRunning.current = false;
        });
    }, [state.fileHistory?.trackedFiles, recommendation]);

    const handleResponse = useCallback(
        (response: LspPromptResponse) => {
            if (!recommendation) return;
            const { pluginId, pluginName, shownAt } = recommendation;
            logInfo(`[useLspPluginRecommendation] User response: ${response} for ${pluginName}`);

            switch (response) {
                case "yes":
                    installRecommendedPlugin(pluginId, pluginName, addNotification);
                    break;
                case "no": {
                    const elapsed = Date.now() - shownAt;
                    if (elapsed >= LSP_RECOMMENDATION_TIMEOUT_IGNORE_MS) {
                        logInfo(`[useLspPluginRecommendation] Timeout detected (${elapsed}ms), incrementing ignored count`);
                        incrementIgnoredCount();
                    }
                    break;
                }
                case "never":
                    addNeverSuggestPlugin(pluginId);
                    break;
                case "disable":
                    disableLspRecommendations();
                    break;
            }

            setRecommendation(null);
        },
        [addNotification, recommendation]
    );

    return {
        recommendation,
        handleResponse
    };
}

// --- Plugin Installation Flow (sQ7) ---
export async function installRecommendedPlugin(
    pluginId: string,
    pluginName: string,
    addNotification: (notification: any) => void
) {
    try {
        logInfo(`[useLspPluginRecommendation] Installing plugin: ${pluginId}`);
        const result = await installPlugin(pluginId, "user");
        if (!result?.success) throw new Error(result?.message ?? `Failed to install ${pluginId}`);
        const settings = getSettings("userSettings") as any;
        updateSettings("userSettings", {
            enabledPlugins: {
                ...(settings?.enabledPlugins ?? {}),
                [pluginId]: true
            }
        });
        logInfo(`[useLspPluginRecommendation] Plugin installed: ${pluginId}`);
        addNotification({
            key: "lsp-plugin-installed",
            text: `${pluginName} installed · restart to apply`,
            priority: "immediate",
            timeoutMs: 5000
        } as any);
    } catch (error) {
        logError(error);
        addNotification({
            key: "lsp-plugin-install-failed",
            text: `Failed to install ${pluginName}`,
            priority: "immediate",
            timeoutMs: 5000
        } as any);
    }
}

// --- LSP Suggestion UI (u29) ---
export function LspRecommendationPrompt({
    pluginName,
    pluginDescription,
    fileExtension,
    onResponse
}: {
    pluginName: string;
    pluginDescription?: string;
    fileExtension: string;
    onResponse: (response: LspPromptResponse) => void;
}) {
    const handlerRef = useRef(onResponse);
    handlerRef.current = onResponse;

    useEffect(() => {
        const timeout = setTimeout(() => {
            handlerRef.current("no");
        }, LSP_RECOMMENDATION_TIMEOUT_MS);
        return () => clearTimeout(timeout);
    }, []);

    const options = useMemo(
        () => [
            {
                label: (
                    <Text>
                        Yes, install <Text bold>{pluginName}</Text>
                    </Text>
                ),
                value: "yes"
            },
            { label: "No, not now", value: "no" },
            {
                label: (
                    <Text>
                        Never for <Text bold>{pluginName}</Text>
                    </Text>
                ),
                value: "never"
            },
            { label: "Disable all LSP recommendations", value: "disable" }
        ],
        [pluginName]
    );

    return (
        <Box flexDirection="column">
            <Box flexDirection="column" paddingX={2} paddingY={1}>
                <Box marginBottom={1}>
                    <Text dimColor>LSP provides code intelligence like go-to-definition and error checking</Text>
                </Box>
                <Box>
                    <Text dimColor>Plugin:</Text>
                    <Text> {pluginName}</Text>
                </Box>
                {pluginDescription && (
                    <Box>
                        <Text dimColor>{pluginDescription}</Text>
                    </Box>
                )}
                <Box>
                    <Text dimColor>Triggered by:</Text>
                    <Text> {fileExtension} files</Text>
                </Box>
                <Box marginTop={1}>
                    <Text>Would you like to install this LSP plugin?</Text>
                </Box>
                <Box>
                    <PermissionSelect
                        options={options}
                        onChange={(value) => onResponse(value as LspPromptResponse)}
                        onCancel={() => onResponse("no")}
                    />
                </Box>
            </Box>
        </Box>
    );
}

function useInterval(callback: () => void, delay: number | null) {
    useEffect(() => {
        if (delay === null) return;
        const interval = setInterval(callback, delay);
        return () => clearInterval(interval);
    }, [callback, delay]);
}

function getLspManagerStatus(): { status: "pending" | "not-started" | "failed" | "ready"; error?: Error } {
    return { status: "not-started" };
}

function getLspManager(): {
    getAllServers: () => Map<string, { state: string; lastError?: Error }>;
} | null {
    return null;
}
