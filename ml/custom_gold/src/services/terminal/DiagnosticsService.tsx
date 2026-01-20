// Logic from chunk_543.ts (Diagnostics & Tree Rendering)

import React from "react";
import { Box, Text } from "ink";
import { figures } from "../../vendor/terminalFigures.js";

const TREE_CHARS = {
    branch: "├",
    lastBranch: "└",
    line: "│",
    empty: " "
};

type RenderTreeOptions = {
    showValues?: boolean;
    hideFunctions?: boolean;
    themeName?: string;
    treeCharColors?: Record<string, string>;
};

type ConfigError = {
    file?: string;
    path?: string;
    message: string;
    suggestion?: string;
    docLink?: string;
    invalidValue?: any;
    mcpErrorMetadata?: { serverName?: string; severity?: "fatal" | "warning" };
};

function tint(text: string, _color?: string, _themeName?: string): string {
    return text;
}

// --- Tree Rendering Utility (b39) ---
export function renderTree(data: Record<string, any>, options: RenderTreeOptions = {}): string {
    const { showValues = true, hideFunctions = false, themeName = "dark", treeCharColors = {} } = options;
    const lines: string[] = [];
    const visited = new WeakSet<object>();

    function format(text: string, colorKey?: string) {
        const color = colorKey ? treeCharColors[colorKey] : undefined;
        return tint(text, color, themeName);
    }

    function walk(value: any, prefix: string, isLast: boolean, depth = 0) {
        if (typeof value === "string") {
            lines.push(prefix + format(value, "value"));
            return;
        }
        if (typeof value !== "object" || value === null) {
            if (showValues) {
                lines.push(prefix + format(String(value), "value"));
            }
            return;
        }
        if (visited.has(value)) {
            lines.push(prefix + format("[Circular]", "value"));
            return;
        }
        visited.add(value);

        const keys = Object.keys(value).filter((key) => {
            const entry = value[key];
            if (hideFunctions && typeof entry === "function") return false;
            return true;
        });

        keys.forEach((key, index) => {
            const entry = value[key];
            const last = index === keys.length - 1;
            const head = depth === 0 && index === 0 ? "" : prefix;
            const branch = last ? TREE_CHARS.lastBranch : TREE_CHARS.branch;
            const treeChar = format(branch, "treeChar");
            const keyLabel = key.trim() === "" ? "" : format(key, "key");
            const label = head + treeChar + (keyLabel ? ` ${keyLabel}` : "");
            const hasKey = key.trim() !== "";

            if (entry && typeof entry === "object" && visited.has(entry)) {
                const valueLabel = format("[Circular]", "value");
                lines.push(label + (hasKey ? ": " : label ? " " : "") + valueLabel);
            } else if (entry && typeof entry === "object" && !Array.isArray(entry)) {
                lines.push(label);
                const nextPrefix = format(last ? TREE_CHARS.empty : TREE_CHARS.line, "treeChar");
                walk(entry, head + nextPrefix + " ", last, depth + 1);
            } else if (Array.isArray(entry)) {
                lines.push(label + (hasKey ? ": " : label ? " " : "") + `[Array(${entry.length})]`);
            } else if (showValues) {
                const valueLabel = typeof entry === "function" ? "[Function]" : String(entry);
                lines.push(label + (hasKey ? ": " : label ? " " : "") + format(valueLabel, "value"));
            } else {
                lines.push(label);
            }
        });
    }

    const topKeys = Object.keys(data);
    if (topKeys.length === 0) return format("(empty)", "value");
    if (topKeys.length === 1 && topKeys[0] !== undefined && topKeys[0].trim() === "" && typeof data[topKeys[0]] === "string") {
        const value = data[topKeys[0]];
        const branch = format(TREE_CHARS.lastBranch, "treeChar");
        const label = format(value, "value");
        return `${branch} ${label}`;
    }

    walk(data, "", true);
    return lines.join("\n");
}

function groupErrorsByPath(errors: ConfigError[]) {
    const grouped: Record<string, string> = {};
    errors.forEach((error) => {
        if (!error.path) {
            grouped[""] = error.message;
            return;
        }
        const pathSegments = error.path.split(".");
        let displayPath = error.path;
        if (error.invalidValue !== null && error.invalidValue !== undefined && pathSegments.length > 0) {
            const displaySegments: string[] = [];
            for (let i = 0; i < pathSegments.length; i += 1) {
                const segment = pathSegments[i];
                if (!segment) continue;
                const index = parseInt(segment, 10);
                if (!Number.isNaN(index) && i === pathSegments.length - 1) {
                    if (typeof error.invalidValue === "string") displaySegments.push(`"${error.invalidValue}"`);
                    else if (error.invalidValue === null) displaySegments.push("null");
                    else if (error.invalidValue === undefined) displaySegments.push("undefined");
                    else displaySegments.push(String(error.invalidValue));
                } else {
                    displaySegments.push(segment);
                }
            }
            displayPath = displaySegments.join(".");
        }
        grouped[displayPath] = error.message;
    });
    return grouped;
}

// --- Config Diagnostics View (VH1) ---
export function ConfigDiagnosticsView({ errors }: { errors: ConfigError[] }) {
    const themeName = "dark";
    if (errors.length === 0) return null;

    const grouped: Record<string, ConfigError[]> = {};
    errors.forEach((error) => {
        const file = error.file || "(file not specified)";
        if (!grouped[file]) grouped[file] = [];
        grouped[file].push(error);
    });

    const files = Object.keys(grouped).sort();

    return (
        <Box flexDirection="column">
            {files.map((file) => {
                const fileErrors = grouped[file] || [];
                fileErrors.sort((a, b) => {
                    if (!a.path && b.path) return -1;
                    if (a.path && !b.path) return 1;
                    return (a.path || "").localeCompare(b.path || "");
                });

                const treeData = groupErrorsByPath(fileErrors);
                const suggestionPairs = new Map<string, { suggestion?: string; docLink?: string }>();
                fileErrors.forEach((error) => {
                    if (error.suggestion || error.docLink) {
                        const key = `${error.suggestion || ""}|${error.docLink || ""}`;
                        if (!suggestionPairs.has(key)) {
                            suggestionPairs.set(key, {
                                suggestion: error.suggestion,
                                docLink: error.docLink
                            });
                        }
                    }
                });

                const tree = renderTree(treeData, {
                    showValues: true,
                    themeName,
                    treeCharColors: {
                        treeChar: "inactive",
                        key: "text",
                        value: "inactive"
                    }
                });

                return (
                    <Box key={file} flexDirection="column">
                        <Text>{file}</Text>
                        <Box marginLeft={1}>
                            <Text dimColor>{tree}</Text>
                        </Box>
                        {suggestionPairs.size > 0 && (
                            <Box flexDirection="column" marginTop={1}>
                                {Array.from(suggestionPairs.values()).map((pair, index) => (
                                    <Box key={`suggestion-${index}`} flexDirection="column" marginBottom={1}>
                                        {pair.suggestion && (
                                            <Text dimColor wrap="wrap">
                                                {pair.suggestion}
                                            </Text>
                                        )}
                                        {pair.docLink && (
                                            <Text dimColor wrap="wrap">
                                                Learn more: {pair.docLink}
                                            </Text>
                                        )}
                                    </Box>
                                ))}
                            </Box>
                        )}
                    </Box>
                );
            })}
        </Box>
    );
}

function formatScopeLabel(scope: string): string {
    return scope;
}

function getScopePath(scope: string): string {
    return scope;
}

// --- Scope Error View (M47) ---
export function ScopeErrorView({ scope, parsingErrors, warnings }: { scope: string; parsingErrors: ConfigError[]; warnings: ConfigError[] }) {
    const hasErrors = parsingErrors.length > 0;
    const hasWarnings = warnings.length > 0;
    if (!hasErrors && !hasWarnings) return null;

    return (
        <Box flexDirection="column" marginTop={1}>
            <Box>
                {(hasErrors || hasWarnings) && (
                    <Text color={hasErrors ? "error" : "warning"}>
                        [{hasErrors ? "Failed to parse" : "Contains warnings"}] {" "}
                    </Text>
                )}
                <Text>{formatScopeLabel(scope)}</Text>
            </Box>
            <Box>
                <Text dimColor>Location: {getScopePath(scope)}</Text>
            </Box>
            <Box marginLeft={1} flexDirection="column">
                {parsingErrors.map((error, index) => (
                    <Box key={`error-${index}`}>
                        <Text>
                            <Text dimColor>└ </Text>
                            <Text color="error">[Error]</Text>
                            <Text dimColor>
                                {" "}
                                {error.mcpErrorMetadata?.serverName ? `[${error.mcpErrorMetadata.serverName}] ` : ""}
                                {error.path && error.path !== "" ? `${error.path}: ` : ""}
                                {error.message}
                            </Text>
                        </Text>
                    </Box>
                ))}
                {warnings.map((warning, index) => (
                    <Box key={`warning-${index}`}>
                        <Text>
                            <Text dimColor>└ </Text>
                            <Text color="warning">[Warning]</Text>
                            <Text dimColor>
                                {" "}
                                {warning.mcpErrorMetadata?.serverName ? `[${warning.mcpErrorMetadata.serverName}] ` : ""}
                                {warning.path && warning.path !== "" ? `${warning.path}: ` : ""}
                                {warning.message}
                            </Text>
                        </Text>
                    </Box>
                ))}
            </Box>
        </Box>
    );
}

function getMcpConfig(_scope: string) {
    return { errors: [] as ConfigError[] };
}

function filterMcpErrors(errors: ConfigError[], severity: "fatal" | "warning") {
    return errors.filter((error) => error.mcpErrorMetadata?.severity === severity);
}

// --- MCP Diagnostics View (DH1) ---
export function McpDiagnosticsView() {
    const scopes = [
        { scope: "user", config: getMcpConfig("user") },
        { scope: "project", config: getMcpConfig("project") },
        { scope: "local", config: getMcpConfig("local") },
        { scope: "enterprise", config: getMcpConfig("enterprise") }
    ];

    const hasFatal = scopes.some(({ config }) => filterMcpErrors(config.errors, "fatal").length > 0);
    const hasWarnings = scopes.some(({ config }) => filterMcpErrors(config.errors, "warning").length > 0);

    if (!hasFatal && !hasWarnings) return null;

    return (
        <Box flexDirection="column" marginTop={1} marginBottom={1}>
            <Text bold>MCP Config Diagnostics</Text>
            <Box marginTop={1}>
                <Text dimColor>
                    For help configuring MCP servers, see: {" "}
                    <Text>https://code.claude.com/docs/en/mcp</Text>
                </Text>
            </Box>
            {scopes.map(({ scope, config }) => (
                <ScopeErrorView
                    key={scope}
                    scope={scope}
                    parsingErrors={filterMcpErrors(config.errors, "fatal")}
                    warnings={filterMcpErrors(config.errors, "warning")}
                />
            ))}
        </Box>
    );
}

// --- Environment variable diagnostics (h39) ---
export function checkEnvironmentVariables() {
    return [] as { name: string; value?: string; status: string }[];
}

// --- CLAUDE.md size diagnostics (R47) ---
export async function checkClaudeMdHealth() {
    return null as null | {
        type: string;
        severity: string;
        message: string;
        details: string[];
        currentValue: number;
        threshold: number;
    };
}

// --- Agent description diagnostics (_47) ---
export async function checkAgentHealth(_agentDefinitions: any) {
    return null as null | {
        type: string;
        severity: string;
        message: string;
        details: string[];
        currentValue: number;
        threshold: number;
    };
}

// --- MCP tool diagnostics (j47) ---
export async function checkMcpHealth(_tools: any[], _toolPermissionContext: any, _agentDefinitions: any) {
    return null as null | {
        type: string;
        severity: string;
        message: string;
        details: string[];
        currentValue: number;
        threshold: number;
    };
}

// --- System health checks (u39) ---
export async function runSystemHealthChecks(tools: any[], agentDefinitions: any, toolPermissionContext: any) {
    const [claudeMdWarning, agentWarning, mcpWarning] = await Promise.all([
        checkClaudeMdHealth(),
        checkAgentHealth(agentDefinitions),
        checkMcpHealth(tools, toolPermissionContext, agentDefinitions)
    ]);

    return {
        claudeMdWarning,
        agentWarning,
        mcpWarning
    };
}

const _unused = figures;
void _unused;
