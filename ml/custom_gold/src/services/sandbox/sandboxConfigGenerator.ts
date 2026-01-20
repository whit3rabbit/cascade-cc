import * as os from "node:os";
import * as fs from "node:fs";
import { resolve as pathResolve, join as pathJoin, sep } from "node:path";
import { SandboxConfig } from "./sandboxManager.js";
import { getSettings, SETTING_SOURCES, SettingsSource } from "../terminal/settings.js";
import { getConfigDir } from "../../utils/settings/runtimeSettingsAndAuth.js";

/**
 * Parses a permission rule of format "toolName(ruleContent)".
 * Deobfuscated from vLA in chunk_224.ts.
 */
export function parsePermissionRule(rule: string): { toolName: string, ruleContent?: string } {
    const match = rule.match(/^([^(]+)\(([^)]+)\)$/);
    if (!match) return { toolName: rule };
    return { toolName: match[1], ruleContent: match[2] };
}

/**
 * Resolves a path for sandbox rules, handling // prefix for "no resolve".
 * Deobfuscated from $n1 in chunk_224.ts.
 */
function resolveSandboxPath(rulePath: string, source: SettingsSource, projectRoot: string): string {
    if (rulePath.startsWith("//")) return rulePath.slice(1);
    if (rulePath.startsWith("/")) {
        // Resolve relative to the source's root
        let root = projectRoot;
        if (source === "userSettings") {
            root = getConfigDir();
        }
        return pathResolve(root, rulePath.slice(1));
    }
    return rulePath;
}

/**
 * Constructs a final SandboxConfig from settings and environment.
 * Deobfuscated from wn1 in chunk_224.ts.
 */
export function getSandboxConfigFromSettings(
    mergedSettings: any,
    projectRoot: string
): SandboxConfig {
    const allowedDomains: string[] = [];
    const deniedDomains: string[] = [];

    // Extract domains from sandbox config in merged settings
    if (mergedSettings.sandbox?.network?.allowedDomains) {
        allowedDomains.push(...mergedSettings.sandbox.network.allowedDomains);
    }
    if (mergedSettings.sandbox?.network?.deniedDomains) {
        deniedDomains.push(...mergedSettings.sandbox.network.deniedDomains);
    }

    const writeAllow: string[] = ["."]; // Always allow current dir
    const writeDeny: string[] = [];
    const readDeny: string[] = [];

    // Sources to iterate for specific tool rules
    const relevantSources: SettingsSource[] = ["userSettings", "projectSettings", "localSettings"];

    for (const source of relevantSources) {
        const s = getSettings(source);
        if (!s.permissions) continue;

        // Path to this source's settings file should be read-only
        const configPath = source === "userSettings"
            ? pathJoin(getConfigDir(), "settings.json")
            : source === "projectSettings"
                ? pathJoin(projectRoot, ".claude", "settings.json")
                : pathJoin(projectRoot, ".claude", "settings.local.json");
        writeDeny.push(configPath);

        // Process allow rules
        for (const rule of s.permissions.allow || []) {
            const parsed = parsePermissionRule(rule);
            if (parsed.toolName === "WebFetch" && parsed.ruleContent?.startsWith("domain:")) {
                allowedDomains.push(parsed.ruleContent.substring(7));
            } else if (parsed.toolName === "Edit" && parsed.ruleContent) {
                writeAllow.push(resolveSandboxPath(parsed.ruleContent, source, projectRoot));
            }
        }

        // Process deny rules
        for (const rule of s.permissions.deny || []) {
            const parsed = parsePermissionRule(rule);
            if (parsed.toolName === "WebFetch" && parsed.ruleContent?.startsWith("domain:")) {
                deniedDomains.push(parsed.ruleContent.substring(7));
            } else if (parsed.toolName === "Edit" && parsed.ruleContent) {
                writeDeny.push(resolveSandboxPath(parsed.ruleContent, source, projectRoot));
            } else if (parsed.toolName === "Read" && parsed.ruleContent) {
                readDeny.push(resolveSandboxPath(parsed.ruleContent, source, projectRoot));
            }
        }
    }

    // Handle .git directory being outside project root
    const gitDir = pathJoin(projectRoot, ".git");
    try {
        const stats = fs.statSync(gitDir);
        if (stats.isFile()) {
            const content = fs.readFileSync(gitDir, "utf8");
            const match = content.match(/^gitdir:\s*(.+)$/m);
            if (match?.[1]) {
                const actualGitDir = match[1].trim();
                const dotGitIdx = actualGitDir.indexOf(".git");
                if (dotGitIdx > 0) {
                    const gitRoot = actualGitDir.substring(0, dotGitIdx - 1);
                    if (gitRoot !== projectRoot) {
                        writeAllow.push(gitRoot);
                    }
                }
            }
        }
    } catch { }

    // Ripgrep configuration
    const ripgrepValue = mergedSettings.sandbox?.ripgrep || {
        command: process.env.CLAUDE_RG_PATH || "rg",
        args: []
    };

    return {
        network: {
            allowedDomains: Array.from(new Set(allowedDomains)),
            deniedDomains: Array.from(new Set(deniedDomains)),
            allowUnixSockets: mergedSettings.sandbox?.network?.allowUnixSockets,
            allowAllUnixSockets: mergedSettings.sandbox?.network?.allowAllUnixSockets,
            allowLocalBinding: mergedSettings.sandbox?.network?.allowLocalBinding,
            httpProxyPort: mergedSettings.sandbox?.network?.httpProxyPort,
            socksProxyPort: mergedSettings.sandbox?.network?.socksProxyPort
        },
        filesystem: {
            allowRead: [], // sandbox-exec defaults to allow all read except denyRead
            denyRead: Array.from(new Set(readDeny)),
            allowWrite: Array.from(new Set(writeAllow)),
            denyWrite: Array.from(new Set(writeDeny)),
            allowGitConfig: mergedSettings.sandbox?.filesystem?.allowGitConfig
        },
        ignoreViolations: mergedSettings.sandbox?.ignoreViolations,
        enableWeakerNestedSandbox: mergedSettings.sandbox?.enableWeakerNestedSandbox,
        allowPty: mergedSettings.sandbox?.allowPty,
        mandatoryDenySearchDepth: mergedSettings.sandbox?.mandatoryDenySearchDepth,
        ripgrep: ripgrepValue
    };
}

