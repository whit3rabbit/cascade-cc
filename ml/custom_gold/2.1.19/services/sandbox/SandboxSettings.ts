/**
 * File: src/services/sandbox/SandboxSettings.ts
 * Role: Manages permissions and settings for the tool execution sandbox.
 */

import { getToolSettings, getSettings } from '../config/SettingsService.js';
import { resolve } from 'node:path';
import { isIPv4 } from 'node:net';

/**
 * Parses a tool permission rule (e.g., "file_read(domain:example.com)").
 */
export function parseToolPermissionRule(rule: string): { toolName: string; ruleContent?: string } {
    const match = rule.match(/^([^(]+)\(([^)]+)\)$/);
    if (!match) return { toolName: rule };
    return { toolName: match[1], ruleContent: match[2] };
}

export interface SandboxNetworkConfig {
    allowedDomains: string[];
    deniedDomains: string[];
    allowUnixSockets?: boolean;
    allowAllUnixSockets?: boolean;
    allowLocalBinding?: boolean;
}

export interface SandboxFilesystemConfig {
    allowWrite: string[];
    denyWrite: string[];
    denyRead: string[];
}

export interface RipgrepConfig {
    command: string;
    args: string[];
}

export interface SandboxConfig {
    network: SandboxNetworkConfig;
    filesystem: SandboxFilesystemConfig;
    ripgrep: RipgrepConfig;
}

/**
 * Aggregates sandbox settings from various config layers.
 */
export function getSandboxConfig(): SandboxConfig {
    const layers = ["flagSettings", "policySettings", "userSettings", "projectSettings", "localSettings"];
    const allowWrite: string[] = ["."];
    const denyWrite: string[] = [];
    const denyRead: string[] = [];
    const allowedDomains: string[] = [];
    const deniedDomains: string[] = [];

    for (const layer of layers) {
        const settings = getToolSettings(layer);
        if (!settings?.permissions) continue;

        for (const allow of settings.permissions.allow || []) {
            const { toolName, ruleContent } = parseToolPermissionRule(allow);
            if (toolName === 'file_write' && ruleContent) {
                allowWrite.push(resolve(process.cwd(), ruleContent));
            }
            if (toolName === 'file_read' && ruleContent?.startsWith("domain:")) {
                allowedDomains.push(ruleContent.substring(7));
            }
        }

        for (const deny of settings.permissions.deny || []) {
            const { toolName, ruleContent } = parseToolPermissionRule(deny);
            if (toolName === 'file_write' && ruleContent) {
                denyWrite.push(resolve(process.cwd(), ruleContent));
            }
            if (toolName === 'file_read' && ruleContent) {
                denyRead.push(resolve(process.cwd(), ruleContent));
            }
        }
    }

    const settings = getSettings();
    const ripgrep = settings?.sandbox?.ripgrep || {
        command: 'rg',
        args: []
    };

    return {
        network: {
            allowedDomains,
            deniedDomains,
            allowUnixSockets: settings?.sandbox?.network?.allowUnixSockets,
            allowAllUnixSockets: settings?.sandbox?.network?.allowAllUnixSockets,
            allowLocalBinding: settings?.sandbox?.network?.allowLocalBinding
        },
        filesystem: {
            allowWrite,
            denyWrite,
            denyRead
        },
        ripgrep
    };
}

/**
 * Checks if the sandbox is globally enabled in settings.
 */
export function isSandboxEnabled(): boolean {
    const settings = getSettings();
    return !!settings?.sandbox?.enabled;
}

export function areUnsandboxedCommandsAllowed(): boolean {
    const settings = getSettings();
    return !!settings?.sandbox?.allowUnsandboxedCommands;
}

export function isDomainAllowed(domain: string): boolean {
    if (!isSandboxEnabled()) return true;
    const config = getSandboxConfig();

    if (isMatch(domain, config.network.deniedDomains)) {
        return false;
    }

    if (config.network.allowedDomains.length > 0) {
        return isMatch(domain, config.network.allowedDomains);
    }

    return true;
}

/**
 * Helper to match a domain/IP against a list of patterns (exact, suffix, CIDR, or Regex).
 */
function isMatch(domain: string, patterns: string[]): boolean {
    return patterns.some(pattern => {
        // 1. Regex Match: if pattern is wrapped in / /
        if (pattern.startsWith('/') && pattern.endsWith('/')) {
            try {
                const regex = new RegExp(pattern.slice(1, -1));
                return regex.test(domain);
            } catch {
                return false;
            }
        }

        // 2. CIDR Match: if pattern contains / and is not a regex
        if (pattern.includes('/')) {
            const [range, prefixStr] = pattern.split('/');
            const prefix = parseInt(prefixStr, 10);
            if (isIPv4(domain) && isIPv4(range) && !isNaN(prefix)) {
                return matchCIDR(domain, range, prefix);
            }
        }

        // 3. Exact match or Suffix match or Wildcard
        return domain === pattern || domain.endsWith("." + pattern) || pattern === "*";
    });
}

/**
 * Matches an IPv4 address against a CIDR range.
 */
function matchCIDR(ip: string, range: string, prefix: number): boolean {
    const ipNum = ipToLong(ip);
    const rangeNum = ipToLong(range);
    const mask = -1 << (32 - prefix);
    return (ipNum & mask) === (rangeNum & mask);
}

/**
 * Converts an IPv4 string to a 32-bit integer.
 */
function ipToLong(ip: string): number {
    return ip.split('.').reduce((acc, part) => (acc << 8) + parseInt(part, 10), 0) >>> 0;
}

/**
 * Checks if a URL is allowed.
 */
export function isUrlAllowed(urlStr: string): boolean {
    try {
        const url = new URL(urlStr);
        return isDomainAllowed(url.hostname);
    } catch {
        // If not a valid URL, we don't block based on domain but maybe based on protocol?
        return true;
    }
}

