/**
 * File: src/services/sandbox/SandboxSettings.ts
 * Role: Manages permissions and settings for the tool execution sandbox.
 */

import { getToolSettings, getSettings } from '../config/SettingsService.js';
import { resolve } from 'node:path';

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
