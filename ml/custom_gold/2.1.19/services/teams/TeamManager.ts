/**
 * File: src/services/teams/TeamManager.ts
 * Role: Logic for managing team-specific configurations and member names.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync, readdirSync, rmSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";

export interface TeamMember {
    name: string;
    email?: string;
    agentId?: string;
    agentType?: string;
    model?: string;
    prompt?: string;
    color?: string;
    planModeRequired?: boolean;
    joinedAt?: number;
    tmuxPaneId?: string;
    tmuxSessionName?: string;
    cwd?: string;
    subscriptions?: string[];
    backendType?: string;
}

export interface TeamConfig {
    name: string;
    members: TeamMember[];
    description?: string;
    leadAgentId?: string;
    createdAt?: number;
}

/**
 * Sanitizes a team name for use as a directory name.
 */
export function sanitizeTeamName(input: string): string {
    return input.replace(/[^a-zA-Z0-9]/g, "-").toLowerCase();
}

/**
 * Returns the base directory for teams configuration.
 */
export function getTeamsDir(): string {
    return join(homedir(), ".claude", "teams");
}

/**
 * Returns the directory for a specific team.
 */
export function getTeamDir(teamName: string): string {
    return join(getTeamsDir(), sanitizeTeamName(teamName));
}

/**
 * Returns the config file path for a specific team directory.
 */
export function getTeamConfigPath(teamName: string): string {
    return join(getTeamDir(teamName), "config.json");
}

/**
 * Returns the legacy team file path (if any).
 */
export function getLegacyTeamFilePath(teamName: string): string {
    return join(getTeamsDir(), `${sanitizeTeamName(teamName)}.json`);
}

/**
 * Returns the tasks directory for a specific team.
 */
export function getTeamTasksDir(teamName: string): string {
    return join(homedir(), ".claude", "tasks", sanitizeTeamName(teamName));
}

/**
 * Reads the configuration for a team.
 */
export function readTeamConfig(teamName: string): TeamConfig | null {
    const configPath = getTeamConfigPath(teamName);
    const legacyPath = getLegacyTeamFilePath(teamName);
    if (!existsSync(configPath) && !existsSync(legacyPath)) return null;
    try {
        const content = existsSync(configPath)
            ? readFileSync(configPath, "utf-8")
            : readFileSync(legacyPath, "utf-8");
        return JSON.parse(content) as TeamConfig;
    } catch {
        return null;
    }
}

/**
 * Writes the configuration for a team.
 */
export function writeTeamConfig(teamName: string, configData: TeamConfig): void {
    const teamDir = getTeamDir(teamName);
    mkdirSync(teamDir, { recursive: true });
    const configPath = getTeamConfigPath(teamName);
    writeFileSync(configPath, JSON.stringify(configData, null, 2));
}

/**
 * Generates a unique member name within a team.
 */
export function generateMemberName(memberName: string, teamName: string): string {
    const config = readTeamConfig(teamName);
    if (!config) return memberName;

    const existing = new Set(config.members.map(m => m.name.toLowerCase()));
    if (!existing.has(memberName.toLowerCase())) return memberName;

    let suffix = 2;
    while (existing.has(`${memberName}-${suffix}`.toLowerCase())) {
        suffix++;
    }
    return `${memberName}-${suffix}`;
}

/**
 * Lists available team configs on disk.
 */
export function listTeams(): TeamConfig[] {
    const teamsDir = getTeamsDir();
    if (!existsSync(teamsDir)) return [];
    const results: TeamConfig[] = [];

    for (const entry of readdirSync(teamsDir, { withFileTypes: true })) {
        if (entry.isDirectory()) {
            const teamName = entry.name;
            const configPath = join(teamsDir, teamName, "config.json");
            if (!existsSync(configPath)) continue;
            try {
                const content = readFileSync(configPath, "utf-8");
                results.push(JSON.parse(content) as TeamConfig);
            } catch {
                continue;
            }
            continue;
        }

        if (entry.isFile() && entry.name.endsWith(".json")) {
            try {
                const content = readFileSync(join(teamsDir, entry.name), "utf-8");
                results.push(JSON.parse(content) as TeamConfig);
            } catch {
                continue;
            }
        }
    }

    return results;
}

/**
 * Creates a team config and tasks directory.
 */
export function createTeam(config: TeamConfig): TeamConfig {
    writeTeamConfig(config.name, {
        ...config,
        createdAt: config.createdAt ?? Date.now(),
        members: config.members ?? []
    });
    mkdirSync(getTeamTasksDir(config.name), { recursive: true });
    return config;
}

/**
 * Removes a team config directory and tasks directory.
 */
export function removeTeam(teamName: string): void {
    const teamDir = getTeamDir(teamName);
    const legacyPath = getLegacyTeamFilePath(teamName);
    const tasksDir = getTeamTasksDir(teamName);

    if (existsSync(teamDir)) {
        rmSync(teamDir, { recursive: true, force: true });
    }
    if (existsSync(legacyPath)) {
        rmSync(legacyPath, { force: true });
    }
    if (existsSync(tasksDir)) {
        rmSync(tasksDir, { recursive: true, force: true });
    }
}
