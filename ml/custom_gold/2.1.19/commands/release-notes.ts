import { join, dirname } from 'node:path';
import { readFileSync, existsSync, mkdirSync, writeFileSync } from 'node:fs';
import axios from 'axios';
import semver from 'semver';
import { createCommandHelper, CommandContext } from './helpers.js';
import { getBaseConfigDir } from '../utils/shared/runtimeAndEnv.js';
import { EnvService } from '../services/config/EnvService.js';

const CHANGELOG_URL = "https://github.com/anthropics/claude-code/blob/main/CHANGELOG.md";
const CHANGELOG_RAW_URL = "https://raw.githubusercontent.com/anthropics/claude-code/refs/heads/main/CHANGELOG.md";
const MAX_VERSIONS = 5;

function getChangelogCachePath(): string {
    return join(getBaseConfigDir(), "cache", "changelog.md");
}

async function fetchAndCacheChangelog(): Promise<void> {
    if (EnvService.isTruthy("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC")) {
        return;
    }

    try {
        const response = await axios.get(CHANGELOG_RAW_URL, { timeout: 2000 });
        if (response.status === 200) {
            const cachePath = getChangelogCachePath();
            mkdirSync(dirname(cachePath), { recursive: true });
            writeFileSync(cachePath, response.data, 'utf-8');
        }
    } catch (error) {
        // Silently fail, we'll use existing cache or fallback
    }
}

function readCachedChangelog(): string {
    const cachePath = getChangelogCachePath();
    try {
        if (existsSync(cachePath)) {
            return readFileSync(cachePath, 'utf-8');
        }
    } catch {
        // Ignore
    }
    return "";
}

interface VersionNotes {
    version: string;
    notes: string[];
}

function parseChangelog(content: string): VersionNotes[] {
    if (!content) return [];

    const results: VersionNotes[] = [];
    const sections = content.split(/^## /gm).slice(1);

    for (const section of sections) {
        const lines = section.trim().split('\n');
        if (lines.length === 0) continue;

        const header = lines[0].trim();
        const versionMatch = header.split(" - ")[0]?.trim();
        if (!versionMatch || !semver.valid(versionMatch)) continue;

        const notes = lines.slice(1)
            .filter(line => line.trim().startsWith("- "))
            .map(line => line.trim().substring(2).trim())
            .filter(Boolean);

        if (notes.length > 0) {
            results.push({ version: versionMatch, notes });
        }
    }

    return results.sort((a, b) => semver.compare(b.version, a.version)).slice(0, MAX_VERSIONS);
}

function formatReleaseNotes(versionNotes: VersionNotes[]): string {
    return versionNotes.map(({ version, notes }) => {
        const header = `Version ${version}:`;
        const bulletPoints = notes.map(note => `• ${note}`).join('\n');
        return `${header}\n${bulletPoints}`;
    }).join('\n\n');
}

/**
 * Command definition for displaying release notes and version updates.
 */
export const releaseNotesCommandDefinition = createCommandHelper("release-notes", "Display the latest release notes and updates", {
    async getPromptForCommand(userInput: string, _context: CommandContext) {
        await fetchAndCacheChangelog();
        const changelog = readCachedChangelog();
        const versionNotes = parseChangelog(changelog);

        let releaseNotesText = "";
        if (versionNotes.length > 0) {
            releaseNotesText = formatReleaseNotes(versionNotes);
        } else {
            releaseNotesText = `Unable to retrieve detailed release notes. Please visit: ${CHANGELOG_URL}`;
        }

        return [
            {
                type: "text",
                text: `You are an AI assistant for the Claude Code CLI tool. Your task is to provide the user with the latest release notes and version history.

## Latest Release Notes

${releaseNotesText}

${userInput ? `\nUser asked about: ${userInput}` : ""}
`
            }
        ];
    },
    userFacingName() {
        return "release notes";
    }
});
