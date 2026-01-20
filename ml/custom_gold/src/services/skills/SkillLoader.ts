/**
 * Logic from chunk_558.ts (Skill / Custom Prompt Loader)
 * This service handles loading and factory creation of custom prompt-based skills.
 */

import fs from "node:fs";
import path from "node:path";
import { homedir } from "node:os";
import { memoize } from "../../utils/shared/lodashLikeRuntimeAndEnv.js";
import { parseFrontmatter } from "../../utils/shared/frontmatter.js";
import { evaluatePromptBashCommands } from "../bash/PromptEvaluator.js";
import { normalizeModelId } from "../claude/modelSettings.js";
import { log } from "../logger/loggerService.js";
import { getConfigDir } from "../../utils/settings/runtimeSettingsAndAuth.js";

const logger = log("skills");

/**
 * Returns the base directory for managed (policy) settings.
 * Matches logic from chunk_558's BL().
 */
function getManagedBaseDir(): string {
    if (process.platform !== "win32" && fs.existsSync("/etc")) return "/etc";
    return homedir();
}

/**
 * Returns the skill directory for a given settings source.
 * Matches logic from chunk_558's eZ9().
 */
export function getSkillDirForSource(source: string): string {
    switch (source) {
        case "policySettings":
            return path.join(getManagedBaseDir(), ".claude", "skills");
        case "userSettings":
            return path.join(getConfigDir(), "skills");
        case "projectSettings":
            return ".claude/skills";
        case "plugin":
            return "plugin";
        default:
            return "";
    }
}

/**
 * Gets a unique identifier for a file based on its device and inode.
 * Matches logic from chunk_558's j77().
 */
function getFileInode(filePath: string): string | null {
    try {
        const stats = fs.statSync(filePath);
        return `${stats.dev}:${stats.ino}`;
    } catch {
        return null;
    }
}

/**
 * Coerces value to boolean.
 */
function isTrue(value: any): boolean {
    return value === true || value === "true";
}

/**
 * Capitalizes the first letter of a string.
 */
function capitalize(s: string): string {
    if (!s) return s;
    return s.charAt(0).toUpperCase() + s.slice(1);
}

/**
 * Formats the source name for display.
 * Matches logic from chunk_558's ci().
 */
function formatSourceKey(source: string): string {
    if (source === "plugin") return "plugin";
    if (source === "policySettings") return "managed";
    if (source === "userSettings") return "user";
    if (source === "projectSettings") return "project";
    return source;
}

/**
 * Extracts a fallback description from markdown content.
 * Matches logic from chunk_558's Um().
 */
function extractDescriptionFromMarkdown(content: string, type: string): string {
    const lines = content.split("\n").map(l => l.trim()).filter(l => l && !l.startsWith("#"));
    return lines[0] || `Custom ${type}`;
}

/**
 * Normalizes tools to an array.
 */
function normalizeTools(tools: any): string[] {
    if (!tools) return [];
    if (Array.isArray(tools)) return tools;
    if (typeof tools === "string") return tools.split(",").map(t => t.trim());
    return [];
}

/**
 * Factory to create a Skill object (Prompt-based tool).
 * Deobfuscated from AY9.
 */
export function createSkill(options: {
    skillName: string;
    displayName?: string;
    description: string;
    hasUserSpecifiedDescription: boolean;
    markdownContent: string;
    allowedTools: string[];
    argumentHint?: string;
    whenToUse?: string;
    version?: string;
    model?: string;
    disableModelInvocation: boolean;
    userInvocable: boolean;
    source: string;
    baseDir?: string;
    loadedFrom: string;
}) {
    const {
        skillName,
        displayName,
        description,
        hasUserSpecifiedDescription,
        markdownContent,
        allowedTools,
        argumentHint,
        whenToUse,
        version,
        model,
        disableModelInvocation,
        userInvocable,
        source,
        baseDir,
        loadedFrom
    } = options;

    const fullDescription = `${description} (${formatSourceKey(source)})`;

    return {
        type: "prompt",
        name: skillName,
        description: fullDescription,
        hasUserSpecifiedDescription,
        allowedTools,
        argumentHint,
        whenToUse,
        version,
        model,
        disableModelInvocation,
        userInvocable,
        contentLength: markdownContent.length,
        isEnabled: () => true,
        isHidden: !userInvocable,
        progressMessage: "running",
        userFacingName() {
            return displayName || skillName;
        },
        source,
        loadedFrom,
        async getPromptForCommand(args: string, context: any) {
            let prompt = baseDir ? `Base directory for this skill: ${baseDir}\n\n${markdownContent}` : markdownContent;

            if (args) {
                if (prompt.includes("$ARGUMENTS")) {
                    prompt = prompt.replaceAll("$ARGUMENTS", args);
                } else {
                    prompt = prompt + `\n\nARGUMENTS: ${args}`;
                }
            }

            // Process prompt with variables and dynamic command execution
            prompt = await evaluatePromptBashCommands(prompt, {
                ...context,
                async getAppState() {
                    const state = await context.getAppState();
                    return {
                        ...state,
                        toolPermissionContext: {
                            ...state.toolPermissionContext,
                            alwaysAllowRules: {
                                ...state.toolPermissionContext.alwaysAllowRules,
                                command: allowedTools
                            }
                        }
                    };
                }
            }, `/${skillName}`);

            return [{
                type: "text",
                text: prompt
            }];
        }
    };
}

/**
 * Loads custom prompts from a directory and converts them to skills.
 * Deobfuscated from oL0.
 */
export async function loadSkillDirectory(dirPath: string, source: string) {
    const skills: { skill: any; filePath: string }[] = [];
    try {
        if (!fs.existsSync(dirPath)) return [];
        const entries = fs.readdirSync(dirPath, { withFileTypes: true });

        for (const entry of entries) {
            try {
                // Skills can be in subdirectories with SKILL.md or as standalone .md files (handled in y77 / loadAll)
                // Here we specifically look for the directory-based structure
                if (entry.isDirectory() || entry.isSymbolicLink()) {
                    const skillDir = path.join(dirPath, entry.name);
                    const skillFile = path.join(skillDir, "SKILL.md");

                    if (fs.existsSync(skillFile)) {
                        const rawContent = fs.readFileSync(skillFile, "utf-8");
                        const { frontmatter, content } = parseFrontmatter(rawContent);

                        const skillName = entry.name;
                        const description = frontmatter.description ?? extractDescriptionFromMarkdown(content, "Skill");
                        const allowedTools = normalizeTools(frontmatter["allowed-tools"]);
                        const userInvocable = isTrue(frontmatter["user-invocable"] ?? true);
                        const disableModelInvocation = isTrue(frontmatter["disable-model-invocation"] ?? false);
                        const model = frontmatter.model === "inherit" ? undefined : frontmatter.model;

                        skills.push({
                            skill: createSkill({
                                skillName,
                                displayName: frontmatter.name,
                                description,
                                hasUserSpecifiedDescription: !!frontmatter.description,
                                markdownContent: content,
                                allowedTools,
                                argumentHint: frontmatter["argument-hint"],
                                whenToUse: frontmatter.when_to_use,
                                version: frontmatter.version,
                                model,
                                disableModelInvocation,
                                userInvocable,
                                source,
                                baseDir: skillDir,
                                loadedFrom: "skills"
                            }),
                            filePath: skillFile
                        });
                    }
                }
            } catch (err) {
                logger.error(`Error loading skill from ${path.join(dirPath, entry.name)}: ${err}`);
            }
        }
    } catch (err) {
        logger.error(`Error reading skill directory ${dirPath}: ${err}`);
    }
    return skills;
}

/**
 * Replaces the old findFilesByPattern (Fd) for deobfuscation.
 * In the original, this finds all .md files in certain directories.
 */
async function findMarkdownFiles(subdir: string, env: any): Promise<{ filePath: string; baseDir: string; source: string }[]> {
    // This is a placeholder for the more complex Fd implementation
    // For now, it returns common locations.
    return [];
}

/**
 * Legacy loader for custom commands.
 * Deobfuscated from y77.
 */
async function loadLegacyCommands(env: any) {
    const files = await findMarkdownFiles("commands", env);
    const results: { skill: any; filePath: string }[] = [];

    for (const file of files) {
        try {
            const rawContent = fs.readFileSync(file.filePath, "utf-8");
            const { frontmatter, content } = parseFrontmatter(rawContent);

            const description = frontmatter.description ?? extractDescriptionFromMarkdown(content, "Custom command");
            const allowedTools = normalizeTools(frontmatter["allowed-tools"]);
            const userInvocable = frontmatter["user-invocable"] === undefined ? true : isTrue(frontmatter["user-invocable"]);
            const disableModelInvocation = isTrue(frontmatter["disable-model-invocation"] ?? false);
            const model = frontmatter.model === "inherit" ? undefined : (frontmatter.model ? normalizeModelId(frontmatter.model) : undefined);

            // Logic for determining skill name from file path (x77)
            const skillName = path.basename(file.filePath, ".md").toLowerCase();

            results.push({
                skill: createSkill({
                    skillName,
                    description,
                    hasUserSpecifiedDescription: !!frontmatter.description,
                    markdownContent: content,
                    allowedTools,
                    argumentHint: frontmatter["argument-hint"],
                    whenToUse: frontmatter.when_to_use,
                    version: frontmatter.version,
                    model,
                    disableModelInvocation,
                    userInvocable,
                    source: file.source,
                    baseDir: path.dirname(file.filePath),
                    loadedFrom: "commands_DEPRECATED"
                }),
                filePath: file.filePath
            });
        } catch (err) {
            logger.error(`Error loading legacy command from ${file.filePath}: ${err}`);
        }
    }
    return results;
}

/**
 * Main loader that aggregates skills from all sources.
 * Deobfuscated from sL0 (memoized).
 */
export const loadAllSkills = memoize(async (env: any) => {
    const managedDir = getSkillDirForSource("policySettings");
    const userDir = getSkillDirForSource("userSettings");

    // Placeholder for project-specific skill directories (pq0)
    const projectDirs: string[] = [];
    if (process.cwd()) {
        const projectSkills = path.join(process.cwd(), ".claude", "skills");
        if (fs.existsSync(projectSkills)) projectDirs.push(projectSkills);
    }

    logger.debug(`Loading skills from: managed=${managedDir}, user=${userDir}, project=[${projectDirs.join(", ")}]`);

    const loaders = [
        loadSkillDirectory(managedDir, "policySettings"),
        loadSkillDirectory(userDir, "userSettings"),
        ...projectDirs.map(d => loadSkillDirectory(d, "projectSettings")),
        loadLegacyCommands(env)
    ];

    const allBundles = await Promise.all(loaders);
    const flattened = allBundles.flat();

    const uniqueSkills = new Map<string, any>();
    const inodeMap = new Map<string, string>();

    for (const { skill, filePath } of flattened) {
        if (skill.type !== "prompt") continue;

        const inode = getFileInode(filePath);
        if (inode === null) {
            uniqueSkills.set(skill.name, skill);
            continue;
        }

        const existingSource = inodeMap.get(inode);
        if (existingSource !== undefined) {
            logger.debug(`Skipping duplicate skill '${skill.name}' from ${skill.source} (same inode already loaded from ${existingSource})`);
            continue;
        }

        inodeMap.set(inode, skill.source);
        uniqueSkills.set(skill.name, skill);
    }

    const deduplicatedResults = Array.from(uniqueSkills.values());
    const countDiff = flattened.length - deduplicatedResults.length;

    if (countDiff > 0) {
        logger.debug(`Deduplicated ${countDiff} skills (same inode or name shadow)`);
    }

    logger.info(`Loaded ${deduplicatedResults.length} unique skills`);
    return deduplicatedResults;
});

/**
 * Clears the skill cache.
 * Deobfuscated from BY9.
 */
export function clearSkillCache() {
    (loadAllSkills as any).cache?.clear?.();
}

/**
 * Formats skill source names for display.
 * Deobfuscated from v77.
 */
export function formatSkillSource(source: string): string {
    if (source === "plugin") return "Plugin skills";
    return `${capitalize(formatSourceKey(source))} skills`;
}
