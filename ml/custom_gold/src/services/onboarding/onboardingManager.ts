import * as fs from "node:fs";
import * as path from "node:path";
import { getSettings, updateSettings } from "../terminal/settings.js";

/**
 * Utility to manage onboarding tasks and state.
 * Deobfuscated from Mi1 and related in chunk_215.ts.
 */

export interface OnboardingTask {
    key: string;
    text: string;
    isComplete: boolean;
    isCompletable: boolean;
    isEnabled: boolean;
}

/**
 * Checks if all completable onboarding tasks are finished.
 */
export function areAllTasksComplete(): boolean {
    return getOnboardingTasks()
        .filter(t => t.isCompletable && t.isEnabled)
        .every(t => t.isComplete);
}

/**
 * Returns the list of onboarding tasks for the current workspace.
 */
export function getOnboardingTasks(projectRoot: string = process.cwd()): OnboardingTask[] {
    const hasClaudeMd = fs.existsSync(path.join(projectRoot, "CLAUDE.md"));
    const isGitRepo = fs.existsSync(path.join(projectRoot, ".git"));

    return [
        {
            key: "workspace",
            text: "Ask Claude to create a new app or clone a repository",
            isComplete: false, // This task is conceptual/action-based, maybe hard to detect automatically without more context
            isCompletable: true,
            isEnabled: isGitRepo
        },
        {
            key: "claudemd",
            text: "Run /init to create a CLAUDE.md file with instructions for Claude",
            isComplete: hasClaudeMd,
            isCompletable: true,
            isEnabled: !isGitRepo
        }
    ];
}

/**
 * Updates onboarding state when tasks are completed.
 */
export function handleTaskCompletion() {
    const settings = getSettings("userSettings");
    if (areAllTasksComplete() && !settings.hasCompletedProjectOnboarding) {
        updateSettings("userSettings", { hasCompletedProjectOnboarding: true });
    }
}

/**
 * Increments the onboarding seen count.
 */
export function recordOnboardingDisplay() {
    const settings = getSettings("userSettings");
    const count = settings.projectOnboardingSeenCount || 0;
    updateSettings("userSettings", { projectOnboardingSeenCount: count + 1 });
}

/**
 * Decides whether to show onboarding UI.
 */
export function shouldShowOnboarding(): boolean {
    const settings = getSettings("userSettings");
    if (settings.hasCompletedProjectOnboarding) return false;
    if ((settings.projectOnboardingSeenCount || 0) >= 4) return false;
    if (process.env.IS_DEMO) return false;
    // Also disable if running in CI or specific modes
    return true;
}
