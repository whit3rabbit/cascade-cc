/**
 * Feature tips and onboarding logic.
 * Deobfuscated from chunk_201.ts.
 */

import { getSettings } from "../terminal/settings.js";

export interface FeatureCategory {
    id: string;
    name: string;
    description: string;
    order: number;
}

export interface FeatureTip {
    id: string;
    name: string;
    description: string;
    categoryId: string;
    tryItPrompt: string;
    hasBeenUsed: () => Promise<boolean>;
}

export const featureCategories: FeatureCategory[] = [
    { id: "quick-wins", name: "Quick Wins", description: "Try these in 30 seconds", order: 1 },
    { id: "speed", name: "10x Your Speed", description: "Efficiency boosters", order: 2 },
    { id: "code", name: "Level Up Your Code", description: "Dev workflows", order: 3 },
    { id: "collaborate", name: "Share & Collaborate", description: "Work with your team", order: 4 },
    { id: "customize", name: "Make It Yours", description: "Personalize Claude", order: 5 },
    { id: "power-user", name: "Power User", description: "Advanced features", order: 6 }
];

async function checkUsage(id: string): Promise<boolean> {
    const settings = getSettings("userSettings");
    const usage = settings.featureUsage || {};
    return !!usage[id] && usage[id] > 0;
}

export const featureTips: FeatureTip[] = [
    {
        id: "image-paste",
        name: "Paste Images",
        description: "Paste screenshots for Claude to analyze",
        categoryId: "quick-wins",
        tryItPrompt: "Press Ctrl+V to paste an image from clipboard",
        hasBeenUsed: () => checkUsage("image-paste")
    },
    {
        id: "resume",
        name: "Resume Conversations",
        description: "Pick up where you left off",
        categoryId: "quick-wins",
        tryItPrompt: "Type /resume to continue a past conversation",
        hasBeenUsed: () => checkUsage("resume")
    },
    {
        id: "cost",
        name: "Track Costs",
        description: "See your session spending",
        categoryId: "quick-wins",
        tryItPrompt: "Type /cost to see session cost",
        hasBeenUsed: () => checkUsage("cost")
    },
    {
        id: "slash-commands",
        name: "Slash Commands",
        description: "Quick actions with /commands",
        categoryId: "quick-wins",
        tryItPrompt: "Type / to see available commands",
        hasBeenUsed: () => checkUsage("slash-commands")
    },
    {
        id: "at-mentions",
        name: "@-mentions",
        description: "Reference files with @filename",
        categoryId: "quick-wins",
        tryItPrompt: "Type @ followed by a filename",
        hasBeenUsed: () => checkUsage("at-mentions")
    }
];

/**
 * Gets feature tips for a specific category.
 */
export function getTipsByCategory(categoryId: string): FeatureTip[] {
    return featureTips.filter(tip => tip.categoryId === categoryId);
}

/**
 * Gets a feature category by ID.
 */
export function getCategoryById(id: string): FeatureCategory | undefined {
    return featureCategories.find(cat => cat.id === id);
}
