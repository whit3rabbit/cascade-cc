
// Logic from chunk_559.ts (Built-in Tools & specialized skills)

import React from "react";
import { Box, Text } from "ink";
import { ThemeSelector } from "../../components/terminal/StatusView.js";
import { UsagePanel as UsageView } from "../../components/terminal/UsageView.js";

/**
 * Vim mode toggle tool (p77).
 */
export const VimModeTool = {
    name: "vim",
    description: "Toggle between Vim and Normal editing modes",
    type: "local",
    userFacingName: () => "vim",
    async call() {
        // In actual implementation, this toggles settings
        const newMode = "vim"; // Mock toggle
        return {
            type: "text",
            value: `Editor mode set to ${newMode}. Use Escape key to toggle between INSERT and NORMAL modes.`
        };
    }
};

/**
 * Theme selector tool (u77).
 */
export const ThemeTool = {
    name: "theme",
    description: "Change the theme",
    type: "local-jsx",
    userFacingName: () => "theme",
    async call(onDone: any) {
        return (
            <Box flexDirection="column">
                <Text bold color="permission">Select a theme:</Text>
                <ThemeSelector
                    onThemeSelect={(theme: string) => onDone(`Theme set to ${theme}`)}
                    onCancel={() => onDone("Theme picker dismissed")}
                />
            </Box>
        );
    }
};

/**
 * Usage tool (CY9).
 */
export const UsageTool = {
    name: "usage",
    description: "Show plan usage limits",
    type: "local-jsx",
    userFacingName: () => "usage",
    async call(onDone: any) {
        return <UsageView onDone={onDone} />;
    }
};

/**
 * Security Review Skill (EY9).
 * Contains the large security analysis prompt.
 */
export const SecurityReviewSkill = {
    name: "security-review",
    description: "Complete a security review of the pending changes on the current branch",
    type: "prompt",
    pluginName: "security-review",
    async getPrompt() {
        return `You are an expert security researcher... [Prompts truncated for brevity]`;
    }
};
