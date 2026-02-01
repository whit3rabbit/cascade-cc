import { EnvService } from '../config/EnvService.js';
import { getSettings, updateSettings } from '../config/SettingsService.js';

export interface ActivityTip {
    id: string;
    content: (context: any) => Promise<string> | string;
    cooldownSessions: number;
    isRelevant: (context: any) => Promise<boolean>;
}

export class ActivityTips {
    private static tips: ActivityTip[] = [
        {
            id: "new-user-warmup",
            content: async () => "Start with small features or bug fixes, tell Claude to propose a plan, and verify its suggested edits",
            cooldownSessions: 3,
            isRelevant: async () => {
                const settings = getSettings();
                return (settings.numStartups || 0) < 10;
            }
        },
        {
            id: "plan-mode-for-complex-tasks",
            content: async () => `Use Plan Mode to prepare for a complex request before making changes.`,
            cooldownSessions: 5,
            isRelevant: async () => {
                const settings = getSettings();
                const lastPlanModeUse = settings.lastPlanModeUse;
                const daysSinceUse = lastPlanModeUse ? (Date.now() - lastPlanModeUse) / 86400000 : Infinity;
                return daysSinceUse > 7;
            }
        },
        {
            id: "git-worktrees",
            content: async () => "Use git worktrees to run multiple Claude sessions in parallel.",
            cooldownSessions: 10,
            isRelevant: async () => {
                const settings = getSettings();
                return (settings.numStartups || 0) > 50;
            }
        },
        {
            id: "memory-command",
            content: async () => "Use /memory to view and manage Claude memory",
            cooldownSessions: 15,
            isRelevant: async () => {
                const settings = getSettings();
                return (settings.memoryUsageCount || 0) <= 0;
            }
        },
        {
            id: "theme-command",
            content: async () => "Use /theme to change the color theme",
            cooldownSessions: 20,
            isRelevant: async () => true
        },
        {
            id: "colorterm-truecolor",
            content: async () => "Try setting environment variable COLORTERM=truecolor for richer colors",
            cooldownSessions: 30,
            isRelevant: async () => !process.env.COLORTERM
        },
        {
            id: "todo-list",
            content: async () => "Ask Claude to create a todo list when working on complex tasks to track progress and remain on track",
            cooldownSessions: 20,
            isRelevant: async () => true
        },
        {
            id: "permissions",
            content: async () => "Use /permissions to pre-approve and pre-deny bash, edit, and MCP tools",
            cooldownSessions: 10,
            isRelevant: async () => {
                const settings = getSettings();
                return (settings.numStartups || 0) > 10;
            }
        },
        {
            id: "continue",
            content: async () => "Run claude --continue or claude --resume to resume a conversation",
            cooldownSessions: 10,
            isRelevant: async () => true
        }
    ];

    public static async getRefreshedTip(context: any): Promise<ActivityTip | null> {
        const settings = getSettings();
        const tipsHistory = settings.tipsHistory || {};
        const numStartups = settings.numStartups || 0;

        // Shuffle and check relevance
        const shuffled = [...ActivityTips.tips].sort(() => Math.random() - 0.5);

        // Filter relevant tips that are not on cooldown
        const relevantTips = [];
        for (const tip of shuffled) {
            const lastSeenStartup = tipsHistory[tip.id] || 0;
            // If never seen (0), sessionsSince = Infinity. 
            // If seen at startup 5, and now is 10, sessionsSince = 5.
            const sessionsSince = lastSeenStartup === 0 ? Infinity : numStartups - lastSeenStartup;

            if (sessionsSince >= tip.cooldownSessions) {
                if (await tip.isRelevant(context)) {
                    relevantTips.push(tip);
                }
            }
        }

        if (relevantTips.length === 0) return null;

        const selected = relevantTips[0];

        // Update history
        const newHistory = { ...tipsHistory, [selected.id]: numStartups };
        updateSettings({ tipsHistory: newHistory });

        return selected;
    }
}
