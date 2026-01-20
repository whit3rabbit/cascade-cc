
// Logic from chunk_575.ts (Upgrade, Statusline, Asciichart)

import React from "react";
import { Box, Text } from "ink";
import { PermissionSelect } from "../../components/permissions/PermissionComponents.js";

// --- Rate Limit Options View (vZ7) ---
export function RateLimitOptionsView({ options, onSelect, onCancel }: any) {
    return (
        <Box
            flexDirection="column"
            borderStyle="round"
            borderColor="suggestion"
            paddingX={1}
            gap={1}
        >
            <Text>What do you want to do?</Text>
            <PermissionSelect
                options={options}
                onChange={onSelect}
                onCancel={onCancel}
            />
        </Box>
    );
}

// --- Asciichart Utility (iI9) ---
export const asciichart = {
    plot: (series: number[][], cfg: any = {}) => {
        return " [ASCII Chart stub] ";
    },
    colors: {
        red: "\x1B[31m",
        green: "\x1B[32m",
        blue: "\x1B[34m",
        reset: "\x1B[0m"
    }
};

// --- Analytics Lock Utility (pO0) ---
let analyticsLock: Promise<void> | null = null;
export async function withAnalyticsLock<T>(fn: () => Promise<T>): Promise<T> {
    while (analyticsLock) await analyticsLock;
    let resolveLock: (() => void) | undefined;
    analyticsLock = new Promise(r => { resolveLock = r; });
    try {
        return await fn();
    } finally {
        analyticsLock = null;
        resolveLock?.();
    }
}

// --- Initial Analytics Schema (BD1) ---
export function createInitialAnalyticsData() {
    return {
        version: 1,
        lastComputedDate: null,
        dailyActivity: [],
        dailyModelTokens: [],
        modelUsage: {},
        totalSessions: 0,
        totalMessages: 0,
        longestSession: null,
        firstSessionDate: null,
        hourCounts: {}
    };
}
