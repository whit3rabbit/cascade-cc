
// Logic from chunk_578.ts (Usage Statistics & Analytics UI)

import React from "react";
import { Box, Text } from "ink";
import { figures } from "../../vendor/terminalFigures.js";
import { plot as asciichart } from "../../utils/shared/asciichart.js";

// --- Analogies for Token Counts ---
export const TOKEN_ANALOGIES = [
    { name: "The Old Man and the Sea", tokens: 35000 },
    { name: "Animal Farm", tokens: 39000 },
    { name: "The Great Gatsby", tokens: 62000 },
    { name: "Brave New World", tokens: 83000 },
    { name: "The Hobbit", tokens: 103000 },
    { name: "1984", tokens: 123000 },
    { name: "To Kill a Mockingbird", tokens: 130000 },
    { name: "Pride and Prejudice", tokens: 156000 },
    { name: "Anna Karenina", tokens: 468000 },
    { name: "Don Quixote", tokens: 520000 },
    { name: "The Lord of the Rings", tokens: 576000 },
    { name: "War and Peace", tokens: 730000 }
];

// --- Analogies for Time Spent ---
export const TIME_ANALOGIES = [
    { name: "a TED talk", minutes: 18 },
    { name: "an episode of The Office", minutes: 22 },
    { name: "a half marathon", minutes: 120 },
    { name: "the movie Inception", minutes: 148 },
    { name: "a transatlantic flight", minutes: 420 }
];

// --- Stats Row Helper (X) ---
function StatsRow({ label1, value1, label2, value2 }: any) {
    return (
        <Box flexDirection="row">
            <Box width={20}>
                <Text dimColor>{label1}: </Text>
                <Text bold>{value1}</Text>
            </Box>
            <Box>
                <Text dimColor>{label2}: </Text>
                <Text bold>{value2}</Text>
            </Box>
        </Box>
    );
}

// --- Model Usage Formatter (_Y7) ---
export function ModelStatsView({ stats }: any) {
    const modelEntries = Object.entries(stats.modelUsage || {}).sort(([, a]: any, [, b]: any) =>
        (b.inputTokens + b.outputTokens) - (a.inputTokens + a.outputTokens)
    );

    if (modelEntries.length === 0) {
        return <Text dimColor>No model usage data available</Text>;
    }

    const totalTokens = modelEntries.reduce((sum, [, usage]: any) => sum + usage.inputTokens + usage.outputTokens, 0);
    const favoriteModel: any = modelEntries[0];

    return (
        <Box flexDirection="column" gap={1}>
            <Text bold underline>Model Usage Breakdown</Text>
            <Text>
                {figures.star} Favorite: <Text color="magenta" bold>{favoriteModel[0]}</Text>
                {"  "}
                {figures.circle} Total: <Text color="magenta">{totalTokens.toLocaleString()}</Text> tokens
            </Text>
            {modelEntries.slice(0, 3).map(([model, usage]: any) => {
                const percentage = (((usage.inputTokens + usage.outputTokens) / totalTokens) * 100).toFixed(1);
                return (
                    <Box key={model} flexDirection="column" marginLeft={2}>
                        <Text>
                            {figures.bullet} <Text bold>{model}</Text> <Text dimColor>({percentage}%)</Text>
                        </Text>
                        <Text dimColor>
                            {"  "}In: {usage.inputTokens.toLocaleString()} · Out: {usage.outputTokens.toLocaleString()}
                        </Text>
                    </Box>
                );
            })}
        </Box>
    );
}

// --- Main Stats View (zW9) ---
export function StatsView({ stats, onClose }: { stats: any, onClose: () => void }) {
    const totalTokens = Object.values(stats.modelUsage || {}).reduce((sum: number, usage: any) => sum + usage.inputTokens + usage.outputTokens, 0);

    return (
        <Box flexDirection="column" padding={1} borderStyle="round" borderColor="suggestion">
            <Box marginBottom={1}>
                <Text bold color="claude">Claude Code Usage Statistics</Text>
            </Box>

            <Box flexDirection="column" gap={1}>
                <StatsRow
                    label1="Sessions" value1={stats.totalSessions}
                    label2="Longest session" value2={stats.longestSession?.duration || "N/A"}
                />
                <StatsRow
                    label1="Current streak" value1={`${stats.streaks.currentStreak} days`}
                    label2="Longest streak" value2={`${stats.streaks.longestStreak} days`}
                />
                <StatsRow
                    label1="Active days" value1={`${stats.activeDays}/${stats.totalDays}`}
                    label2="Peak hour" value2={stats.peakActivityHour ? `${stats.peakActivityHour}:00` : "N/A"}
                />
            </Box>

            <Box marginTop={1}>
                <ModelStatsView stats={stats} />
            </Box>

            <Box marginTop={1}>
                <Text dimColor italic>Stats from the last {stats.totalDays} days</Text>
            </Box>

            <Box marginTop={1}>
                <Text dimColor>Press any key to close · Esc to exit</Text>
            </Box>
        </Box>
    );
}
