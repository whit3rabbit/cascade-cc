// Logic from chunk_543.ts (Discover Claude Code UI)

import React, { useEffect, useMemo, useState } from "react";
import { Box, Text, useInput } from "ink";
import { figures } from "../../vendor/terminalFigures.js";
import { ProgressBar } from "./UsageView.js";
import { useCtrlExit } from "../../hooks/useCtrlExit.js";

const DISCOVER_CATEGORIES = [
    {
        id: "getting-started",
        name: "Getting Started",
        description: "Set up core workflows and basic commands."
    },
    {
        id: "automation",
        name: "Automation",
        description: "Automate tasks with tools and pipelines."
    },
    {
        id: "quality",
        name: "Quality",
        description: "Improve code quality with reviews and checks."
    }
];

const DISCOVER_FEATURES: Record<
    string,
    { id: string; name: string; description: string; tryItPrompt?: string; hasBeenUsed: () => Promise<boolean> }[]
> = {
    "getting-started": [
        {
            id: "first-command",
            name: "Run your first command",
            description: "Learn the basic prompt flow and tool usage.",
            tryItPrompt: "Try: /help",
            hasBeenUsed: async () => false
        }
    ],
    automation: [
        {
            id: "use-tools",
            name: "Automate with tools",
            description: "Use tools to execute tasks and gather context.",
            tryItPrompt: "Try: create a plan",
            hasBeenUsed: async () => false
        }
    ],
    quality: [
        {
            id: "review",
            name: "Review output",
            description: "Ask for reviews and diagnostics to keep work clean.",
            tryItPrompt: "Try: review my changes",
            hasBeenUsed: async () => false
        }
    ]
};

type DiscoverStats = {
    explored: number;
    total: number;
    byCategory: Record<string, { explored: number; total: number }>;
};

type DiscoverCategory = {
    id: string;
    name: string;
    description: string;
};

type DiscoverFeature = {
    id: string;
    name: string;
    description: string;
    tryItPrompt?: string;
    hasBeenUsed: () => Promise<boolean>;
};

function getDiscoverCategories(): DiscoverCategory[] {
    return DISCOVER_CATEGORIES;
}

function getCategoryById(categoryId: string): DiscoverCategory | null {
    return DISCOVER_CATEGORIES.find((category) => category.id === categoryId) ?? null;
}

function getFeaturesForCategory(categoryId: string): DiscoverFeature[] {
    return DISCOVER_FEATURES[categoryId] ?? [];
}

async function getDiscoverStats(): Promise<DiscoverStats> {
    const categories = getDiscoverCategories();
    const byCategory: Record<string, { explored: number; total: number }> = {};
    let total = 0;
    let explored = 0;

    for (const category of categories) {
        const features = getFeaturesForCategory(category.id);
        total += features.length;
        const exploredCount = 0;
        explored += exploredCount;
        byCategory[category.id] = { explored: exploredCount, total: features.length };
    }

    return { explored, total, byCategory };
}

// --- Discover Feature Row (T39) ---
export function DiscoverFeatureRow({
    feature,
    isUsed,
    isFocused
}: {
    feature: DiscoverFeature;
    isUsed: boolean;
    isFocused: boolean;
}) {
    const icon = isUsed ? figures.tick : figures.circle;
    const color = isUsed ? "success" : "inactive";
    const focusColor = isFocused ? "suggestion" : undefined;

    return (
        <Box flexDirection="column">
            <Box gap={1}>
                <Text color={focusColor}>{isFocused ? `${figures.pointer} ` : "  "}</Text>
                <Text color={color}>{icon}</Text>
                <Text color={focusColor} bold={isFocused}>
                    {feature.name}
                </Text>
            </Box>
            {isFocused && (
                <Box flexDirection="column" marginLeft={4}>
                    <Text dimColor>{feature.description}</Text>
                    {!isUsed && feature.tryItPrompt && (
                        <Text color="warning" dimColor>
                            Try it: {feature.tryItPrompt}
                        </Text>
                    )}
                </Box>
            )}
        </Box>
    );
}

// --- Discover Category Selector (_39) ---
export function DiscoverCategorySelector({
    categories,
    stats,
    onSelect
}: {
    categories: DiscoverCategory[];
    stats: DiscoverStats["byCategory"] | null;
    onSelect: (categoryId: string) => void;
}) {
    const [selectedIndex, setSelectedIndex] = useState(0);

    useInput((input, key) => {
        if (key.upArrow || input === "k") setSelectedIndex((index) => (index > 0 ? index - 1 : categories.length - 1));
        else if (key.downArrow || input === "j") setSelectedIndex((index) => (index < categories.length - 1 ? index + 1 : 0));
        else if (key.return) {
            const category = categories[selectedIndex];
            if (category) onSelect(category.id);
        } else if (input >= "1" && input <= "9") {
            const index = parseInt(input, 10) - 1;
            if (index < categories.length) {
                setSelectedIndex(index);
                const category = categories[index];
                if (category) onSelect(category.id);
            }
        }
    });

    return (
        <Box flexDirection="column">
            {categories.map((category, index) => {
                const isFocused = index === selectedIndex;
                const stat = stats?.[category.id] || { explored: 0, total: 0 };
                const explored = stat.explored ?? 0;
                const total = stat.total ?? 0;
                let statusIcon = figures.circle;
                let statusColor = "inactive";
                if (explored === 0) {
                    statusIcon = figures.circle;
                    statusColor = "inactive";
                } else if (explored === total) {
                    statusIcon = figures.tick;
                    statusColor = "success";
                } else {
                    statusIcon = figures.circleFilled;
                    statusColor = "warning";
                }

                return (
                    <Box key={category.id} gap={1}>
                        <Text color={isFocused ? "suggestion" : undefined}>
                            {isFocused ? figures.pointer : " "}
                        </Text>
                        <Text color={statusColor}>{statusIcon}</Text>
                        <Box width={24}>
                            <Text color={isFocused ? "suggestion" : undefined} bold={isFocused}>
                                {category.name}
                            </Text>
                        </Box>
                        <Text dimColor>
                            [{explored}/{total} {explored === total ? "completed" : explored === 0 ? "unexplored" : "explored"}]
                        </Text>
                    </Box>
                );
            })}
        </Box>
    );
}

// --- Discover Category View (S39) ---
export function DiscoverCategoryView({
    categoryId,
    onBack,
    onClose
}: {
    categoryId: string;
    onBack: () => void;
    onClose: () => void;
}) {
    const [selectedIndex, setSelectedIndex] = useState(0);
    const [usageMap, setUsageMap] = useState<Record<string, boolean>>({});
    const exitState = useCtrlExit();
    const category = getCategoryById(categoryId);
    const features = useMemo(() => getFeaturesForCategory(categoryId), [categoryId]);

    useEffect(() => {
        Promise.all(features.map(async (feature) => [feature.id, await feature.hasBeenUsed()] as const)).then(
            (entries) => {
                setUsageMap(Object.fromEntries(entries));
            }
        );
    }, [features]);

    useInput((input, key) => {
        if (key.escape) onClose();
        else if (key.backspace || key.delete) onBack();
        else if (key.upArrow || input === "k") setSelectedIndex((index) => (index > 0 ? index - 1 : features.length - 1));
        else if (key.downArrow || input === "j") setSelectedIndex((index) => (index < features.length - 1 ? index + 1 : 0));
    });

    if (!category) return <Text color="error">Category not found</Text>;

    return (
        <Box flexDirection="column" paddingBottom={1}>
            <Box>
                <Text dimColor>{"-".repeat(40)}</Text>
            </Box>
            <Box flexDirection="column" paddingX={1} gap={1}>
                <Box flexDirection="column">
                    <Text bold color="suggestion">
                        {category.name}
                    </Text>
                    <Text dimColor>{category.description}</Text>
                </Box>
                <Box flexDirection="column">
                    {features.map((feature, index) => (
                        <DiscoverFeatureRow
                            key={feature.id}
                            feature={feature}
                            isUsed={usageMap[feature.id] ?? false}
                            isFocused={index === selectedIndex}
                        />
                    ))}
                </Box>
            </Box>
            <Box paddingX={1}>
                <Text dimColor italic>
                    {exitState.pending ? (
                        <>Press {exitState.keyName} again to exit</>
                    ) : (
                        "↑/↓ navigate · Backspace back · Esc close"
                    )}
                </Text>
            </Box>
        </Box>
    );
}

// --- Discover View (y39) ---
export function DiscoverView({ onClose }: { onClose: () => void }) {
    const [selectedCategory, setSelectedCategory] = useState<string | null>(null);
    const [stats, setStats] = useState<DiscoverStats | null>(null);
    const exitState = useCtrlExit();

    useEffect(() => {
        getDiscoverStats().then(setStats);
    }, [selectedCategory]);

    useInput((_input, key) => {
        if (key.escape && !selectedCategory) onClose();
    });

    if (selectedCategory) {
        return (
            <DiscoverCategoryView
                categoryId={selectedCategory}
                onBack={() => setSelectedCategory(null)}
                onClose={onClose}
            />
        );
    }

    const ratio = stats ? stats.explored / Math.max(stats.total, 1) : 0;
    const percent = stats ? Math.round((stats.explored / Math.max(stats.total, 1)) * 100) : 0;

    return (
        <Box flexDirection="column" paddingBottom={1}>
            <Box>
                <Text dimColor>{"-".repeat(40)}</Text>
            </Box>
            <Box flexDirection="column" paddingX={1} gap={1}>
                <Box flexDirection="column">
                    <Text bold color="suggestion">
                        Discover Claude Code
                    </Text>
                    <Text dimColor>Explore features and track your progress</Text>
                </Box>
                {stats && (
                    <Box flexDirection="column" gap={0}>
                        <Text>
                            You've explored <Text bold color="success">{stats.explored}</Text> of {stats.total} features ({percent}%)
                        </Text>
                        <Box>
                            <ProgressBar ratio={ratio} width={40} />
                        </Box>
                    </Box>
                )}
                <DiscoverCategorySelector
                    categories={getDiscoverCategories()}
                    stats={stats?.byCategory ?? null}
                    onSelect={setSelectedCategory}
                />
            </Box>
            <Box paddingX={1}>
                <Text dimColor italic>
                    {exitState.pending ? (
                        <>Press {exitState.keyName} again to exit</>
                    ) : (
                        "↑/↓ navigate · Enter explore · Esc close"
                    )}
                </Text>
            </Box>
        </Box>
    );
}

export const DiscoverCommand = {
    type: "local-jsx",
    name: "discover",
    description: "Explore Claude Code features and track your progress",
    isEnabled: () => true,
    isHidden: false,
    async call(onClose: () => void) {
        return <DiscoverView onClose={onClose} />;
    },
    userFacingName() {
        return "discover";
    }
};
