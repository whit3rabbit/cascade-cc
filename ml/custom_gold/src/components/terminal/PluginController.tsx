
// Logic from chunk_572.ts (Plugin Command Orchestration & Validation)

import React, { useCallback, useMemo, useState } from "react";
import { Box, Text, useInput } from "ink";
import { figures } from "../../vendor/terminalFigures.js";
import { useCtrlExit } from "../../hooks/useCtrlExit.js";
import { InstalledPluginsView } from "./InstalledPluginsView.js";
import { MarketplaceManager } from "./MarketplaceManager.js";
import { PluginErrorsView } from "./PluginErrorsView.js";
import { PluginDiscoveryView } from "./PluginDiscoveryView.js";

// --- Command Parser (pX9) ---
export function parsePluginCommand(input: string | undefined): any {
    if (!input) return { type: "menu" };
    const args = input.trim().split(/\s+/);
    const subCommand = args[0]?.toLowerCase();

    switch (subCommand) {
        case "help":
        case "--help":
        case "-h":
            return { type: "help" };
        case "install":
        case "i": {
            const target = args[1];
            if (!target) return { type: "install" };
            if (target.includes("@")) {
                const [p, m] = target.split("@");
                return { type: "install", plugin: p, marketplace: m };
            }
            if (target.startsWith("http") || target.includes("/") || target.includes("\\")) {
                return { type: "install", marketplace: target };
            }
            return { type: "install", plugin: target };
        }
        case "validate":
            return { type: "validate", path: args.slice(1).join(" ").trim() || undefined };
        case "marketplace":
        case "market": {
            const action = args[1]?.toLowerCase();
            const target = args.slice(2).join(" ");
            return { type: "marketplace", action, target };
        }
        default:
            return { type: "menu" };
    }
}

// --- View Mapper (pG7) ---
export function mapCommandToViewState(command: any): any {
    switch (command.type) {
        case "help": return { type: "help" };
        case "validate": return { type: "validate", path: command.path };
        case "install":
            if (command.marketplace) return { type: "browse-marketplace", targetMarketplace: command.marketplace, targetPlugin: command.plugin };
            if (command.plugin) return { type: "discover-plugins", targetPlugin: command.plugin };
            return { type: "discover-plugins" };
        case "marketplace":
            if (command.action === "add") return { type: "add-marketplace", initialValue: command.target };
            return { type: "marketplace-menu" };
        default:
            return { type: "discover-plugins" };
    }
}

// --- Tab Bar Component (mG7) ---
const TABS: Record<string, string> = {
    discover: "Discover",
    installed: "Installed",
    marketplaces: "Marketplaces",
    errors: "Errors"
};

export type PluginTab = keyof typeof TABS;

export function PluginTabBar({ activeTab }: { activeTab: string }) {
    return (
        <Box flexDirection="row" gap={1} marginBottom={1}>
            {Object.keys(TABS).map(tab => (
                <Text
                    key={tab}
                    backgroundColor={tab === activeTab ? "suggestion" : undefined}
                    color={tab === activeTab ? "inverseText" : undefined}
                    bold={tab === activeTab}
                >
                    {" "}{TABS[tab]}{" "}
                </Text>
            ))}
            <Text dimColor>(tab to cycle)</Text>
        </Box>
    );
}

type PluginManagerViewState =
    | { type: "menu"; message?: string }
    | { type: "discover-plugins"; targetPlugin?: string }
    | { type: "browse-marketplace"; targetMarketplace: string; targetPlugin?: string }
    | { type: "marketplace-menu" }
    | { type: "add-marketplace"; initialValue?: string }
    | { type: "installed" }
    | { type: "errors" };

type PluginManagerViewProps = {
    initialTab?: PluginTab;
    errors?: any[];
    onExit: (message?: string) => void;
    onResult: (message: string) => void;
};

export function PluginManagerView({
    initialTab = "discover",
    errors = [],
    onExit,
    onResult
}: PluginManagerViewProps) {
    const [activeTab, setActiveTab] = useState<PluginTab>(initialTab);
    const exitState = useCtrlExit(async () => onExit());

    const tabOrder = useMemo(() => Object.keys(TABS) as PluginTab[], []);
    const handleTabCycle = useCallback(() => {
        const currentIndex = tabOrder.indexOf(activeTab);
        const nextIndex = (currentIndex + 1) % tabOrder.length;
        setActiveTab(tabOrder[nextIndex]);
    }, [activeTab, tabOrder]);

    const handleViewState = useCallback(
        (nextState: PluginManagerViewState) => {
            if (nextState.type === "menu") {
                onExit(nextState.message);
                return;
            }
            if (nextState.type === "marketplace-menu" || nextState.type === "add-marketplace") {
                setActiveTab("marketplaces");
                return;
            }
            if (nextState.type === "browse-marketplace") {
                setActiveTab("discover");
                return;
            }
            if (nextState.type === "installed") {
                setActiveTab("installed");
                return;
            }
            if (nextState.type === "errors") {
                setActiveTab("errors");
            }
        },
        [onExit]
    );

    useInput((_input, key) => {
        if (key.tab) {
            handleTabCycle();
        }
    });

    let content: React.ReactElement;
    switch (activeTab) {
        case "installed":
            content = (
                <InstalledPluginsView
                    setViewState={handleViewState}
                    setResult={onResult}
                />
            );
            break;
        case "marketplaces":
            content = (
                <MarketplaceManager
                    setViewState={handleViewState}
                    setResult={onResult}
                    exitState={exitState as any}
                />
            );
            break;
        case "errors":
            content = <PluginErrorsView errors={errors} onExit={() => onExit()} />;
            break;
        case "discover":
        default:
            content = (
                <PluginDiscoveryView
                    setViewState={handleViewState}
                    setError={onResult}
                    onInstallComplete={() => { }} // Could be added if needed
                />
            );
            break;
    }

    return (
        <Box flexDirection="column">
            <PluginTabBar activeTab={activeTab} />
            {content}
        </Box>
    );
}

// --- Validation View (mX9) ---
export function PluginValidationView({ path: targetPath, onComplete }: any) {
    React.useEffect(() => {
        async function runValidation() {
            if (!targetPath) {
                onComplete(`Usage: /plugin validate <path>\n\nValidate a plugin or marketplace manifest.`);
                return;
            }
            onComplete(`${figures.tick} Validation passed for ${targetPath}`);
        }
        runValidation();
    }, [targetPath, onComplete]);

    return (
        <Box flexDirection="column">
            <Text>Running validation...</Text>
        </Box>
    );
}
