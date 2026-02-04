/**
 * File: src/commands/help.tsx
 * Role: Implementation of /help and /mobile commands.
 */

import React, { useState } from 'react';
import { Box, Text, useInput } from 'ink';
import { commandRegistry } from '../services/terminal/CommandRegistry.js';
import { PRODUCT_NAME } from '../constants/product.js';
import { BUILD_INFO } from '../constants/build.js';

interface ComponentProps {
    onDone: () => void;
}

/**
 * Renders the help information for all registered commands.
 */
function HelpComponent({ onDone }: ComponentProps) {
    const commands = commandRegistry.getAllCommands();

    useInput((input, key) => {
        if (key.escape || input === 'q') {
            onDone();
        }
    });

    return (
        <Box flexDirection="column" paddingX={2} paddingY={1}>
            <Text bold color="cyan">{PRODUCT_NAME} v{BUILD_INFO.VERSION}</Text>
            <Box marginTop={1} flexDirection="column">
                <Text bold underline>Available Commands:</Text>
                {commands
                    .filter(cmd => !cmd.isHidden)
                    .sort((a, b) => a.name.localeCompare(b.name))
                    .map(cmd => (
                        <Box key={cmd.name} marginTop={0}>
                            <Text color="yellow">/{cmd.name.padEnd(15)}</Text>
                            <Text>{cmd.description}</Text>
                        </Box>
                    ))
                }
            </Box>
            <Box marginTop={1}>
                <Text dimColor>For more information, visit {BUILD_INFO.README_URL}</Text>
                <Text dimColor italic> (Press Esc or 'q' to close)</Text>
            </Box>
        </Box>
    );
}


import { createCommandHelper } from './helpers.js';

export const helpCommandDefinition = createCommandHelper("help", "Show available commands and help", {
    type: "local-jsx",
    async call(onDone: () => void) {
        return <HelpComponent onDone={onDone} />;
    },
    userFacingName() {
        return "help";
    }
});

/**
 * Renders a display for the mobile app download links.
 */
function MobileComponent({ onDone }: ComponentProps) {
    const [platform, setPlatform] = useState<"ios" | "android">("ios");

    useInput((input, key) => {
        if (key.escape || input === 'q') {
            onDone();
        }
        if (key.tab || key.leftArrow || key.rightArrow) {
            setPlatform(p => (p === "ios" ? "android" : "ios"));
        }
    });

    const urls = {
        ios: "https://apps.apple.com/app/claude-by-anthropic/id6473753684",
        android: "https://play.google.com/store/apps/details?id=com.anthropic.claude"
    };

    return (
        <Box flexDirection="column" paddingX={2} paddingY={1}>
            <Text bold color="magenta">Claude Mobile App</Text>

            <Box marginTop={1} flexDirection="row">
                <Box borderStyle="round" borderColor={platform === "ios" ? "cyan" : "gray"} paddingX={1}>
                    <Text bold={platform === "ios"}>iOS</Text>
                </Box>
                <Box marginLeft={2} borderStyle="round" borderColor={platform === "android" ? "cyan" : "gray"} paddingX={1}>
                    <Text bold={platform === "android"}>Android</Text>
                </Box>
            </Box>

            <Box marginTop={1} flexDirection="column">
                <Text>Download link for {platform === "ios" ? "App Store" : "Play Store"}:</Text>
                <Text color="blue" underline>{urls[platform]}</Text>
            </Box>

            <Box marginTop={1} padding={1} borderStyle="round" borderColor="yellow" flexDirection="column" alignItems="center">
                <Text color="yellow" bold>Scan to download</Text>
                <Box marginTop={1}>
                    <Text>
                        {"█▀▀▀▀▀█ ▄ ▄  █▀▀▀▀▀█\n"}
                        {"█ ███ █ ▀ █▀ █ ███ █\n"}
                        {"█ ▀▀▀ █ ▄▀█▀ █ ▀▀▀ █\n"}
                        {"▀▀▀▀▀▀▀ █▄▀ ▀▀▀▀▀▀▀▀\n"}
                        {"▀▀▄▀▀▄▀▄▀▀ ▀▄▄▀▄ ▀▀ \n"}
                        {"█▀▀▀▀▀█ ▀█▀█▀  ▀▀▀ ▄\n"}
                        {"█ ███ █ █ ▀ █▀▀ ▀▄▀▀\n"}
                        {"█ ▀▀▀ █ ▀█ ▄ █▄█ ▀▄ \n"}
                        {"▀▀▀▀▀▀▀ ▀  ▀▀  ▀▀▀▀▀"}
                    </Text>
                </Box>
                <Box marginTop={1}>
                    <Text dimColor>({platform === 'ios' ? 'iOS App Store' : 'Android Play Store'})</Text>
                </Box>
            </Box>

            <Box marginTop={1}>
                <Text dimColor italic>Tab to switch platform • Esc to close</Text>
            </Box>
        </Box>
    );
}

export const mobileCommandDefinition = createCommandHelper("mobile", "Download the Claude mobile app", {
    aliases: ["ios", "android"],
    type: "local-jsx",
    async call(onDone: () => void) {
        return <MobileComponent onDone={onDone} />;
    },
    userFacingName() {
        return "mobile";
    }
});