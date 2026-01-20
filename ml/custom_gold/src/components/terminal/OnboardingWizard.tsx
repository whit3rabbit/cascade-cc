
// Logic from chunk_595.ts (Onboarding Engine & Security Variants)

import React, { useState, useMemo } from "react";
import { Box, Text } from "ink";
import { ConnectivityCheckView, WelcomeBanner } from "./OnboardingViews.js";
import { Select } from "./MarketplaceManager.js";

/**
 * Main Onboarding Wizard (rH9)
 */
export function OnboardingWizard({ onDone }: { onDone: () => void }) {
    const [step, setStep] = useState(0);
    const [theme, setTheme] = useState("dark");

    const next = () => setStep(s => s + 1);

    const steps = [
        {
            id: "preflight",
            component: <ConnectivityCheckView onSuccess={next} />
        },
        {
            id: "theme",
            component: (
                <Box flexDirection="column">
                    <Text bold>Choose your theme:</Text>
                    <Select
                        options={[
                            { label: "Dark", value: "dark" },
                            { label: "Light", value: "light" }
                        ]}
                        onChange={(val) => { setTheme(val); next(); }}
                    />
                </Box>
            )
        },
        {
            id: "security",
            component: (
                <Box flexDirection="column" gap={1}>
                    <Text bold>Security note:</Text>
                    <Text dimColor>Claude can make mistakes. Always review code before running it.</Text>
                    <Text bold>Do you trust the files in this folder?</Text>
                    <Select
                        options={[
                            { label: "Yes, I trust this folder", value: "yes" },
                            { label: "No, exit", value: "no" }
                        ]}
                        onChange={(val) => { if (val === "no") process.exit(0); next(); }}
                    />
                </Box>
            )
        }
    ];

    if (step >= steps.length) {
        onDone();
        return null;
    }

    return (
        <Box flexDirection="column">
            <WelcomeBanner />
            <Box marginTop={1}>
                {steps[step].component}
            </Box>
        </Box>
    );
}

/**
 * Security trust variants (jH9).
 */
export const SecurityTrustVariants = {
    control: {
        title: "Do you trust the files in this folder?",
        yes: "Yes, proceed",
        no: "No, exit"
    },
    explicit: {
        title: "Do you want to work in this folder?",
        body: "Only continue if this is your code or a project you trust.",
        yes: "Yes, continue",
        no: "No, exit"
    }
};
