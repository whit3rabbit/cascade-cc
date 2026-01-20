import React from "react";
import { GitHubAppInstallWizard } from "../../components/terminal/GitHubAppInstallWizard.js";
import { isIdeTerminal } from "../../services/ide/IdeIntegration.js";
import { trackFeatureUsage } from "../../services/onboarding/usageTracker.js";
import { openInBrowser } from "../../utils/browser/BrowserUtils.js";
import { logStatsigEvent } from "../../services/telemetry/statsig.js";
import { updateSettings, getSettings } from "../../services/terminal/settings.js";

const SLACK_APP_INSTALL_URL = "https://slack.com/marketplace/A08SF47R6P4-claude";

/**
 * Command for setting up Claude GitHub Actions.
 * Deobfuscated from e47 in chunk_548.ts.
 */
export const InstallGithubAppCommand = {
    type: "local-jsx",
    name: "install-github-app",
    description: "Set up Claude GitHub Actions for a repository",
    isEnabled: () => !process.env.DISABLE_INSTALL_GITHUB_APP_COMMAND && !isIdeTerminal(),
    isHidden: false,
    async call(onDone: (message: string) => void) {
        trackFeatureUsage("github-app");
        logStatsigEvent("tengu_install_github_app_clicked", {});
        return <GitHubAppInstallWizard onDone={onDone} />;
    },
    userFacingName() {
        return "install-github-app";
    }
} as const;

/**
 * Command for installing the Claude Slack app.
 * Deobfuscated from A67 in chunk_548.ts.
 */
export const InstallSlackAppCommand = {
    type: "local",
    name: "install-slack-app",
    description: "Install the Claude Slack app",
    isEnabled: () => true,
    isHidden: false,
    supportsNonInteractive: false,
    async call() {
        trackFeatureUsage("slack-app");
        logStatsigEvent("tengu_install_slack_app_clicked", {});

        const currentSettings = getSettings("userSettings");
        updateSettings("userSettings", {
            slackAppInstallCount: (currentSettings.slackAppInstallCount ?? 0) + 1
        });

        if (await openInBrowser(SLACK_APP_INSTALL_URL)) {
            return {
                type: "text",
                value: "Opening Slack app installation page in browser…"
            };
        } else {
            return {
                type: "text",
                value: `Couldn't open browser. Visit: ${SLACK_APP_INSTALL_URL}`
            };
        }
    },
    userFacingName() {
        return "install-slack-app";
    }
} as const;
