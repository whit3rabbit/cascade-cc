
import React from "react";
import { GitHubAppInstallWizard } from "./GitHubAppInstallWizard.js";
import { openInBrowser } from "../../utils/browser/BrowserUtils.js";

const SLACK_INSTALL_URL = "https://slack.com/marketplace/A08SF47R6P4-claude";
const installStats = { slackAppInstallCount: 0 };

function trackEvent(_name: string, _payload?: Record<string, any>) {}
function setInstallSource(_source: string) {}
function isGitHubInstallCommandDisabled() {
    return false;
}

function updateInstallStats(updater: (current: typeof installStats) => typeof installStats) {
    const next = updater(installStats);
    installStats.slackAppInstallCount = next.slackAppInstallCount;
}

export const InstallGitHubAppCommand = {
    type: "local-jsx",
    name: "install-github-app",
    description: "Set up Claude GitHub Actions for a repository",
    isEnabled: () => !process.env.DISABLE_INSTALL_GITHUB_APP_COMMAND && !isGitHubInstallCommandDisabled(),
    isHidden: false,
    async call(_args: any, context: any) {
        setInstallSource("github-app");
        return <GitHubAppInstallWizard onDone={context.onDone} />;
    },
    userFacingName() {
        return "install-github-app";
    }
};

export const InstallSlackAppCommand = {
    type: "local",
    name: "install-slack-app",
    description: "Install the Claude Slack app",
    isEnabled: () => true,
    isHidden: false,
    supportsNonInteractive: false,
    async call(_args: any, _context: any) {
        setInstallSource("slack-app");
        trackEvent("tengu_install_slack_app_clicked", {});
        updateInstallStats((current) => ({
            ...current,
            slackAppInstallCount: (current.slackAppInstallCount ?? 0) + 1
        }));
        const opened = await openInBrowser(SLACK_INSTALL_URL);
        if (opened) {
            return {
                type: "text",
                value: "Opening Slack app installation page in browser…"
            };
        }
        return {
            type: "text",
            value: `Couldn't open browser. Visit: ${SLACK_INSTALL_URL}`
        };
    },
    userFacingName() {
        return "install-slack-app";
    }
};
