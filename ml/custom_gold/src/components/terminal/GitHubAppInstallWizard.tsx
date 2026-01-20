
import React, { useCallback, useEffect, useRef, useState } from "react";
import { execFile, execSync } from "child_process";
import { promisify } from "util";
import { useCtrlExit } from "../../hooks/useCtrlExit.js";
import { setupGithubActions } from "../../services/terminal/GitHubActionSetupService.js";
import { openInBrowser } from "../../utils/browser/BrowserUtils.js";
import { getApiKeyWithSource } from "../../services/auth/apiKeyManager.js";
import { isInteractiveAuthRequired } from "../../services/auth/authSource.js";
import { GitHubCliCheckView, GitHubRepoSelection } from "../../tools/commands/InitCommand.js";
import {
    APIKeySecretChoiceView,
    APIKeySelectionView,
    ExistingWorkflowDialog,
    GitHubAppInstallPrompt,
    InstallationErrorView,
    InstallationProgressView,
    InstallationSuccessView,
    MANUAL_SETUP_URL
} from "./GitHubAppInstallView.js";
import { InstallWarningsView, WorkflowSelectionView, type InstallWarning } from "./GitHubAppInstallSteps.js";
import { GitHubOAuthWizard } from "./GitHubOAuthWizard.js";

type InstallStep =
    | "check-gh"
    | "warnings"
    | "choose-repo"
    | "install-app"
    | "check-existing-workflow"
    | "select-workflows"
    | "check-existing-secret"
    | "api-key"
    | "creating"
    | "success"
    | "error"
    | "oauth-flow";

type InstallState = {
    step: InstallStep;
    selectedRepoName: string;
    currentRepo: string;
    useCurrentRepo: boolean;
    apiKeyOrOAuthToken: string;
    useExistingKey: boolean;
    currentWorkflowInstallStep: number;
    warnings: InstallWarning[];
    secretExists: boolean;
    secretName: string;
    useExistingSecret: boolean;
    workflowExists: boolean;
    selectedWorkflows: string[];
    selectedApiKeyOption: "existing" | "oauth" | "new";
    authType: "api_key" | "oauth_token";
    workflowAction?: "update" | "skip" | "exit";
    error?: string;
    errorReason?: string;
    errorInstructions?: string[];
};

import { logStatsigEvent } from "../../services/telemetry/statsig.js";
import { trackFeatureUsage } from "../../services/onboarding/usageTracker.js";

const execFileAsync = promisify(execFile);
const GITHUB_APP_URL = "https://github.com/apps/claude";
const SECRET_NAME_PATTERN = /^[a-zA-Z0-9_]+$/;

const INITIAL_INSTALL_STATE: InstallState = {
    step: "check-gh",
    selectedRepoName: "",
    currentRepo: "",
    useCurrentRepo: false,
    apiKeyOrOAuthToken: "",
    useExistingKey: true,
    currentWorkflowInstallStep: 0,
    warnings: [],
    secretExists: false,
    secretName: "ANTHROPIC_API_KEY",
    useExistingSecret: true,
    workflowExists: false,
    selectedWorkflows: ["claude", "claude-review"],
    selectedApiKeyOption: "new",
    authType: "api_key"
};

function trackEvent(name: string, payload?: Record<string, any>) {
    logStatsigEvent(name, payload ?? {});
}

async function execGh(args: string[]) {
    try {
        const { stdout, stderr } = await execFileAsync("gh", args, { encoding: "utf8" });
        return { code: 0, stdout: stdout ?? "", stderr: stderr ?? "" };
    } catch (error: any) {
        return {
            code: typeof error?.code === "number" ? error.code : 1,
            stdout: error?.stdout?.toString?.() ?? "",
            stderr: error?.stderr?.toString?.() ?? error?.message ?? ""
        };
    }
}

export function GitHubAppInstallWizard({ onDone }: { onDone: (message: string) => void }) {
    const [existingApiKey] = useState(() => getApiKeyWithSource().key ?? null);
    const [state, setState] = useState<InstallState>(() => ({
        ...INITIAL_INSTALL_STATE,
        useExistingKey: !!existingApiKey,
        selectedApiKeyOption: existingApiKey ? "existing" : isInteractiveAuthRequired() ? "oauth" : "new"
    }));

    useCtrlExit();

    useEffect(() => {
        trackEvent("tengu_install_github_app_started", {});
    }, []);

    const checkGitHubCli = useCallback(async () => {
        const warnings: InstallWarning[] = [];

        try {
            execSync("gh --version", { stdio: "ignore" });
        } catch {
            warnings.push({
                title: "GitHub CLI not found",
                message: "GitHub CLI (gh) does not appear to be installed or accessible.",
                instructions: [
                    "Install GitHub CLI from https://cli.github.com/",
                    "macOS: brew install gh",
                    "Windows: winget install --id GitHub.cli",
                    "Linux: See installation instructions at https://github.com/cli/cli#installation"
                ]
            });
        }

        try {
            const authOutput = execSync("gh auth status -a", { encoding: "utf8" });
            const scopeLine = authOutput.match(/Token scopes:.*$/m);
            if (scopeLine) {
                const scopeText = scopeLine[0];
                const missing: string[] = [];
                if (!scopeText.includes("repo")) missing.push("repo");
                if (!scopeText.includes("workflow")) missing.push("workflow");
                if (missing.length > 0) {
                    setState((current) => ({
                        ...current,
                        step: "error",
                        error: `GitHub CLI is missing required permissions: ${missing.join(", ")}.`,
                        errorReason: "Missing required scopes",
                        errorInstructions: [
                            `Your GitHub CLI authentication is missing the "${missing.join('" and "')}" scope${missing.length > 1 ? "s" : ""} needed to manage GitHub Actions and secrets.`,
                            "",
                            "To fix this, run:",
                            "  gh auth refresh -h github.com -s repo,workflow",
                            "",
                            "This will add the necessary permissions to manage workflows and secrets."
                        ]
                    }));
                    return;
                }
            }
        } catch {
            warnings.push({
                title: "GitHub CLI not authenticated",
                message: "GitHub CLI does not appear to be authenticated.",
                instructions: [
                    "Run: gh auth login",
                    "Follow the prompts to authenticate with GitHub",
                    "Or set up authentication using environment variables or other methods"
                ]
            });
        }

        let currentRepo = "";
        try {
            execSync("git rev-parse --is-inside-work-tree", { stdio: "ignore" });
            const repoUrl = execSync("git remote get-url origin", { encoding: "utf8" })
                .trim()
                .match(new RegExp("github\\.com[:/]([^/]+/[^/]+)(\\.git)?$"));
            if (repoUrl) currentRepo = repoUrl[1]?.replace(/\\.git$/, "") || "";
        } catch { }

        trackEvent("tengu_install_github_app_step_completed", { step: "check-gh" });
        setState((current) => ({
            ...current,
            warnings,
            currentRepo,
            selectedRepoName: currentRepo,
            useCurrentRepo: !!currentRepo,
            step: warnings.length > 0 ? "warnings" : "choose-repo"
        }));
    }, []);

    useEffect(() => {
        if (state.step === "check-gh") checkGitHubCli();
    }, [state.step, checkGitHubCli]);

    const setupActions = useCallback(
        async (apiKey: string | null, secretName: string) => {
            setState((current) => ({
                ...current,
                step: "creating",
                currentWorkflowInstallStep: 0
            }));

            try {
                await setupGithubActions(
                    state.selectedRepoName,
                    apiKey,
                    secretName,
                    () => {
                        setState((current) => ({
                            ...current,
                            currentWorkflowInstallStep: current.currentWorkflowInstallStep + 1
                        }));
                    },
                    state.workflowAction === "skip",
                    state.selectedWorkflows,
                    state.authType,
                    {
                        useCurrentRepo: state.useCurrentRepo,
                        workflowExists: state.workflowExists,
                        secretExists: state.secretExists
                    }
                );
                trackEvent("tengu_install_github_app_step_completed", { step: "creating" });
                setState((current) => ({
                    ...current,
                    step: "success"
                }));
            } catch (error) {
                const message = error instanceof Error ? error.message : "Failed to set up GitHub Actions";
                if (message.includes("workflow file already exists")) {
                    trackEvent("tengu_install_github_app_error", { reason: "workflow_file_exists" });
                    setState((current) => ({
                        ...current,
                        step: "error",
                        error: "A Claude workflow file already exists in this repository.",
                        errorReason: "Workflow file conflict",
                        errorInstructions: [
                            "The file .github/workflows/claude.yml already exists",
                            "You can either:",
                            "  1. Delete the existing file and run this command again",
                            "  2. Update the existing file manually using the template from:",
                            `     ${MANUAL_SETUP_URL}`
                        ]
                    }));
                } else {
                    trackEvent("tengu_install_github_app_error", { reason: "setup_github_actions_failed" });
                    setState((current) => ({
                        ...current,
                        step: "error",
                        error: message,
                        errorReason: "GitHub Actions setup failed",
                        errorInstructions: []
                    }));
                }
            }
        },
        [
            state.authType,
            state.selectedRepoName,
            state.secretExists,
            state.selectedWorkflows,
            state.useCurrentRepo,
            state.workflowAction,
            state.workflowExists
        ]
    );

    const openGitHubApp = useCallback(async () => {
        await openInBrowser(GITHUB_APP_URL);
    }, []);

    const checkRepoAccess = useCallback(async (repo: string) => {
        try {
            const response = await execGh(["api", `repos/${repo}`, "--jq", ".permissions.admin"]);
            if (response.code === 0) {
                return { hasAccess: response.stdout.trim() === "true" };
            }
            if (response.stderr.includes("404") || response.stderr.includes("Not Found")) {
                return { hasAccess: false, error: "repository_not_found" as const };
            }
            return { hasAccess: false };
        } catch {
            return { hasAccess: false };
        }
    }, []);

    const checkWorkflowExists = useCallback(async (repo: string) => {
        const response = await execGh(["api", `repos/${repo}/contents/.github/workflows/claude.yml`, "--jq", ".sha"]);
        return response.code === 0;
    }, []);

    const checkForExistingSecret = useCallback(async () => {
        const response = await execGh(["secret", "list", "--app", "actions", "--repo", state.selectedRepoName]);
        if (response.code === 0) {
            if (response.stdout.split("\n").some((line: string) => /^ANTHROPIC_API_KEY\s+/.test(line))) {
                setState((current) => ({
                    ...current,
                    secretExists: true,
                    step: "check-existing-secret"
                }));
            } else if (existingApiKey) {
                setState((current) => ({
                    ...current,
                    apiKeyOrOAuthToken: existingApiKey,
                    useExistingKey: true
                }));
                await setupActions(existingApiKey, state.secretName);
            } else {
                setState((current) => ({
                    ...current,
                    step: "api-key"
                }));
            }
        } else if (existingApiKey) {
            setState((current) => ({
                ...current,
                apiKeyOrOAuthToken: existingApiKey,
                useExistingKey: true
            }));
            await setupActions(existingApiKey, state.secretName);
        } else {
            setState((current) => ({
                ...current,
                step: "api-key"
            }));
        }
    }, [existingApiKey, setupActions, state.secretName, state.selectedRepoName]);

    const handleContinue = useCallback(async () => {
        if (state.step === "warnings") {
            trackEvent("tengu_install_github_app_step_completed", { step: "warnings" });
            setState((current) => ({
                ...current,
                step: "install-app"
            }));
            setTimeout(() => {
                openGitHubApp();
            }, 0);
            return;
        }

        if (state.step === "choose-repo") {
            let repoName = state.useCurrentRepo ? state.currentRepo : state.selectedRepoName;
            if (!repoName.trim()) return;

            const warnings: InstallWarning[] = [];

            if (repoName.includes("github.com")) {
                const match = repoName.match(new RegExp("github\\.com[:/]([^/]+/[^/]+)(\\.git)?$"));
                if (!match) {
                    warnings.push({
                        title: "Invalid GitHub URL format",
                        message: "The repository URL format appears to be invalid.",
                        instructions: [
                            "Use format: owner/repo or https://github.com/owner/repo",
                            "Example: anthropics/claude-cli"
                        ]
                    });
                } else {
                    repoName = match[1]?.replace(/\\.git$/, "") || "";
                }
            }

            if (!repoName.includes("/")) {
                warnings.push({
                    title: "Repository format warning",
                    message: 'Repository should be in format "owner/repo"',
                    instructions: ["Use format: owner/repo", "Example: anthropics/claude-cli"]
                });
            }

            const access = await checkRepoAccess(repoName);
            if (access.error === "repository_not_found") {
                warnings.push({
                    title: "Repository not found",
                    message: `Repository ${repoName} was not found or you don't have access.`,
                    instructions: [
                        `Check that the repository name is correct: ${repoName}`,
                        "Ensure you have access to this repository",
                        'For private repositories, make sure your GitHub token has the "repo" scope',
                        "You can add the repo scope with: gh auth refresh -h github.com -s repo,workflow"
                    ]
                });
            } else if (!access.hasAccess) {
                warnings.push({
                    title: "Admin permissions required",
                    message: `You might need admin permissions on ${repoName} to set up GitHub Actions.`,
                    instructions: [
                        "Repository admins can install GitHub Apps and set secrets",
                        "Ask a repository admin to run this command if setup fails",
                        "Alternatively, you can use the manual setup instructions"
                    ]
                });
            }

            const workflowExists = await checkWorkflowExists(repoName);
            if (warnings.length > 0) {
                const mergedWarnings = [...state.warnings, ...warnings];
                setState((current) => ({
                    ...current,
                    selectedRepoName: repoName,
                    workflowExists,
                    warnings: mergedWarnings,
                    step: "warnings"
                }));
            } else {
                trackEvent("tengu_install_github_app_step_completed", { step: "choose-repo" });
                setState((current) => ({
                    ...current,
                    selectedRepoName: repoName,
                    workflowExists,
                    step: "install-app"
                }));
                setTimeout(() => {
                    openGitHubApp();
                }, 0);
            }
            return;
        }

        if (state.step === "install-app") {
            trackEvent("tengu_install_github_app_step_completed", { step: "install-app" });
            if (state.workflowExists) {
                setState((current) => ({
                    ...current,
                    step: "check-existing-workflow"
                }));
            } else {
                setState((current) => ({
                    ...current,
                    step: "select-workflows"
                }));
            }
            return;
        }

        if (state.step === "check-existing-secret") {
            trackEvent("tengu_install_github_app_step_completed", { step: "check-existing-secret" });
            if (state.useExistingSecret) await setupActions(null, state.secretName);
            else await setupActions(state.apiKeyOrOAuthToken, state.secretName);
            return;
        }

        if (state.step === "api-key") {
            if (state.selectedApiKeyOption === "oauth") return;
            const apiKey = state.selectedApiKeyOption === "existing" ? existingApiKey : state.apiKeyOrOAuthToken;
            if (!apiKey) {
                trackEvent("tengu_install_github_app_error", { reason: "api_key_missing" });
                setState((current) => ({
                    ...current,
                    step: "error",
                    error: "API key is required"
                }));
                return;
            }

            setState((current) => ({
                ...current,
                apiKeyOrOAuthToken: apiKey,
                useExistingKey: state.selectedApiKeyOption === "existing"
            }));

            const response = await execGh(["secret", "list", "--app", "actions", "--repo", state.selectedRepoName]);
            if (response.code === 0) {
                if (response.stdout.split("\n").some((line: string) => /^ANTHROPIC_API_KEY\s+/.test(line))) {
                    trackEvent("tengu_install_github_app_step_completed", { step: "api-key" });
                    setState((current) => ({
                        ...current,
                        secretExists: true,
                        step: "check-existing-secret"
                    }));
                } else {
                    trackEvent("tengu_install_github_app_step_completed", { step: "api-key" });
                    await setupActions(apiKey, state.secretName);
                }
            } else {
                trackEvent("tengu_install_github_app_step_completed", { step: "api-key" });
                await setupActions(apiKey, state.secretName);
            }
        }
    }, [
        checkRepoAccess,
        checkWorkflowExists,
        existingApiKey,
        openGitHubApp,
        setupActions,
        state.apiKeyOrOAuthToken,
        state.currentRepo,
        state.secretName,
        state.selectedApiKeyOption,
        state.selectedRepoName,
        state.step,
        state.useCurrentRepo,
        state.useExistingSecret,
        state.workflowExists,
        state.warnings
    ]);

    const handleRepoChange = useCallback((repo: string) => {
        setState((current) => ({ ...current, selectedRepoName: repo }));
    }, []);

    const handleApiKeyChange = useCallback((value: string) => {
        setState((current) => ({ ...current, apiKeyOrOAuthToken: value }));
    }, []);

    const handleApiKeyOption = useCallback((option: "existing" | "oauth" | "new") => {
        setState((current) => ({ ...current, selectedApiKeyOption: option }));
    }, []);

    const handleCreateOAuthToken = useCallback(() => {
        trackEvent("tengu_install_github_app_step_completed", { step: "api-key" });
        setState((current) => ({ ...current, step: "oauth-flow" }));
    }, []);

    const handleOAuthSuccess = useCallback(
        (token: string) => {
            trackEvent("tengu_install_github_app_step_completed", { step: "oauth-flow" });
            setState((current) => ({
                ...current,
                apiKeyOrOAuthToken: token,
                useExistingKey: false,
                secretName: "CLAUDE_CODE_OAUTH_TOKEN",
                authType: "oauth_token"
            }));
            setupActions(token, "CLAUDE_CODE_OAUTH_TOKEN");
        },
        [setupActions]
    );

    const handleOAuthCancel = useCallback(() => {
        setState((current) => ({ ...current, step: "api-key" }));
    }, []);

    const handleSecretNameChange = useCallback((value: string) => {
        if (value && !SECRET_NAME_PATTERN.test(value)) return;
        setState((current) => ({ ...current, secretName: value }));
    }, []);

    const handleToggleUseCurrentRepo = useCallback((value: boolean) => {
        setState((current) => ({
            ...current,
            useCurrentRepo: value,
            selectedRepoName: value ? current.currentRepo : ""
        }));
    }, []);

    const handleToggleUseExistingKey = useCallback((value: boolean) => {
        setState((current) => ({ ...current, useExistingKey: value }));
    }, []);

    const handleToggleUseExistingSecret = useCallback((value: boolean) => {
        setState((current) => ({
            ...current,
            useExistingSecret: value,
            secretName: value ? "ANTHROPIC_API_KEY" : ""
        }));
    }, []);

    const handleExistingWorkflowAction = useCallback(
        async (action: string) => {
            const workflowAction = action as "update" | "skip" | "exit";
            if (workflowAction === "exit") {
                onDone("Installation cancelled by user");
                return;
            }
            trackEvent("tengu_install_github_app_step_completed", { step: "check-existing-workflow" });
            setState((current) => ({ ...current, workflowAction: action as "update" | "skip" | "exit" }));
            if (action === "skip" || action === "update") {
                if (existingApiKey) await checkForExistingSecret();
                else {
                    setState((current) => ({ ...current, step: "api-key" }));
                }
            }
        },
        [checkForExistingSecret, existingApiKey, onDone]
    );

    const completedRef = useRef(false);
    useEffect(() => {
        if (completedRef.current) return;
        if (state.step === "success" || state.step === "error") {
            completedRef.current = true;
            if (state.step === "success") trackEvent("tengu_install_github_app_completed", {});
            if (state.step === "success") {
                onDone("GitHub Actions setup complete!");
            } else if (state.error) {
                onDone(`Couldn't install GitHub App: ${state.error}\nFor manual setup instructions, see: ${MANUAL_SETUP_URL}`);
            } else {
                onDone(`GitHub App installation failed\nFor manual setup instructions, see: ${MANUAL_SETUP_URL}`);
            }
        }
    }, [onDone, state.error, state.step]);

    switch (state.step) {
        case "check-gh":
            return <GitHubCliCheckView />;
        case "warnings":
            return <InstallWarningsView warnings={state.warnings} onContinue={handleContinue} />;
        case "choose-repo":
            return (
                <GitHubRepoSelection
                    currentRepo={state.currentRepo}
                    useCurrentRepo={state.useCurrentRepo}
                    repoUrl={state.selectedRepoName}
                    onRepoUrlChange={handleRepoChange}
                    onToggleUseCurrentRepo={handleToggleUseCurrentRepo}
                    onSubmit={handleContinue}
                />
            );
        case "install-app":
            return <GitHubAppInstallPrompt repoUrl={state.selectedRepoName} onSubmit={handleContinue} />;
        case "check-existing-workflow":
            return <ExistingWorkflowDialog repoName={state.selectedRepoName} onSelectAction={handleExistingWorkflowAction} />;
        case "check-existing-secret":
            return (
                <APIKeySecretChoiceView
                    useExistingSecret={state.useExistingSecret}
                    secretName={state.secretName}
                    onToggleUseExistingSecret={handleToggleUseExistingSecret}
                    onSecretNameChange={handleSecretNameChange}
                    onSubmit={handleContinue}
                />
            );
        case "api-key":
            return (
                <APIKeySelectionView
                    existingApiKey={existingApiKey ?? undefined}
                    apiKeyOrOAuthToken={state.apiKeyOrOAuthToken}
                    onApiKeyChange={handleApiKeyChange}
                    onToggleUseExistingKey={handleToggleUseExistingKey}
                    onSubmit={handleContinue}
                    onCreateOAuthToken={isInteractiveAuthRequired() ? handleCreateOAuthToken : undefined}
                    selectedOption={state.selectedApiKeyOption}
                    onSelectOption={handleApiKeyOption}
                />
            );
        case "creating":
            return (
                <InstallationProgressView
                    currentWorkflowInstallStep={state.currentWorkflowInstallStep}
                    secretExists={state.secretExists}
                    useExistingSecret={state.useExistingSecret}
                    secretName={state.secretName}
                    skipWorkflow={state.workflowAction === "skip"}
                    selectedWorkflows={state.selectedWorkflows}
                />
            );
        case "success":
            return (
                <InstallationSuccessView
                    secretExists={state.secretExists}
                    useExistingSecret={state.useExistingSecret}
                    secretName={state.secretName}
                    skipWorkflow={state.workflowAction === "skip"}
                />
            );
        case "error":
            return (
                <InstallationErrorView
                    error={state.error || "Unknown error"}
                    errorReason={state.errorReason}
                    errorInstructions={state.errorInstructions}
                />
            );
        case "select-workflows":
            return (
                <WorkflowSelectionView
                    defaultSelections={state.selectedWorkflows}
                    onSubmit={(selections: string[]) => {
                        trackEvent("tengu_install_github_app_step_completed", { step: "select-workflows" });
                        setState((current) => ({ ...current, selectedWorkflows: selections }));
                        if (existingApiKey) checkForExistingSecret();
                        else setState((current) => ({ ...current, step: "api-key" }));
                    }}
                />
            );
        case "oauth-flow":
            return <GitHubOAuthWizard onSuccess={handleOAuthSuccess} onCancel={handleOAuthCancel} />;
        default:
            return null;
    }
}
