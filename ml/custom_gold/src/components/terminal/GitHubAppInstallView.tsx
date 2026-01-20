// Logic from chunk_546.ts (GitHub App Installation UI)

import React, { useState } from "react";
import { Box, Text, useInput, useStdout } from "ink";
import InkTextInput from "ink-text-input";
import { PermissionSelect } from "../permissions/PermissionComponents.js";
import { useTheme } from "../../services/terminal/themeManager.js";
import { figures } from "../../vendor/terminalFigures.js";

const TextInput = InkTextInput as unknown as React.FC<{
    value: string;
    onChange: (value: string) => void;
    onSubmit?: () => void;
    onPaste?: (value: string) => void;
    placeholder?: string;
    focus?: boolean;
    showCursor?: boolean;
    mask?: string;
    columns?: number;
    cursorOffset?: number;
    onChangeCursorOffset?: (offset: number) => void;
}>;

const WORKFLOW_PROMPT_INSTALL = "Add Claude Code GitHub Workflow";
export const MANUAL_SETUP_URL = "https://github.com/anthropics/claude-code-action/blob/main/docs/setup.md";
const GITHUB_APP_URL = "https://github.com/apps/claude";
const WORKFLOW_TEMPLATE_URL = "https://github.com/anthropics/claude-code-action/blob/main/examples/claude.yml";

const DEFAULT_WORKFLOW_YAML = `name: Claude Code

on:
  issue_comment:
    types: [created]
  pull_request_review_comment:
    types: [created]
  issues:
    types: [opened, assigned]
  pull_request_review:
    types: [submitted]

jobs:
  claude:
    if: |
      (github.event_name == 'issue_comment' && contains(github.event.comment.body, '@claude')) ||
      (github.event_name == 'pull_request_review_comment' && contains(github.event.comment.body, '@claude')) ||
      (github.event_name == 'pull_request_review' && contains(github.event.review.body, '@claude')) ||
      (github.event_name == 'issues' && (contains(github.event.issue.body, '@claude') || contains(github.event.issue.title, '@claude')))
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: read
      issues: read
      id-token: write
      actions: read # Required for Claude to read CI results on PRs
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 1

      - name: Run Claude Code
        id: claude
        uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: \${{ secrets.ANTHROPIC_API_KEY }}

          # This is an optional setting that allows Claude to read CI results on PRs
          additional_permissions: |
            actions: read

          # Optional: Give a custom prompt to Claude. If this is not specified, Claude will perform the instructions specified in the comment that tagged it.
          # prompt: 'Update the pull request description to include a summary of changes.'

          # Optional: Add claude_args to customize behavior and configuration
          # See https://github.com/anthropics/claude-code-action/blob/main/docs/usage.md
          # or https://code.claude.com/docs/en/cli-reference for available options
          # claude_args: '--allowed-tools Bash(gh pr:*)'

`;

const DEFAULT_WORKFLOW_PR_BODY = `## 🤖 Installing Claude Code GitHub App

This PR adds a GitHub Actions workflow that enables Claude Code integration in our repository.

### What is Claude Code?

[Claude Code](https://claude.com/claude-code) is an AI coding agent that can help with:
- Bug fixes and improvements  
- Documentation updates
- Implementing new features
- Code reviews and suggestions
- Writing tests
- And more!

### How it works

Once this PR is merged, we'll be able to interact with Claude by mentioning @claude in a pull request or issue comment.
Once the workflow is triggered, Claude will analyze the comment and surrounding context, and execute on the request in a GitHub action.

### Important Notes

- **This workflow won't take effect until this PR is merged**
- **@claude mentions won't work until after the merge is complete**
- The workflow runs automatically whenever Claude is mentioned in PR or issue comments
- Claude gets access to the entire PR or issue context including files, diffs, and previous comments

### Security

- Our Anthropic API key is securely stored as a GitHub Actions secret
- Only users with write access to the repository can trigger the workflow
- All Claude runs are stored in the GitHub Actions run history
- Claude's default tools are limited to reading/writing files and interacting with our repo by creating comments, branches, and commits.
- We can add more allowed tools by adding them to the workflow file like:

\`\`\`
allowed_tools: Bash(npm install),Bash(npm run build),Bash(npm run lint),Bash(npm run test)
\`\`\`

There's more information in the [Claude Code action repo](https://github.com/anthropics/claude-code-action).

After merging this PR, let's try mentioning @claude in a comment on any PR to get started!`;

const DEFAULT_REVIEW_WORKFLOW = `name: Claude Code Review

on:
  pull_request:
    types: [opened, synchronize]
    # Optional: Only run on specific file changes
    # paths:
    #   - "src/**/*.ts"
    #   - "src/**/*.tsx"
    #   - "src/**/*.js"
    #   - "src/**/*.jsx"

jobs:
  claude-review:
    # Optional: Filter by PR author
    # if: |
    #   github.event.pull_request.user.login == 'external-contributor' ||
    #   github.event.pull_request.user.login == 'new-developer' ||
    #   github.event.pull_request.author_association == 'FIRST_TIME_CONTRIBUTOR'

    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: read
      issues: read
      id-token: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 1

      - name: Run Claude Code Review
        id: claude-review
        uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: \${{ secrets.ANTHROPIC_API_KEY }}
          prompt: |
            REPO: \${{ github.repository }}
            PR NUMBER: \${{ github.event.pull_request.number }}

            Please review this pull request and provide feedback on:
            - Code quality and best practices
            - Potential bugs or issues
            - Performance considerations
            - Security concerns
            - Test coverage

            Use the repository's CLAUDE.md for guidance on style and conventions. Be constructive and helpful in your feedback.

            Use \`gh pr comment\` with your Bash tool to leave your review as a comment on the PR.

          # See https://github.com/anthropics/claude-code-action/blob/main/docs/usage.md
          # or https://code.claude.com/docs/en/cli-reference for available options
          claude_args: '--allowed-tools "Bash(gh issue view:*),Bash(gh search:*),Bash(gh issue list:*),Bash(gh pr comment:*),Bash(gh pr diff:*),Bash(gh pr view:*),Bash(gh pr list:*)"'

`;

const PLUGIN_REVIEW_WORKFLOW = `name: Claude Code Review

on:
  pull_request:
    types: [opened, synchronize, ready_for_review, reopened]
    # Optional: Only run on specific file changes
    # paths:
    #   - "src/**/*.ts"
    #   - "src/**/*.tsx"
    #   - "src/**/*.js"
    #   - "src/**/*.jsx"

jobs:
  claude-review:
    # Optional: Filter by PR author
    # if: |
    #   github.event.pull_request.user.login == 'external-contributor' ||
    #   github.event.pull_request.user.login == 'new-developer' ||
    #   github.event.pull_request.author_association == 'FIRST_TIME_CONTRIBUTOR'

    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: read
      issues: read
      id-token: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 1

      - name: Run Claude Code Review
        id: claude-review
        uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: \${{ secrets.ANTHROPIC_API_KEY }}
          plugin_marketplaces: 'https://github.com/anthropics/claude-code.git'
          plugins: 'code-review@claude-code-plugins'
          prompt: '/code-review:code-review \${{ github.repository }}/pull/\${{ github.event.pull_request.number }}'
          # See https://github.com/anthropics/claude-code-action/blob/main/docs/usage.md
          # or https://code.claude.com/docs/en/cli-reference for available options

`;

// --- GitHub App install prompt (b59) ---
export function GitHubAppInstallPrompt({ repoUrl, onSubmit }: { repoUrl: string; onSubmit: () => void }) {
    useInput((_input, key) => {
        if (key.return) onSubmit();
    });

    return (
        <Box flexDirection="column" borderStyle="round" borderDimColor paddingX={1}>
            <Box flexDirection="column" marginBottom={1}>
                <Text bold>Install the Claude GitHub App</Text>
            </Box>
            <Box marginBottom={1}>
                <Text>Opening browser to install the Claude GitHub App…</Text>
            </Box>
            <Box marginBottom={1}>
                <Text>If your browser doesn't open automatically, visit:</Text>
            </Box>
            <Box marginBottom={1}>
                <Text underline>{GITHUB_APP_URL}</Text>
            </Box>
            <Box marginBottom={1}>
                <Text>
                    Please install the app for repository: <Text bold>{repoUrl}</Text>
                </Text>
            </Box>
            <Box marginBottom={1}>
                <Text dimColor>Important: Make sure to grant access to this specific repository</Text>
            </Box>
            <Box>
                <Text bold color="permission">
                    Press Enter once you've installed the app{figures.ellipsis}
                </Text>
            </Box>
            <Box marginTop={1}>
                <Text dimColor>
                    Having trouble? See manual setup instructions at: <Text color="claude">{MANUAL_SETUP_URL}</Text>
                </Text>
            </Box>
        </Box>
    );
}

// --- API Key Secret Warning View (h59) ---
export function APIKeySecretChoiceView({
    useExistingSecret,
    secretName,
    onToggleUseExistingSecret,
    onSecretNameChange,
    onSubmit
}: {
    useExistingSecret: boolean;
    secretName: string;
    onToggleUseExistingSecret: (value: boolean) => void;
    onSecretNameChange: (value: string) => void;
    onSubmit: () => void;
}) {
    const [cursorOffset, setCursorOffset] = useState(0);
    const { stdout } = useStdout();
    const columns = stdout?.columns ?? 80;
    const [theme] = useTheme();

    useInput((_input, key) => {
        if (key.upArrow) onToggleUseExistingSecret(true);
        else if (key.downArrow) onToggleUseExistingSecret(false);
        else if (key.return) onSubmit();
    });

    return (
        <>
            <Box flexDirection="column" borderStyle="round" borderDimColor paddingX={1}>
                <Box flexDirection="column" marginBottom={1}>
                    <Text bold>Install GitHub App</Text>
                    <Text dimColor>Setup API key secret</Text>
                </Box>
                <Box marginBottom={1}>
                    <Text color="warning">ANTHROPIC_API_KEY already exists in repository secrets!</Text>
                </Box>
                <Box marginBottom={1}>
                    <Text>Would you like to:</Text>
                </Box>
                <Box marginBottom={1}>
                    <Text>{useExistingSecret ? <Text color={theme?.success}>{figures.pointer} </Text> : "  "}Use the existing API key</Text>
                </Box>
                <Box marginBottom={1}>
                    <Text>{!useExistingSecret ? <Text color={theme?.success}>{figures.pointer} </Text> : "  "}Create a new secret with a different name</Text>
                </Box>
                {!useExistingSecret && (
                    <>
                        <Box marginBottom={1}>
                            <Text>Enter new secret name (alphanumeric with underscores):</Text>
                        </Box>
                        <TextInput
                            value={secretName}
                            onChange={onSecretNameChange}
                            onSubmit={onSubmit}
                            focus={true}
                            placeholder="e.g., CLAUDE_API_KEY"
                            columns={columns}
                            cursorOffset={cursorOffset}
                            onChangeCursorOffset={setCursorOffset}
                            showCursor={true}
                        />
                    </>
                )}
                <Box marginLeft={3}>
                    <Text dimColor>↑/↓ to select · Enter to continue</Text>
                </Box>
            </Box>
        </>
    );
}

// --- API Key Choice View (m59) ---
export function APIKeySelectionView({
    existingApiKey,
    apiKeyOrOAuthToken,
    onApiKeyChange,
    onSubmit,
    onToggleUseExistingKey,
    onCreateOAuthToken,
    selectedOption = existingApiKey ? "existing" : onCreateOAuthToken ? "oauth" : "new",
    onSelectOption
}: {
    existingApiKey?: string;
    apiKeyOrOAuthToken: string;
    onApiKeyChange: (value: string) => void;
    onSubmit: () => void;
    onToggleUseExistingKey: (value: boolean) => void;
    onCreateOAuthToken?: () => void;
    selectedOption?: "existing" | "oauth" | "new";
    onSelectOption?: (value: "existing" | "oauth" | "new") => void;
}) {
    const [cursorOffset, setCursorOffset] = useState(0);
    const { stdout } = useStdout();
    const columns = stdout?.columns ?? 80;
    const [theme] = useTheme();

    useInput((_input, key) => {
        if (key.upArrow) {
            if (selectedOption === "new" && onCreateOAuthToken) onSelectOption?.("oauth");
            else if (selectedOption === "oauth" && existingApiKey) {
                onSelectOption?.("existing");
                onToggleUseExistingKey(true);
            }
        } else if (key.downArrow) {
            if (selectedOption === "existing") {
                onSelectOption?.(onCreateOAuthToken ? "oauth" : "new");
                onToggleUseExistingKey(false);
            } else if (selectedOption === "oauth") {
                onSelectOption?.("new");
            }
        }
        if (key.return) {
            if (selectedOption === "oauth" && onCreateOAuthToken) onCreateOAuthToken();
            else onSubmit();
        }
    });

    return (
        <>
            <Box flexDirection="column" borderStyle="round" borderDimColor paddingX={1}>
                <Box flexDirection="column" marginBottom={1}>
                    <Text bold>Install GitHub App</Text>
                    <Text dimColor>Choose API key</Text>
                </Box>
                {existingApiKey && (
                    <Box marginBottom={1}>
                        <Text>{selectedOption === "existing" ? <Text color={theme?.success}>{figures.pointer} </Text> : "  "}Use your existing Claude Code API key</Text>
                    </Box>
                )}
                {onCreateOAuthToken && (
                    <Box marginBottom={1}>
                        <Text>{selectedOption === "oauth" ? <Text color={theme?.success}>{figures.pointer} </Text> : "  "}Create a long-lived token with your Claude subscription</Text>
                    </Box>
                )}
                <Box marginBottom={1}>
                    <Text>{selectedOption === "new" ? <Text color={theme?.success}>{figures.pointer} </Text> : "  "}Enter a new API key</Text>
                </Box>
                {selectedOption === "new" && (
                    <TextInput
                        value={apiKeyOrOAuthToken}
                        onChange={onApiKeyChange}
                        onSubmit={onSubmit}
                        onPaste={onApiKeyChange}
                        focus={true}
                        placeholder="sk-ant… (Create a new key at https://console.anthropic.com/settings/keys)"
                        mask="*"
                        columns={columns}
                        cursorOffset={cursorOffset}
                        onChangeCursorOffset={setCursorOffset}
                        showCursor={true}
                    />
                )}
                <Box marginLeft={3}>
                    <Text dimColor>↑/↓ to select · Enter to continue</Text>
                </Box>
            </Box>
        </>
    );
}

// --- Installation Progress View (p59) ---
export function InstallationProgressView({
    currentWorkflowInstallStep,
    secretExists,
    useExistingSecret,
    secretName,
    skipWorkflow = false,
    selectedWorkflows
}: {
    currentWorkflowInstallStep: number;
    secretExists: boolean;
    useExistingSecret: boolean;
    secretName: string;
    skipWorkflow?: boolean;
    selectedWorkflows: string[];
}) {
    const steps = skipWorkflow
        ? [
            "Getting repository information",
            secretExists && useExistingSecret ? "Using existing API key secret" : `Setting up ${secretName} secret`
        ]
        : [
            "Getting repository information",
            "Creating branch",
            selectedWorkflows.length > 1 ? "Creating workflow files" : "Creating workflow file",
            secretExists && useExistingSecret ? "Using existing API key secret" : `Setting up ${secretName} secret`,
            "Opening pull request page"
        ];

    return (
        <>
            <Box flexDirection="column" borderStyle="round" borderDimColor paddingX={1}>
                <Box flexDirection="column" marginBottom={1}>
                    <Text bold>Install GitHub App</Text>
                    <Text dimColor>Create GitHub Actions workflow</Text>
                </Box>
                {steps.map((label, index) => {
                    let status: "pending" | "completed" | "in-progress" = "pending";
                    if (index < currentWorkflowInstallStep) status = "completed";
                    else if (index === currentWorkflowInstallStep) status = "in-progress";

                    return (
                        <Box key={index}>
                            <Text color={status === "completed" ? "success" : status === "in-progress" ? "warning" : undefined}>
                                {status === "completed" ? "✓ " : ""}{label}{status === "in-progress" ? "…" : ""}
                            </Text>
                        </Box>
                    );
                })}
            </Box>
        </>
    );
}

// --- Installation Success View (l59) ---
export function InstallationSuccessView({
    secretExists,
    useExistingSecret,
    secretName,
    skipWorkflow = false
}: {
    secretExists: boolean;
    useExistingSecret: boolean;
    secretName: string;
    skipWorkflow?: boolean;
}) {
    return (
        <>
            <Box flexDirection="column" borderStyle="round" borderDimColor paddingX={1}>
                <Box flexDirection="column" marginBottom={1}>
                    <Text bold>Install GitHub App</Text>
                    <Text dimColor>Success</Text>
                </Box>
                {!skipWorkflow && <Text color="success">✓ GitHub Actions workflow created!</Text>}
                {secretExists && useExistingSecret && (
                    <Box marginTop={1}>
                        <Text color="success">✓ Using existing ANTHROPIC_API_KEY secret</Text>
                    </Box>
                )}
                {(!secretExists || !useExistingSecret) && (
                    <Box marginTop={1}>
                        <Text color="success">✓ API key saved as {secretName} secret</Text>
                    </Box>
                )}
                <Box marginTop={1}>
                    <Text>Next steps:</Text>
                </Box>
                {skipWorkflow ? (
                    <>
                        <Text>1. Install the Claude GitHub App if you haven't already</Text>
                        <Text>2. Your workflow file was kept unchanged</Text>
                        <Text>3. API key is configured and ready to use</Text>
                    </>
                ) : (
                    <>
                        <Text>1. A pre-filled PR page has been created</Text>
                        <Text>2. Install the Claude GitHub App if you haven't already</Text>
                        <Text>3. Merge the PR to enable Claude PR assistance</Text>
                    </>
                )}
                <Box marginLeft={3}>
                    <Text dimColor>Press any key to exit</Text>
                </Box>
            </Box>
        </>
    );
}

// --- Installation Error View (n59) ---
export function InstallationErrorView({
    error,
    errorReason,
    errorInstructions
}: {
    error: string;
    errorReason?: string;
    errorInstructions?: string[];
}) {
    return (
        <>
            <Box flexDirection="column" borderStyle="round" borderDimColor paddingX={1}>
                <Box flexDirection="column" marginBottom={1}>
                    <Text bold>Install GitHub App</Text>
                </Box>
                <Text color="error">Error: {error}</Text>
                {errorReason && (
                    <Box marginTop={1}>
                        <Text dimColor>Reason: {errorReason}</Text>
                    </Box>
                )}
                {errorInstructions && errorInstructions.length > 0 && (
                    <Box flexDirection="column" marginTop={1}>
                        <Text dimColor>How to fix:</Text>
                        {errorInstructions.map((instruction, index) => (
                            <Box key={index} marginLeft={2}>
                                <Text dimColor>• </Text>
                                <Text>{instruction}</Text>
                            </Box>
                        ))}
                    </Box>
                )}
                <Box marginTop={1}>
                    <Text dimColor>
                        For manual setup instructions, see: <Text color="claude">{MANUAL_SETUP_URL}</Text>
                    </Text>
                </Box>
            </Box>
            <Box marginLeft={3}>
                <Text dimColor>Press any key to exit</Text>
            </Box>
        </>
    );
}

// --- Existing Workflow Dialog (o59) ---
export function ExistingWorkflowDialog({
    repoName,
    onSelectAction
}: {
    repoName: string;
    onSelectAction: (value: string) => void;
}) {
    return (
        <Box flexDirection="column" borderStyle="round" borderDimColor paddingX={1}>
            <Box flexDirection="column" marginBottom={1}>
                <Text bold>Existing Workflow Found</Text>
                <Text dimColor>Repository: {repoName}</Text>
            </Box>
            <Box flexDirection="column" marginBottom={1}>
                <Text>
                    A Claude workflow file already exists at <Text color="claude">.github/workflows/claude.yml</Text>
                </Text>
                <Text dimColor>What would you like to do?</Text>
            </Box>
            <PermissionSelect
                options={[
                    { label: "Update workflow file with latest version", value: "update" },
                    { label: "Skip workflow update (configure secrets only)", value: "skip" },
                    { label: "Exit without making changes", value: "exit" }
                ]}
                onChange={onSelectAction}
                onCancel={() => onSelectAction("exit")}
            />
            <Box marginTop={1}>
                <Text dimColor>
                    View the latest workflow template at:{" "}
                    <Text color="claude">{WORKFLOW_TEMPLATE_URL}</Text>
                </Text>
            </Box>
        </Box>
    );
}

const _unused = {
    WORKFLOW_PROMPT_INSTALL,
    DEFAULT_WORKFLOW_YAML,
    DEFAULT_WORKFLOW_PR_BODY,
    DEFAULT_REVIEW_WORKFLOW,
    PLUGIN_REVIEW_WORKFLOW
};
void _unused;
