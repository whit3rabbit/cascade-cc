
import { execFile } from "child_process";
import { promisify } from "util";
import { logStatsigEvent } from "../telemetry/statsig.js";
import { updateSettings } from "./settings.js";
import { openInBrowser } from "../../utils/browser/BrowserUtils.js";

const execFileAsync = promisify(execFile);

const CLAUDE_PR_ASSISTANT_WORKFLOW = `name: Claude PR Assistant

on:
  issue_comment:
    types: [created]

permissions:
  contents: write
  issues: write
  pull-requests: write

jobs:
  claude:
    if: endsWith(github.event.comment.body, '@claude')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: anthropics/claude-code-action@main
        with:
          anthropic_api_key: \${{ secrets.ANTHROPIC_API_KEY }}
`;

const CLAUDE_CODE_REVIEW_WORKFLOW = `name: Claude Code Review

on:
  pull_request:
    types: [opened, synchronize]

permissions:
  contents: read
  pull-requests: write

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: \${{ secrets.ANTHROPIC_API_KEY }}
`;

const CLAUDE_CODE_REVIEW_PLUGIN_WORKFLOW = `name: Claude Code Review

on:
  pull_request:
    types: [opened, synchronize]

permissions:
  contents: read
  pull-requests: write

jobs:
  review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: \${{ secrets.ANTHROPIC_API_KEY }}
          plugin: code-review
`;

const MANUAL_SETUP_URL = "https://github.com/anthropics/claude-code-action";
const PR_TITLE = "Add Claude GitHub Actions";
const PR_BODY = "This PR adds Claude GitHub Actions to your repository. This will allow you to tag @claude in issue comments and PRs to get help from Claude, and automatically run code reviews on new PRs.";

/**
 * Capture output and status from a command.
 */
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

/**
 * Creates or updates a workflow file in the repository.
 * Deobfuscated from r47 in chunk_547.ts.
 */
export async function createWorkflowFile(
    repo: string,
    branch: string,
    path: string,
    content: string,
    secretName: string,
    message: string,
    telemetryContext: any
) {
    let sha: string | null = null;
    const shaResponse = await execGh(["api", `repos/${repo}/contents/${path}`, "--jq", ".sha"]);
    if (shaResponse.code === 0) {
        sha = shaResponse.stdout.trim();
    }

    let finalContent = content;
    if (secretName === "CLAUDE_CODE_OAUTH_TOKEN") {
        finalContent = content.replace(
            /anthropic_api_key: \$\{\{ secrets\.ANTHROPIC_API_KEY \}\}/g,
            "claude_code_oauth_token: ${{ secrets.CLAUDE_CODE_OAUTH_TOKEN }}"
        );
    } else if (secretName !== "ANTHROPIC_API_KEY") {
        finalContent = content.replace(
            /anthropic_api_key: \$\{\{ secrets\.ANTHROPIC_API_KEY \}\}/g,
            `anthropic_api_key: \${{ secrets.${secretName} }}`
        );
    }

    const contentBase64 = Buffer.from(finalContent).toString("base64");
    const args = [
        "api",
        "--method",
        "PUT",
        `repos/${repo}/contents/${path}`,
        "-f",
        `message=${sha ? `"Update ${message}"` : `"${message}"`}`,
        "-f",
        `content=${contentBase64}`,
        "-f",
        `branch=${branch}`
    ];
    if (sha) {
        args.push("-f", `sha=${sha}`);
    }

    const response = await execGh(args);
    if (response.code !== 0) {
        if (response.stderr.includes("422") && response.stderr.includes("sha")) {
            logStatsigEvent("tengu_setup_github_actions_failed", {
                reason: "failed_to_create_workflow_file",
                exit_code: response.code,
                ...telemetryContext
            });
            throw new Error(
                `Failed to create workflow file ${path}: A Claude workflow file already exists in this repository. Please remove it first or update it manually.`
            );
        }

        logStatsigEvent("tengu_setup_github_actions_failed", {
            reason: "failed_to_create_workflow_file",
            exit_code: response.code,
            ...telemetryContext
        });

        const helpMsg = `

Need help? Common issues:
• Permission denied → Run: gh auth refresh -h github.com -s repo,workflow
• Not authorized → Ensure you have admin access to the repository
• For manual setup → Visit: ${MANUAL_SETUP_URL}`;

        throw new Error(`Failed to create workflow file ${path}: ${response.stderr}${helpMsg}`);
    }
}

/**
 * Orchestrates the full setup of GitHub Actions for a repository.
 * Deobfuscated from A79 in chunk_547.ts.
 */
export async function setupGithubActions(
    repo: string,
    apiKey: string | null,
    secretName: string,
    onStep: () => void,
    skipWorkflow: boolean = false,
    selectedWorkflows: string[],
    authType: string,
    telemetryContext: any
) {
    try {
        logStatsigEvent("tengu_setup_github_actions_started", {
            skip_workflow: skipWorkflow,
            has_api_key: !!apiKey,
            using_default_secret_name: secretName === "ANTHROPIC_API_KEY",
            selected_claude_workflow: selectedWorkflows.includes("claude"),
            selected_claude_review_workflow: selectedWorkflows.includes("claude-review"),
            ...telemetryContext
        });

        // 1. Check repo access and get ID
        const repoIdResponse = await execGh(["api", `repos/${repo}`, "--jq", ".id"]);
        if (repoIdResponse.code !== 0) {
            logStatsigEvent("tengu_setup_github_actions_failed", {
                reason: "repo_not_found",
                exit_code: repoIdResponse.code,
                ...telemetryContext
            });
            throw new Error(`Failed to access repository ${repo}`);
        }

        // 2. Get default branch
        const branchResponse = await execGh(["api", `repos/${repo}`, "--jq", ".default_branch"]);
        if (branchResponse.code !== 0) {
            logStatsigEvent("tengu_setup_github_actions_failed", {
                reason: "failed_to_get_default_branch",
                exit_code: branchResponse.code,
                ...telemetryContext
            });
            throw new Error(`Failed to get default branch: ${branchResponse.stderr}`);
        }

        const defaultBranch = branchResponse.stdout.trim();

        // 3. Get branch SHA
        const shaResponse = await execGh(["api", `repos/${repo}/git/ref/heads/${defaultBranch}`, "--jq", ".object.sha"]);
        if (shaResponse.code !== 0) {
            logStatsigEvent("tengu_setup_github_actions_failed", {
                reason: "failed_to_get_branch_sha",
                exit_code: shaResponse.code,
                ...telemetryContext
            });
            throw new Error(`Failed to get branch SHA: ${shaResponse.stderr}`);
        }

        const branchSha = shaResponse.stdout.trim();
        let newBranchName: string | null = null;

        if (!skipWorkflow) {
            onStep(); // Step 1: Branch creation
            newBranchName = `add-claude-github-actions-${Date.now()}`;
            const createBranchResponse = await execGh([
                "api",
                "--method",
                "POST",
                `repos/${repo}/git/refs`,
                "-f",
                `ref=refs/heads/${newBranchName}`,
                "-f",
                `sha=${branchSha}`
            ]);

            if (createBranchResponse.code !== 0) {
                logStatsigEvent("tengu_setup_github_actions_failed", {
                    reason: "failed_to_create_branch",
                    exit_code: createBranchResponse.code,
                    ...telemetryContext
                });
                throw new Error(`Failed to create branch: ${createBranchResponse.stderr}`);
            }

            onStep(); // Step 2: Workflow creation

            const workflowConfigs: { path: string; content: string; message: string }[] = [];
            if (selectedWorkflows.includes("claude")) {
                workflowConfigs.push({
                    path: ".github/workflows/claude.yml",
                    content: CLAUDE_PR_ASSISTANT_WORKFLOW,
                    message: "Claude PR Assistant workflow"
                });
            }

            if (selectedWorkflows.includes("claude-review")) {
                // In original code there's a check for gZ("tengu_gha_plugin_code_review")
                // which likely means they are testing a new version of the review action.
                // We'll use the plugin-based one as the default for now if it exists,
                // but for 1-1 we should ideally mock that check.
                const usePluginVersion = true; // Mocking the experiment/flag
                workflowConfigs.push({
                    path: ".github/workflows/claude-code-review.yml",
                    content: usePluginVersion ? CLAUDE_CODE_REVIEW_PLUGIN_WORKFLOW : CLAUDE_CODE_REVIEW_WORKFLOW,
                    message: "Claude Code Review workflow"
                });
            }

            for (const config of workflowConfigs) {
                await createWorkflowFile(repo, newBranchName, config.path, config.content, secretName, config.message, telemetryContext);
            }
        }

        onStep(); // Step 3: Secret setting (or finalization)

        if (apiKey) {
            const secretResponse = await execGh(["secret", "set", secretName, "--body", apiKey, "--repo", repo]);
            if (secretResponse.code !== 0) {
                logStatsigEvent("tengu_setup_github_actions_failed", {
                    reason: "failed_to_set_api_key_secret",
                    exit_code: secretResponse.code,
                    ...telemetryContext
                });
                const helpMsg = `

Need help? Common issues:
• Permission denied → Run: gh auth refresh -h github.com -s repo
• Not authorized → Ensure you have admin access to the repository
• For manual setup → Visit: ${MANUAL_SETUP_URL}`;
                throw new Error(`Failed to set API key secret: ${secretResponse.stderr || "Unknown error"}${helpMsg}`);
            }
        }

        if (!skipWorkflow && newBranchName) {
            onStep(); // Final Step: PR Opening
            const compareUrl = `https://github.com/${repo}/compare/${defaultBranch}...${newBranchName}?quick_pull=1&title=${encodeURIComponent(
                PR_TITLE
            )}&body=${encodeURIComponent(PR_BODY)}`;
            await openInBrowser(compareUrl);
        }

        logStatsigEvent("tengu_setup_github_actions_completed", {
            skip_workflow: skipWorkflow,
            has_api_key: !!apiKey,
            auth_type: authType,
            using_default_secret_name: secretName === "ANTHROPIC_API_KEY",
            selected_claude_workflow: selectedWorkflows.includes("claude"),
            selected_claude_review_workflow: selectedWorkflows.includes("claude-review"),
            ...telemetryContext
        });

        // Track setup count in settings
        updateSettings("userSettings", (prev: any) => ({
            ...prev,
            githubActionSetupCount: (prev.githubActionSetupCount ?? 0) + 1
        }));
    } catch (error) {
        if (!(error instanceof Error) || !error.message.includes("Failed to")) {
            logStatsigEvent("tengu_setup_github_actions_failed", {
                reason: "unexpected_error",
                ...telemetryContext
            });
        }
        // In original code there's a t(I) call which is likely an error logger
        throw error;
    }
}
