
// Logic deobfuscated from chunk_443.ts (Teleport/Remote Tasks) and related chunks.

import React from 'react';
import { Text, Box } from 'ink';
import chalk from 'chalk';
import axios from 'axios';
import { randomUUID } from 'node:crypto';
import { execShellCommand } from '../../utils/shared/shellUtils.js';
import { log, logError } from '../logger/loggerService.js';
import { logTelemetryEvent } from '../telemetry/telemetryInit.js';
import { getStoredOauthTokens } from '../auth/oauthManager.js';
import { getActiveModel } from '../claude/claudeUtils.js';
import { callClaude } from '../claude/claudeApi.js';
import { getSettings } from '../terminal/settings.js';
import { getGitRepoName, parseGitRepoName, isGitClean } from '../../utils/shared/gitUtils.js';
import { initTaskOutput, appendTaskOutput, getTaskOutputPath } from '../persistence/persistenceUtils.js';
import { getAppState, updateAppState } from '../../contexts/AppStateContext.js';

// --- Custom Error Class (DV in chunk_443.ts) ---

export class DisplayError extends Error {
    constructor(message: string, public displayMessage: string) {
        super(message);
        this.name = "DisplayError";
    }
}

// --- Constants ---

const REMOTE_DESC_TITLE_BRANCH_PROMPT = `You are coming up with a succinct title and git branch name for a coding session based on the provided description. The title should be clear, concise, and accurately reflect the content of the coding task.
You should keep it short and simple, ideally no more than 6 words. Avoid using jargon or overly technical terms unless absolutely necessary. The title should be easy to understand for anyone reading it.
You should wrap the title in <title> tags.

The branch name should be clear, concise, and accurately reflect the content of the coding task.
You should keep it short and simple, ideally no more than 4 words. The branch should always start with "claude/" and should be all lower case, with words separated by dashes.
You should wrap the branch name in <branch> tags.

The title should always come first, followed by the branch. Do not include any other text other than the title and branch.

Example 1:
<title>Fix login button not working on mobile</title>
<branch>claude/fix-mobile-login-button</branch>

Example 2:
<title>Update README with installation instructions</title>
<branch>claude/update-readme</branch>

Example 3:
<title>Improve performance of data processing script</title>
<branch>claude/improve-data-processing</branch>

Here is the session description:
<description>{description}</description>
Please generate a title and branch name for this session.`;

// --- Internal Helper Functions ---

function getAuthHeaders(token: string) {
    return {
        "Authorization": `Bearer ${token}`,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    };
}

async function getOrganizationUuid(): Promise<string | null> {
    const settings = getSettings("userSettings");
    return settings.forceLoginOrgUUID || process.env.CLAUDE_CODE_ORGANIZATION_UUID || null;
}

const BASE_API_URL = process.env.CLAUDE_BASE_API_URL || "https://api.anthropic.com";

const logger = log("teleport");

// --- Git Utilities (Yx5, Jx5, Xx5, CJ1, T0A) ---

export async function gitCheckoutBranch(branch: string) {
    const { code, stderr } = await execShellCommand("git", ["checkout", branch]);
    if (code !== 0) {
        logger(`Local checkout failed, trying to checkout from origin: ${stderr}`);
        let result = await execShellCommand("git", ["checkout", "-b", branch, "--track", `origin/${branch}`]);
        if (result.code !== 0) {
            logger(`Remote checkout with -b failed, trying without -b: ${result.stderr}`);
            result = await execShellCommand("git", ["checkout", "--track", `origin/${branch}`]);
        }
        if (result.code !== 0) {
            logTelemetryEvent("tengu_teleport_error_branch_checkout_failed", {});
            throw new DisplayError(`Failed to checkout branch '${branch}': ${result.stderr}`, chalk.red(`Failed to checkout branch '${chalk.bold(branch)}'\n`));
        }
    }
    await gitBranchSetUpstream(branch);
}

async function gitBranchSetUpstream(branch: string) {
    const { code: upstreamCheckCode } = await execShellCommand("git", ["rev-parse", "--abbrev-ref", `${branch}@{upstream}`]);
    if (upstreamCheckCode === 0) {
        logger(`Branch '${branch}' already has upstream set`);
        return;
    }

    const { code: originCheckCode } = await execShellCommand("git", ["rev-parse", "--verify", `origin/${branch}`]);
    if (originCheckCode === 0) {
        logger(`Setting upstream for '${branch}' to 'origin/${branch}'`);
        const { code, stderr } = await execShellCommand("git", ["branch", "--set-upstream-to", `origin/${branch}`, branch]);
        if (code !== 0) {
            logger(`Failed to set upstream for '${branch}': ${stderr}`);
        } else {
            logger(`Successfully set upstream for '${branch}'`);
        }
    } else {
        logger(`Remote branch 'origin/${branch}' does not exist, skipping upstream setup`);
    }
}

async function checkGitClean() {
    if (!await isGitClean()) {
        logTelemetryEvent("tengu_teleport_error_git_not_clean", {});
        throw new DisplayError(
            "Git working directory is not clean. Please commit or stash your changes before using --teleport.",
            chalk.red("Error: Git working directory is not clean. Please commit or stash your changes before using --teleport.\n")
        );
    }
}

// --- Session Logic (CH0, Er, Fj2, ZyA, W50, Bx5, Qx5, zSA, RTA, B12) ---

async function validateSessionRepo(session: any): Promise<{ status: string; sessionRepo?: string; currentRepo?: string | null; errorMessage?: string }> {
    const currentRepo = await getGitRepoName();
    const repoSource = session.session_context?.sources?.find((s: any) => s.type === "git_repository");

    if (!repoSource?.url) {
        log(currentRepo ? "Session has no associated repository, proceeding without validation" : "Session has no repo requirement and not in git directory, proceeding");
        return { status: "no_repo_required" };
    }

    const sessionRepo = parseGitRepoName(repoSource.url);
    if (!sessionRepo) return { status: "no_repo_required" };

    logger(`Session is for repository: ${sessionRepo}, current repo: ${currentRepo ?? "none"}`);

    if (!currentRepo) {
        return { status: "not_in_repo", sessionRepo, currentRepo: null };
    }

    if (currentRepo.toLowerCase() === sessionRepo.toLowerCase()) {
        return { status: "match", sessionRepo, currentRepo };
    }

    return { status: "mismatch", sessionRepo, currentRepo };
}

async function fetchSession(sessionId: string, token: string, orgUuid: string): Promise<any> {
    const url = `${BASE_API_URL}/v1/sessions/${sessionId}`;
    const response = await axios.get(url, {
        headers: {
            ...getAuthHeaders(token),
            "x-organization-uuid": orgUuid
        }
    });
    return response.data;
}

async function fetchSessionLogs(sessionId: string, token: string, orgUuid: string): Promise<any[]> {
    const url = `${BASE_API_URL}/v1/sessions/${sessionId}/logs`;
    const response = await axios.get(url, {
        headers: {
            ...getAuthHeaders(token),
            "x-organization-uuid": orgUuid
        }
    });
    return response.data;
}

function getBranchFromSession(session: any): string | undefined {
    // Logic from W50 in chunk_333.ts
    const outcome = session.session_context?.outcomes?.find((o: any) => o.type === "git_repository");
    return outcome?.git_info?.branches?.[0];
}

export async function resumeSession(sessionId: string, onUpdate?: (status: string) => void) {
    logger(`Resuming code session ID: ${sessionId}`);
    try {
        const tokens = getStoredOauthTokens();
        const token = tokens?.accessToken;
        if (!token) {
            logTelemetryEvent("tengu_teleport_resume_error", { error_type: "no_access_token" });
            throw new Error("Claude Code web sessions require authentication with a Claude.ai account. API key authentication is not sufficient. Please run /login to authenticate, or check your authentication status with /status.");
        }

        const orgUuid = await getOrganizationUuid();
        if (!orgUuid) {
            logTelemetryEvent("tengu_teleport_resume_error", { error_type: "no_org_uuid" });
            throw new Error("Unable to get organization UUID for constructing session URL");
        }

        onUpdate?.("validating");
        const session = await fetchSession(sessionId, token, orgUuid);
        const validation = await validateSessionRepo(session);

        switch (validation.status) {
            case "match":
            case "no_repo_required":
                break;
            case "not_in_repo":
                logTelemetryEvent("tengu_teleport_error_repo_not_in_git_dir_sessions_api", { sessionId });
                throw new DisplayError(
                    `You must run claude --teleport ${sessionId} from a checkout of ${validation.sessionRepo}.`,
                    chalk.red(`You must run claude --teleport ${sessionId} from a checkout of ${chalk.bold(validation.sessionRepo ?? '')}.\n`)
                );
            case "mismatch":
                logTelemetryEvent("tengu_teleport_error_repo_mismatch_sessions_api", { sessionId });
                throw new DisplayError(
                    `You must run claude --teleport ${sessionId} from a checkout of ${validation.sessionRepo}.\nThis repo is ${validation.currentRepo}.`,
                    chalk.red(`You must run claude --teleport ${sessionId} from a checkout of ${chalk.bold(validation.sessionRepo ?? '')}.\nThis repo is ${chalk.bold(validation.currentRepo ?? '')}.\n`)
                );
            default:
                throw new DisplayError(validation.errorMessage || "Failed to validate session repository", chalk.red(`Error: ${validation.errorMessage || "Failed to validate session repository"}\n`));
        }

        // Fetch logs and branch
        const startTime = Date.now();
        logger(`[teleport] Starting fetch for session: ${sessionId}`);
        onUpdate?.("fetching_logs");

        const logs = await fetchSessionLogs(sessionId, token, orgUuid);
        logger(`[teleport] Session logs fetched in ${Date.now() - startTime}ms`);

        // Filter messages (simplified logic from jTA)
        const messages = logs.filter(entry =>
            (entry.type === "user" || entry.type === "assistant") && !entry.isSidechain
        );

        onUpdate?.("fetching_branch");
        const branch = getBranchFromSession(session);
        if (branch) logger(`[teleport] Found branch: ${branch}`);

        return { log: messages, branch };

    } catch (err) {
        if (err instanceof DisplayError) throw err;
        const error = err instanceof Error ? err : new Error(String(err));
        logError("teleport", error);
        logTelemetryEvent("tengu_teleport_resume_error", { error_type: "resume_session_id_catch" });

        if (axios.isAxiosError(err) && err.response?.status === 404) {
            throw new DisplayError(`${sessionId} not found.`, `${sessionId} not found.\n${chalk.dim("Run /status in Claude Code to check your account.")}`);
        }

        throw new DisplayError(error.message, chalk.red(`Error: ${error.message}\n`));
    }
}

export async function pollSessionEvents(sessionId: string) {
    const tokens = getStoredOauthTokens();
    const token = tokens?.accessToken;
    if (!token) throw new Error("No access token for polling");

    const orgUuid = await getOrganizationUuid();
    if (!orgUuid) throw new Error("No org UUID for polling");

    const url = `${BASE_API_URL}/v1/sessions/${sessionId}/events`;
    const response = await axios.get(url, {
        headers: {
            ...getAuthHeaders(token),
            "x-organization-uuid": orgUuid
        },
        timeout: 30000
    });

    if (response.status !== 200) throw new Error(`Failed to fetch session events: ${response.statusText}`);

    const data = response.data;
    if (!data?.data || !Array.isArray(data.data)) throw new Error("Invalid events response");

    const events = data.data.filter((e: any) =>
        e && typeof e === "object" && e.type !== "env_manager_log" && e.type !== "control_response" && "session_id" in e
    );

    let branch: string | undefined;
    try {
        const session = await fetchSession(sessionId, token, orgUuid);
        branch = getBranchFromSession(session);
    } catch { }

    return { log: events, branch };
}

function getAnthropicResponseText(response: any): string {
    if (!response || !response.content || !Array.isArray(response.content)) return "";
    return response.content
        .filter((c: any) => c.type === "text")
        .map((c: any) => c.text)
        .join("");
}

async function generateTaskMetadata(description: string, signal?: AbortSignal): Promise<{ title: string; branchName: string }> {
    try {
        const prompt = REMOTE_DESC_TITLE_BRANCH_PROMPT.replace("{description}", description);
        const response = await callClaude({
            systemPrompt: "You are a helpful assistant.",
            messages: [{ role: "user", content: prompt }],
            model: getActiveModel(),
            signal
        });
        const text = getAnthropicResponseText(response);
        const titleMatch = text.match(/<title>(.*?)<\/title>/);
        const branchMatch = text.match(/<branch>(.*?)<\/branch>/);
        return {
            title: titleMatch ? titleMatch[1].trim() : "Background Task",
            branchName: branchMatch ? branchMatch[1].trim() : `claude/task-${randomUUID().substring(0, 8)}`
        };
    } catch (err) {
        logError("teleport", err as Error);
        return { title: "Background Task", branchName: `claude/task-${randomUUID().substring(0, 8)}` };
    }
}

async function fetchAvailableEnvironments(token: string, orgUuid: string): Promise<any[]> {
    const url = `${BASE_API_URL}/v1/environment_providers`;
    const response = await axios.get(url, {
        headers: {
            ...getAuthHeaders(token),
            "x-organization-uuid": orgUuid
        }
    });
    return response.data;
}

export async function createRemoteSession(initialMessage: string, description?: string, signal?: AbortSignal) {
    try {
        // await checkAuthentication(); // Assuming this is handled before calling this
        const tokens = getStoredOauthTokens();
        const token = tokens?.accessToken;
        if (!token) {
            logError("teleport", new Error("No access token found for remote session creation"));
            return null;
        }

        const orgUuid = await getOrganizationUuid();
        if (!orgUuid) {
            logError("teleport", new Error("Unable to get organization UUID for remote session creation"));
            return null;
        }

        await checkGitClean();

        const repoName = await getGitRepoName();
        const { title, branchName } = await generateTaskMetadata(description || initialMessage || "Background task", signal);

        let source: any = null;
        let outcome: any = null;

        if (repoName) {
            const [owner, name] = repoName.split("/");
            if (owner && name) {
                source = {
                    type: "git_repository",
                    url: `https://github.com/${owner}/${name}`,
                    revision: branchName
                };
                outcome = {
                    type: "git_repository",
                    git_info: {
                        type: "github",
                        repo: `${owner}/${name}`,
                        branches: [branchName]
                    }
                };
            } else {
                logError("teleport", new Error(`Invalid repository format: ${repoName} - expected 'owner/name'`));
            }
        }

        const envs = await fetchAvailableEnvironments(token, orgUuid);
        if (!envs || envs.length === 0) {
            logError("teleport", new Error("No environments available for session creation"));
            return null;
        }

        // Use default environment from settings if available
        const settings = getSettings("userSettings") as any;
        const defaultEnvId = settings.remote?.defaultEnvironmentId;
        let selectedEnv = envs[0];
        if (defaultEnvId) {
            const found = envs.find((e: any) => e.environment_id === defaultEnvId);
            if (found) {
                selectedEnv = found;
                logger(`Using configured default environment: ${defaultEnvId}`);
            } else {
                logger(`Configured default environment ${defaultEnvId} not found in available environments, using first available`);
            }
        }

        const envId = selectedEnv.environment_id;
        logger(`Selected environment: ${envId} (${selectedEnv.name})`);

        const url = `${BASE_API_URL}/v1/sessions`;
        const payload = {
            title,
            events: initialMessage ? [{
                type: "event",
                data: {
                    uuid: randomUUID(),
                    session_id: "",
                    type: "user",
                    parent_tool_use_id: null,
                    message: { role: "user", content: initialMessage }
                }
            }] : [],
            session_context: {
                sources: source ? [source] : [],
                outcomes: outcome ? [outcome] : [],
                model: getActiveModel()
            },
            environment_id: envId
        };

        logger(`Creating session with payload: ${JSON.stringify(payload, null, 2)}`);
        const response = await axios.post(url, payload, {
            headers: {
                ...getAuthHeaders(token),
                "x-organization-uuid": orgUuid
            },
            signal
        });

        if (response.status !== 200 && response.status !== 201) {
            logError("teleport", new Error(`API request failed with status ${response.status}: ${response.statusText}\n\nResponse data: ${JSON.stringify(response.data, null, 2)}`));
            return null;
        }

        const result = response.data;
        if (result && typeof result.id === "string") {
            logger(`Successfully created remote session: ${result.id}`);
            return {
                id: result.id,
                title: result.title || title
            };
        }

        logError("teleport", new Error(`Cannot determine session ID from API response: ${JSON.stringify(response.data)}`));
        return null;

    } catch (err) {
        logError("teleport", err instanceof Error ? err : Error(String(err)));
        return null;
    }
}

// --- Task Polling & Notifications (Dx5, Hx5, Vx5, Kx5) ---

async function summarizeDelta(messages: any[], previousSummary: string | null): Promise<string | null> {
    try {
        const response = await callClaude({
            systemPrompt: "You are given a few messages from a conversation, as well as a summary of the conversation so far. Your task is to summarize the new messages in the conversation based on the summary so far. Aim for 1-2 sentences at most, focusing on the most important details. The summary MUST be in <summary>summary goes here</summary> tags. If there is no new information, return an empty string: <summary></summary>.",
            messages: [{ role: "user", content: `Summary so far: ${previousSummary || "No summary yet"}\n\nNew messages: ${JSON.stringify(messages)}` }],
            model: getActiveModel()
        });
        const text = getAnthropicResponseText(response);
        const match = text.match(/<summary>([\s\S]*?)<\/summary>/);
        return match ? match[1].trim() : null;
    } catch (err) {
        logError("teleport", err as Error);
        return null;
    }
}

function getTodosFromLog(logEntries: any[]): string[] {
    // Simplified logic to extract todos from the latest assistant message with a specific tool use
    // In original code, it looks for tool use named "TODO_TOOL" (MX.name)
    const lastAssistant = [...logEntries].reverse().find((e: any) => e.type === "assistant" && e.message?.content?.some((c: any) => c.type === "tool_use" && c.name === "todo"));
    if (!lastAssistant) return [];

    const toolUse = lastAssistant.message.content.find((c: any) => c.type === "tool_use" && c.name === "todo");
    return toolUse?.input?.todos || [];
}

function notifyTaskStatus(taskId: string, title: string, status: "completed" | "failed" | "killed", _setAppState: any) {
    const summary = `Remote task "${title}" ${status === "completed" ? "completed successfully" : status === "failed" ? "failed" : "was killed"}.`;
    const notification = `<task-notification>
<task-id>${taskId}</task-id>
<task-type>remote_agent</task-type>
<status>${status}</status>
<summary>${summary}</summary>
Use TaskOutputTool with task_id="${taskId}" to retrieve the output.
</task-notification>`;

    updateAppState(prev => ({
        ...prev,
        queuedCommands: [...(prev.queuedCommands || []), { value: notification, mode: "agent-notification" }],
        tasks: {
            ...prev.tasks,
            [taskId]: { ...prev.tasks[taskId], notified: true }
        }
    }));
}

function pollRemoteTask(taskId: string, _setAppState: any) {
    let active = true;
    const interval = 1000;

    const run = async () => {
        if (!active) return;
        try {
            const state = getAppState();
            const task = state.tasks?.[taskId];
            if (!task || task.status !== "running") return;

            const { log: currentLog } = await pollSessionEvents(task.sessionId);
            const resultEntry = currentLog.find((e: any) => e.type === "result");
            const newStatus = resultEntry ? (resultEntry.subtype === "success" ? "completed" : "failed") : (currentLog.length > 0 ? "running" : "starting");

            const newEntries = currentLog.slice(task.log.length);
            let summaryDelta = task.deltaSummarySinceLastFlushToAttachment;

            if (newEntries.length > 0) {
                summaryDelta = await summarizeDelta(newEntries, summaryDelta);
                const newOutput = newEntries.map((e: any) => {
                    if (e.type === "assistant") {
                        return e.message?.content?.filter((c: any) => c.type === "text").map((c: any) => c.text).join('\n') || "";
                    }
                    return JSON.stringify(e);
                }).join('\n');

                if (newOutput) {
                    appendTaskOutput(taskId, newOutput + '\n');
                }
            }

            updateAppState(prev => ({
                ...prev,
                tasks: {
                    ...prev.tasks,
                    [taskId]: {
                        ...prev.tasks[taskId],
                        status: newStatus === "starting" ? "running" : newStatus,
                        log: currentLog,
                        todoList: getTodosFromLog(currentLog),
                        deltaSummarySinceLastFlushToAttachment: summaryDelta,
                        endTime: resultEntry ? Date.now() : undefined
                    }
                }
            }));

            if (resultEntry) {
                notifyTaskStatus(taskId, task.title, newStatus as any, _setAppState);
                return;
            }
        } catch (err) {
            logError("teleport", err as Error);
        }
        if (active) setTimeout(run, interval);
    };

    run();
    return () => { active = false; };
}

// --- RemoteAgentTask Tool Implementation ---

export const RemoteAgentTaskRequest = {
    name: "RemoteAgentTask",
    type: "remote_agent",
    async spawn(params: any, context: any) {
        const { command, title } = params;
        const { setAppState, abortController } = context;

        logger(`RemoteAgentTask spawning: ${title}`);
        const session = await createRemoteSession(command, title, abortController.signal);
        if (!session) throw new Error("Failed to create remote session");

        const taskId = `r${session.id.substring(0, 6)}`;
        initTaskOutput(taskId);

        const task = {
            id: taskId,
            type: "remote_agent",
            status: "running",
            sessionId: session.id,
            command,
            title: session.title || title,
            todoList: [],
            log: [],
            deltaSummarySinceLastFlushToAttachment: null,
            startTime: Date.now(),
            outputFile: getTaskOutputPath(taskId),
            outputOffset: 0,
            notified: false
        };

        updateAppState(prev => ({
            ...prev,
            tasks: {
                ...prev.tasks,
                [taskId]: task
            }
        }));

        const cleanupPoll = pollRemoteTask(taskId, setAppState);

        return {
            taskId,
            cleanup: () => {
                cleanupPoll();
            }
        };
    },
    async kill(taskId: string, _context: any) {
        updateAppState(prev => {
            const task = prev.tasks?.[taskId];
            if (!task || task.status !== "running") return prev;
            return {
                ...prev,
                tasks: {
                    ...prev.tasks,
                    [taskId]: {
                        ...task,
                        status: "killed",
                        endTime: Date.now()
                    }
                }
            };
        });
        logger(`RemoteAgentTask ${taskId} marked as killed (local only)`);
    },
    renderStatus(task: any) {
        const status = task.status;
        const color = status === "running" ? "yellow" : status === "completed" ? "green" : status === "failed" ? "red" : "gray";
        return (
            <Box>
                <Text color={color}>[{status}] </Text>
                <Text>{task.title}</Text>
            </Box>
        );
    },
    renderOutput(output: string) {
        return (
            <Box paddingLeft={2}>
                <Text dimColor>{output}</Text>
            </Box>
        );
    },
    getProgressMessage(task: any) {
        const summary = task.deltaSummarySinceLastFlushToAttachment;
        if (!summary) return null;
        return `Remote task ${task.id} progress: ${summary}. Read ${task.outputFile} to see full output.`;
    }
};

// --- Utilities ---

export function getSessionUrl(sessionId: string) {
    return `https://claude.ai/code/${sessionId}`;
}

export function getTeleportCommand(sessionId: string) {
    return `claude --teleport ${sessionId}`;
}
