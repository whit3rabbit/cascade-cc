// Logic from chunk_539.ts (Feedback & Session Lifecycle)

import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Box, Text, useInput } from "ink";
import InkTextInput from "ink-text-input";
import { useCtrlExit } from "../../hooks/useCtrlExit.js";

const TextInput = InkTextInput as unknown as React.FC<{
    value: string;
    onChange: (value: string) => void;
    onSubmit?: () => void;
    columns?: number;
    cursorOffset?: number;
    onChangeCursorOffset?: (offset: number) => void;
}>;

const ISSUE_URL_MAX_LENGTH = 7250;
const ISSUE_REPO_URL = "https://github.com/anthropics/claude-code/issues";

const VERSION_INFO = {
    ISSUES_EXPLAINER: "report the issue at https://github.com/anthropics/claude-code/issues",
    PACKAGE_URL: "@anthropic-ai/claude-code",
    README_URL: "https://code.claude.com/docs/en/overview",
    VERSION: "2.0.76",
    FEEDBACK_CHANNEL: "https://github.com/anthropics/claude-code/issues",
    BUILD_TIME: "2025-12-22T23:56:12Z"
};

type FeedbackDialogProps = {
    abortSignal?: AbortSignal;
    messages: any[];
    initialDescription?: string;
    onDone: (message: string, meta?: any) => void;
};

type FeedbackStep = "userInput" | "consent" | "submitting" | "done";

type GitState = {
    branchName?: string;
    commitHash?: string;
    remoteUrl?: string;
    isHeadOnRemote?: boolean;
    isClean?: boolean;
};

type FeedbackPayload = {
    latestAssistantMessageId: string | null;
    message_count: number;
    datetime: string;
    description: string;
    platform: string;
    gitRepo: boolean;
    terminal: string;
    version: string;
    transcript: any[];
    errors: any[];
    lastApiRequest: any;
    subagentTranscripts?: Record<string, any[]>;
};

function redactString(value: string): string {
    return value.replace(/sk-[A-Za-z0-9_-]+/g, "sk-***");
}

// --- Transcript parser (A47) ---
export function getSubagentIds(messages: any[]): string[] {
    const ids: string[] = [];
    for (const message of messages) {
        if (message.type !== "user") continue;
        const content = message.message?.content;
        if (!Array.isArray(content)) continue;
        for (const block of content) {
            if (block.type !== "tool_result") continue;
            const result = block.content;
            if (typeof result === "string") {
                try {
                    const parsed = JSON.parse(result);
                    if (parsed && typeof parsed.agentId === "string") ids.push(parsed.agentId);
                } catch {
                    // ignore parse errors
                }
            } else if (Array.isArray(result)) {
                for (const entry of result) {
                    if (entry.type !== "text" || typeof entry.text !== "string") continue;
                    try {
                        const parsed = JSON.parse(entry.text);
                        if (parsed && typeof parsed.agentId === "string") ids.push(parsed.agentId);
                    } catch {
                        // ignore parse errors
                    }
                }
            }
        }
    }
    return Array.from(new Set(ids));
}

async function fetchSubagentTranscript(_agentId: string): Promise<any[]> {
    return [];
}

// --- Subagent transcript fetch (Q47) ---
export async function batchFetchTranscripts(agentIds: string[]): Promise<Record<string, any[]>> {
    const results = await Promise.all(
        agentIds.map(async (agentId) => {
            try {
                const transcript = await fetchSubagentTranscript(agentId);
                if (transcript && transcript.length > 0) return { agentId, transcript };
                return null;
            } catch {
                return null;
            }
        })
    );

    const byAgent: Record<string, any[]> = {};
    for (const entry of results) {
        if (entry) byAgent[entry.agentId] = entry.transcript;
    }
    return byAgent;
}

function getRawErrors(): any[] {
    return [];
}

// --- Redacted errors (x89) ---
export function getRedactedErrors(): any[] {
    return getRawErrors().map((error) => {
        const clone = { ...error };
        if (clone && typeof clone.error === "string") clone.error = redactString(clone.error);
        return clone;
    });
}

function getLatestAssistantMessageId(messages: any[]): string | null {
    const last = [...messages].reverse().find((message) => message.type === "assistant");
    return last?.requestId ?? null;
}

function getLastApiRequest(): any {
    return null;
}

function getTerminalName(): string {
    return process.env.TERM_PROGRAM ?? "terminal";
}

async function detectGitState(): Promise<{ isGit: boolean; gitState: GitState | null }> {
    return { isGit: false, gitState: null };
}

function generateIssueTitleFallback(description: string): string {
    const firstLine = description.split("\n")[0] || "";
    if (firstLine.length <= 60 && firstLine.length > 5) return firstLine;
    let trimmed = firstLine.slice(0, 60);
    if (firstLine.length > 60) {
        const lastSpace = trimmed.lastIndexOf(" ");
        if (lastSpace > 30) trimmed = trimmed.slice(0, lastSpace);
        trimmed += "...";
    }
    return trimmed.length < 10 ? "Bug Report" : trimmed;
}

// --- Title generator (G47) ---
export async function generateIssueTitle(description: string, _signal?: AbortSignal): Promise<string> {
    return generateIssueTitleFallback(description);
}

function handleFeedbackError(error: unknown): void {
    if (error instanceof Error) {
        const safe = new Error(redactString(error.message));
        if (error.stack) safe.stack = redactString(error.stack);
        console.error(safe);
    } else {
        console.error(new Error(redactString(String(error))));
    }
}

// --- Feedback submission (Z47) ---
export async function submitFeedbackToApi(_payload: FeedbackPayload): Promise<{ success: boolean; feedbackId?: string; isZdrOrg?: boolean }> {
    return { success: true, feedbackId: "feedback-id" };
}

// --- GitHub issue URL (B47) ---
export function generateGitHubIssueUrl(feedbackId: string, issueTitle: string, description: string, errors: any[]): string {
    const safeTitle = redactString(issueTitle);
    const safeDescription = redactString(description);
    const header = encodeURIComponent(`**Bug Description**\n${safeDescription}\n\n**Environment Info**\n- Platform: ${process.platform}\n- Terminal: ${getTerminalName()}\n- Version: ${VERSION_INFO.VERSION || "unknown"}\n- Feedback ID: ${feedbackId}\n\n**Errors**\n\
\
\
`);
    const footer = encodeURIComponent("\n\
\
\
");
    const truncatedNote = encodeURIComponent("\n**Note:** Error logs were truncated.\n");
    const errorsJson = JSON.stringify(errors);
    const encodedErrors = encodeURIComponent(errorsJson);
    const baseUrl = `${ISSUE_REPO_URL}/new?title=${encodeURIComponent(safeTitle)}&labels=user-reported,bug&body=`;
    const remaining = ISSUE_URL_MAX_LENGTH - baseUrl.length - header.length - footer.length - truncatedNote.length;

    let body = "";
    if (encodedErrors.length <= remaining) {
        body = header + encodedErrors + footer;
    } else {
        const truncated = encodedErrors.substring(0, remaining);
        body = header + truncated + footer + truncatedNote;
    }

    return `${ISSUE_REPO_URL}/new?title=${encodeURIComponent(safeTitle)}&body=${body}&labels=user-reported,bug`;
}

function openExternalUrl(url: string) {
    const handler = (globalThis as any).openExternalUrl || (globalThis as any).openUrl;
    if (typeof handler === "function") handler(url);
}

export function FeedbackDialog({ abortSignal, messages, initialDescription, onDone }: FeedbackDialogProps) {
    const [step, setStep] = useState<FeedbackStep>("userInput");
    const [cursorOffset, setCursorOffset] = useState(0);
    const [description, setDescription] = useState(initialDescription ?? "");
    const [feedbackId, setFeedbackId] = useState<string | null>(null);
    const [issueTitle, setIssueTitle] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [gitInfo, setGitInfo] = useState<{ isGit: boolean; gitState: GitState | null }>({
        isGit: false,
        gitState: null
    });
    const exitState = useCtrlExit();
    const inputWidth = Math.max((process.stdout?.columns ?? 80) - 4, 40);

    useEffect(() => {
        detectGitState().then(setGitInfo);
    }, []);

    const submit = useCallback(async () => {
        setStep("submitting");
        setError(null);
        setFeedbackId(null);

        const redactedErrors = getRedactedErrors();
        const requestId = getLatestAssistantMessageId(messages);
        const subagentIds = getSubagentIds(messages);
        const subagentTranscripts = await batchFetchTranscripts(subagentIds);

        const payload: FeedbackPayload = {
            latestAssistantMessageId: requestId,
            message_count: messages.length,
            datetime: new Date().toISOString(),
            description,
            platform: process.platform,
            gitRepo: gitInfo.isGit,
            terminal: getTerminalName(),
            version: VERSION_INFO.VERSION,
            transcript: messages,
            errors: redactedErrors,
            lastApiRequest: getLastApiRequest(),
            ...(Object.keys(subagentTranscripts).length > 0 ? { subagentTranscripts } : {})
        };

        try {
            const [response, title] = await Promise.all([
                submitFeedbackToApi(payload),
                generateIssueTitle(description, abortSignal)
            ]);

            setIssueTitle(title);
            if (response.success) {
                if (response.feedbackId) setFeedbackId(response.feedbackId);
                setStep("done");
            } else {
                if (response.isZdrOrg) {
                    setError(
                        "Feedback collection is not available for organizations with custom data retention policies."
                    );
                } else {
                    setError("Could not submit feedback. Please try again later.");
                }
                setStep("done");
            }
        } catch (err) {
            handleFeedbackError(err);
            setError("Could not submit feedback. Please try again later.");
            setStep("done");
        }
    }, [abortSignal, description, gitInfo.isGit, messages]);

    useInput((input, key) => {
        if (step === "done") {
            if (key.return && issueTitle) {
                const url = generateGitHubIssueUrl(feedbackId ?? "", issueTitle, description, getRedactedErrors());
                openExternalUrl(url);
            }
            if (error) {
                onDone("Error submitting feedback / bug report", { display: "system" });
            } else {
                onDone("Feedback / bug report submitted", { display: "system" });
            }
            return;
        }

        if (error) {
            onDone("Error submitting feedback / bug report", { display: "system" });
            return;
        }

        if (key.escape) {
            onDone("Feedback / bug report cancelled", { display: "system" });
            return;
        }

        if (step === "consent" && (key.return || input === " ")) {
            void submit();
        }
    });

    const footerMessage = useMemo(() => {
        if (exitState.pending) return `Press ${exitState.keyName} again to exit`;
        if (step === "userInput") return "Enter to continue · Esc to cancel";
        if (step === "consent") return "Enter to submit · Esc to cancel";
        return "";
    }, [exitState.pending, exitState.keyName, step]);

    return (
        <>
            <Box
                flexDirection="column"
                borderStyle="round"
                borderColor="permission"
                paddingX={1}
                paddingBottom={1}
                gap={1}
            >
                <Text bold color="permission">
                    Submit Feedback / Bug Report
                </Text>
                {step === "userInput" && (
                    <Box flexDirection="column" gap={1}>
                        <Text>Describe the issue below:</Text>
                        <TextInput
                            value={description}
                            onChange={setDescription}
                            columns={inputWidth}
                            onSubmit={() => setStep("consent")}
                            cursorOffset={cursorOffset}
                            onChangeCursorOffset={setCursorOffset}
                        />
                        {error && (
                            <Box flexDirection="column" gap={1}>
                                <Text color="error">{error}</Text>
                                <Text dimColor>Press any key to close</Text>
                            </Box>
                        )}
                    </Box>
                )}
                {step === "consent" && (
                    <Box flexDirection="column">
                        <Text>This report will include:</Text>
                        <Box marginLeft={2} flexDirection="column">
                            <Text>
                                - Your feedback / bug description: <Text dimColor>{description}</Text>
                            </Text>
                            <Text>
                                - Environment info: <Text dimColor>{process.platform}, {getTerminalName()}, v{VERSION_INFO.VERSION}</Text>
                            </Text>
                            {gitInfo.gitState && (
                                <Text>
                                    - Git repo metadata:{" "}
                                    <Text dimColor>
                                        {gitInfo.gitState.branchName}
                                        {gitInfo.gitState.commitHash
                                            ? `, ${gitInfo.gitState.commitHash.slice(0, 7)}`
                                            : ""}
                                        {gitInfo.gitState.remoteUrl ? ` @ ${gitInfo.gitState.remoteUrl}` : ""}
                                        {gitInfo.gitState.isHeadOnRemote === false ? ", not synced" : ""}
                                        {gitInfo.gitState.isClean === false ? ", has local changes" : ""}
                                    </Text>
                                </Text>
                            )}
                            <Text>- Current session transcript</Text>
                        </Box>
                        <Box marginTop={1}>
                            <Text wrap="wrap" dimColor>
                                We will use your feedback to debug related issues or to improve Claude Code's
                                functionality (eg. to reduce the risk of bugs occurring in the future).
                            </Text>
                        </Box>
                        <Box marginTop={1}>
                            <Text>
                                Press <Text bold>Enter</Text> to confirm and submit.
                            </Text>
                        </Box>
                    </Box>
                )}
                {step === "submitting" && (
                    <Box flexDirection="row" gap={1}>
                        <Text>Submitting report…</Text>
                    </Box>
                )}
                {step === "done" && (
                    <Box flexDirection="column">
                        {error ? (
                            <Text color="error">{error}</Text>
                        ) : (
                            <Text color="success">Thank you for your report!</Text>
                        )}
                        {feedbackId && (
                            <Text dimColor>
                                Feedback ID: {feedbackId}
                            </Text>
                        )}
                        <Box marginTop={1}>
                            <Text>
                                Press <Text bold>Enter</Text> to open your browser and draft a GitHub issue, or any
                                other key to close.
                            </Text>
                        </Box>
                    </Box>
                )}
            </Box>
            <Box marginLeft={1}>
                <Text dimColor>{footerMessage}</Text>
            </Box>
        </>
    );
}
