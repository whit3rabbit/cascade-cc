import React, { useCallback, useEffect, useRef, useState } from "react";
import { Box, Text, useInput, useStdout } from "ink";
import InkTextInput from "ink-text-input";
import ExternalLink from "./ExternalLink.js";

type OAuthFlowState =
    | { state: "starting" }
    | { state: "waiting_for_login"; url: string }
    | { state: "processing" }
    | { state: "success"; token: string }
    | { state: "about_to_retry"; nextState: OAuthFlowState }
    | { state: "error"; message: string; toRetry?: OAuthFlowState };

type OAuthResult = { accessToken: string };

const TextInput = InkTextInput as unknown as React.FC<{
    value: string;
    onChange: (value: string) => void;
    onSubmit?: (value: string) => void;
    placeholder?: string;
    focus?: boolean;
    showCursor?: boolean;
    columns?: number;
    cursorOffset?: number;
    onChangeCursorOffset?: (offset: number) => void;
}>;

const AUTH_PROMPT = "Paste code here if prompted > ";

function trackEvent(_name: string, _payload?: Record<string, any>) { }
function logError(_error: Error) { }
function isInteractive(): boolean {
    return true;
}
async function ensureUiReady(): Promise<void> { }
function persistOAuthToken(_result: OAuthResult): { warning?: string } {
    return {};
}

class GitHubAuthClient {
    private pendingResolve?: (value: OAuthResult) => void;
    private pendingReject?: (error: Error) => void;
    private pendingTimer?: NodeJS.Timeout;

    async startOAuthFlow(
        onLoginUrl: (url: string) => Promise<void> | void,
        _options: { loginWithClaudeAi: boolean; inferenceOnly: boolean; expiresIn: number }
    ): Promise<OAuthResult> {
        const url = "https://claude.ai/code/oauth";
        await onLoginUrl(url);

        return new Promise<OAuthResult>((resolve, reject) => {
            this.pendingResolve = resolve;
            this.pendingReject = reject;
            this.pendingTimer = setTimeout(() => {
                this.pendingResolve?.({ accessToken: "oauth-token" });
                this.cleanup();
            }, 1500);
        });
    }

    handleManualAuthCodeInput(_payload: { authorizationCode: string; state: string }) {
        if (!this.pendingResolve) return;
        this.pendingResolve({ accessToken: "oauth-token" });
        this.cleanup();
    }

    cleanup() {
        if (this.pendingTimer) clearTimeout(this.pendingTimer);
        this.pendingTimer = undefined;
        this.pendingResolve = undefined;
        this.pendingReject = undefined;
    }
}

function InlineSpinner() {
    return <Text color="success">●</Text>;
}

export function GitHubOAuthWizard({
    onSuccess,
    onCancel
}: {
    onSuccess: (token: string) => void;
    onCancel: () => void;
}) {
    const [flow, setFlow] = useState<OAuthFlowState>({ state: "starting" });
    const [authClient] = useState(() => new GitHubAuthClient());
    const [manualCode, setManualCode] = useState("");
    const [cursorOffset, setCursorOffset] = useState(0);
    const [showManualEntry, setShowManualEntry] = useState(false);
    const timeoutsRef = useRef(new Set<NodeJS.Timeout>());
    const { stdout } = useStdout();
    const columns = stdout?.columns || 80;
    const inputWidth = Math.max(50, columns - AUTH_PROMPT.length - 4);

    useInput((_input, key) => {
        if (flow.state !== "error") return;
        if (key.return && flow.toRetry) {
            setManualCode("");
            setCursorOffset(0);
            setFlow({ state: "about_to_retry", nextState: flow.toRetry });
        } else {
            onCancel();
        }
    });

    const handleManualCodeSubmit = useCallback(
        async (value: string, url: string) => {
            try {
                const [authorizationCode, state] = value.split("#");
                if (!authorizationCode || !state) {
                    setFlow({
                        state: "error",
                        message: "Invalid code. Please make sure the full code was copied",
                        toRetry: { state: "waiting_for_login", url }
                    });
                    return;
                }
                trackEvent("tengu_oauth_manual_entry", {});
                authClient.handleManualAuthCodeInput({ authorizationCode, state });
            } catch (error) {
                const err = error instanceof Error ? error : new Error(String(error));
                logError(err);
                setFlow({
                    state: "error",
                    message: err.message,
                    toRetry: { state: "waiting_for_login", url }
                });
            }
        },
        [authClient]
    );

    const startOAuth = useCallback(async () => {
        timeoutsRef.current.forEach((timeout) => clearTimeout(timeout));
        timeoutsRef.current.clear();

        try {
            const result = await authClient.startOAuthFlow(async (url) => {
                setFlow({ state: "waiting_for_login", url });
                const manualTimer = setTimeout(() => setShowManualEntry(true), 3000);
                timeoutsRef.current.add(manualTimer);
            }, {
                loginWithClaudeAi: true,
                inferenceOnly: true,
                expiresIn: 31536000
            });

            if (!isInteractive()) await ensureUiReady();

            setFlow({ state: "processing" });
            const persisted = persistOAuthToken(result);
            if (persisted.warning) {
                trackEvent("tengu_oauth_storage_warning", { warning: persisted.warning });
            }

            const successTimer = setTimeout(() => {
                setFlow({ state: "success", token: result.accessToken });
                const doneTimer = setTimeout(() => {
                    onSuccess(result.accessToken);
                }, 1000);
                timeoutsRef.current.add(doneTimer);
            }, 100);
            timeoutsRef.current.add(successTimer);
        } catch (error) {
            if (!isInteractive()) await ensureUiReady();
            const err = error instanceof Error ? error : new Error(String(error));
            setFlow({ state: "error", message: err.message, toRetry: { state: "starting" } });
            logError(err);
            trackEvent("tengu_oauth_error", { error: err.message });
        }
    }, [authClient, onSuccess]);

    useEffect(() => {
        if (flow.state === "starting") startOAuth();
    }, [flow.state, startOAuth]);

    useEffect(() => {
        if (flow.state !== "about_to_retry") return;
        if (!isInteractive()) ensureUiReady();
        const retryTimer = setTimeout(() => {
            if (flow.nextState.state === "waiting_for_login") setShowManualEntry(true);
            else setShowManualEntry(false);
            setFlow(flow.nextState);
        }, 500);
        timeoutsRef.current.add(retryTimer);
    }, [flow]);

    useEffect(() => {
        const timeouts = timeoutsRef.current;
        return () => {
            authClient.cleanup();
            timeouts.forEach((timeout) => clearTimeout(timeout));
            timeouts.clear();
        };
    }, [authClient]);

    const renderStatus = () => {
        switch (flow.state) {
            case "starting":
                return (
                    <Box>
                        <InlineSpinner />
                        <Text> Starting authentication…</Text>
                    </Box>
                );
            case "waiting_for_login":
                return (
                    <Box flexDirection="column" gap={1}>
                        {!showManualEntry && (
                            <Box>
                                <InlineSpinner />
                                <Text> Opening browser to sign in with your Claude account…</Text>
                            </Box>
                        )}
                        {showManualEntry && (
                            <Box>
                                <Text>{AUTH_PROMPT}</Text>
                                <TextInput
                                    value={manualCode}
                                    onChange={setManualCode}
                                    onSubmit={(value) => handleManualCodeSubmit(value, flow.url)}
                                    cursorOffset={cursorOffset}
                                    onChangeCursorOffset={setCursorOffset}
                                    columns={inputWidth}
                                />
                            </Box>
                        )}
                    </Box>
                );
            case "processing":
                return (
                    <Box>
                        <InlineSpinner />
                        <Text> Processing authentication…</Text>
                    </Box>
                );
            case "success":
                return (
                    <Box flexDirection="column" gap={1}>
                        <Text color="success">✓ Authentication token created successfully!</Text>
                        <Text dimColor>Using token for GitHub Actions setup…</Text>
                    </Box>
                );
            case "error":
                return (
                    <Box flexDirection="column" gap={1}>
                        <Text color="error">OAuth error: {flow.message}</Text>
                        {flow.toRetry ? (
                            <Text dimColor>Press Enter to try again, or any other key to cancel</Text>
                        ) : (
                            <Text dimColor>Press any key to return to API key selection</Text>
                        )}
                    </Box>
                );
            case "about_to_retry":
                return (
                    <Box flexDirection="column" gap={1}>
                        <Text color="permission">Retrying…</Text>
                    </Box>
                );
            default:
                return null;
        }
    };

    return (
        <Box flexDirection="column" gap={1}>
            {flow.state === "starting" && (
                <Box flexDirection="column" gap={1} paddingBottom={1}>
                    <Text bold>Create Authentication Token</Text>
                    <Text dimColor>Creating a long-lived token for GitHub Actions</Text>
                </Box>
            )}

            {flow.state !== "success" && flow.state !== "starting" && flow.state !== "processing" && (
                <Box flexDirection="column" gap={1} paddingBottom={1}>
                    <Text bold>Create Authentication Token</Text>
                    <Text dimColor>Creating a long-lived token for GitHub Actions</Text>
                </Box>
            )}

            {flow.state === "waiting_for_login" && showManualEntry && (
                <Box flexDirection="column" gap={1} paddingBottom={1}>
                    <Box paddingX={1}>
                        <Text dimColor>Browser didn't open? Use the url below to sign in:</Text>
                    </Box>
                    <Box>
                        <ExternalLink url={flow.url} label={flow.url} />
                    </Box>
                </Box>
            )}

            <Box paddingLeft={1} flexDirection="column" gap={1}>
                {renderStatus()}
            </Box>
        </Box>
    );
}
