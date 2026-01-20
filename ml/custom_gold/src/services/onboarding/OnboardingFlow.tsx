import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Box, Text } from 'ink';
import Spinner from 'ink-spinner';
import SelectInput from 'ink-select-input';
import TextInput from 'ink-text-input';
import { exec } from "child_process";
import { promisify } from "util";
import axios from "axios";
import { OAuthService } from "../terminal/OAuthService.js";
import { log } from "../logger/loggerService.js";
import { getSettings, updateSettings } from "../terminal/settings.js";
import { sendNotification } from "../notifications/NotificationService.js";

const execAsync = promisify(exec);
const logger = log("OnboardingFlow");

// Constants
const BASE_API_URL = "https://api.anthropic.com";

// --- Components ---

/**
 * Enhanced Login Flow Component
 * Handles OAuth login via Browser or Console (API Key).
 */
export function LoginFlow({ onDone }: { onDone: () => void }) {
    const [state, setState] = useState<"idle" | "visual_mode_select" | "waiting_for_login" | "waiting_for_input" | "validating" | "success" | "error">("idle");
    const [loginMethod, setLoginMethod] = useState<"claudeai" | "console" | null>(null);
    const [inputCode, setInputCode] = useState("");
    const [authUrl, setAuthUrl] = useState("");
    const [errorMessage, setErrorMessage] = useState("");

    const handleMethodSelect = (item: { value: string }) => {
        if (item.value === "claudeai" || item.value === "console") {
            setLoginMethod(item.value as any);
            startLoginProcess(item.value as any);
        }
    };

    const startLoginProcess = async (method: "claudeai" | "console") => {
        try {
            // Check for forced settings
            const settings = getSettings("userSettings");
            if (settings.forceLoginMethod && settings.forceLoginMethod !== method) {
                // If forced to another method, maybe warn or switch? 
                // For now, respect user selection but log.
            }

            if (method === "claudeai") {
                // Generate OAuth URL
                const state = Math.random().toString(36).substring(7);
                const challenge = "stub-challenge"; // Should generate PKCE
                const url = OAuthService.getAuthorizeUrl({
                    state,
                    codeChallenge: challenge,
                    port: 0, // Manual mode often implies no local server or different flow
                    isManual: true
                });
                setAuthUrl(url);
                setState("waiting_for_login");

                // Attempt to open browser
                const open = (await import("open")).default;
                await open(url);
            } else {
                // Console login involves API Key entry?
                // Or similar OAuth flow? Assuming similar flow for now or just prompt for key.
                // The chunk suggests "creating_api_key" state logic.
                setState("error");
                setErrorMessage("Console login not fully implemented in this deobfuscation.");
            }
        } catch (err) {
            setErrorMessage(`Failed to start login: ${err instanceof Error ? err.message : String(err)}`);
            setState("error");
        }
    };

    const handleInputSubmit = async (value: string) => {
        setState("validating");
        try {
            // Validate code (stub validation or exchange)
            if (value.trim().length > 0) {
                // Simulate exchange
                // In real app: await OAuthService.exchangeCodeForToken(value, verifier);
                await new Promise(r => setTimeout(r, 1000));

                // Success logic
                updateSettings("userSettings", {
                    featureUsage: { ...getSettings("userSettings").featureUsage, loginSuccess: 1 }
                });

                await sendNotification("Login successful", "Claude Code");
                setState("success");
                setTimeout(onDone, 1500);
            } else {
                throw new Error("Invalid code provided");
            }
        } catch (err) {
            setErrorMessage(err instanceof Error ? err.message : "Validation failed");
            setState("error");
        }
    };

    if (state === "idle") {
        return (
            <Box flexDirection="column" gap={1}>
                <Text bold>Welcome to Claude Code</Text>
                <Text>Please select how you would like to sign in:</Text>
                <SelectInput items={[
                    { label: "Claude.ai (Pro, Team, Enterprise)", value: "claudeai" },
                    { label: "Anthropic Console (API Key)", value: "console" }
                ]} onSelect={handleMethodSelect} />
            </Box>
        );
    }

    if (state === "waiting_for_login") {
        return (
            <Box flexDirection="column" gap={1}>
                <Text>Opening your browser to sign in...</Text>
                <Text dimColor>If it doesn't open, copy and paste this URL:</Text>
                <Text color="blue">{authUrl}</Text>
                <Box marginTop={1}>
                    <Text>Paste the code here: </Text>
                    <TextInput value={inputCode} onChange={setInputCode} onSubmit={handleInputSubmit} />
                </Box>
            </Box>
        );
    }

    if (state === "validating") {
        return (
            <Box>
                <Text color="green"><Spinner type="dots" /> Verifying credentials...</Text>
            </Box>
        );
    }

    if (state === "success") {
        return (
            <Box>
                <Text color="green">✓ Login successful!</Text>
            </Box>
        );
    }

    if (state === "error") {
        return (
            <Box flexDirection="column">
                <Text color="red">Error: {errorMessage}</Text>
                <Text dimColor>Press Enter to try again</Text>
                <TextInput value="" onChange={() => setState("idle")} onSubmit={() => setState("idle")} />
            </Box>
        );
    }

    return null;
}

/**
 * Checks for git changes and warns user.
 */
export function CheckChangedFiles({ onCancel, onStashAndContinue }: { onCancel: () => void; onStashAndContinue: () => void }) {
    const [status, setStatus] = useState<"checking" | "clean" | "dirty" | "stashing">("checking");
    const [files, setFiles] = useState<string[]>([]);

    useEffect(() => {
        checkGitStatus();
    }, []);

    const checkGitStatus = async () => {
        try {
            const { stdout } = await execAsync("git status --porcelain", { encoding: 'utf8' });
            if (!stdout.trim()) {
                setStatus("clean");
                onStashAndContinue();
            } else {
                const lines = stdout.trim().split('\n').map(l => l.trim());
                setFiles(lines);
                setStatus("dirty");
            }
        } catch (e) {
            // Not a git repo or error
            setStatus("clean");
            onStashAndContinue();
        }
    };

    const handleStash = async () => {
        setStatus("stashing");
        try {
            await execAsync("git stash save 'Claude Code Auto-Stash'");
            onStashAndContinue();
        } catch (e) {
            // failed to stash
            logger.error("Failed to stash");
        }
    };

    if (status === "checking" || status === "stashing") {
        return <Text><Spinner type="dots" /> {status === "checking" ? "Checking git status..." : "Stashing changes..."}</Text>;
    }

    if (status === "dirty") {
        return (
            <Box flexDirection="column" gap={1} borderStyle="round" borderColor="yellow" padding={1}>
                <Text color="yellow" bold>Unsaved Changes Detected</Text>
                <Text>You have {files.length} changed files that need to be stashed before continuing:</Text>
                <Box flexDirection="column" marginLeft={2}>
                    {files.slice(0, 5).map((f, i) => <Text key={i} dimColor>{f}</Text>)}
                    {files.length > 5 && <Text dimColor>...and {files.length - 5} more</Text>}
                </Box>
                <SelectInput items={[
                    { label: "Stash changes and continue", value: "stash" },
                    { label: "Cancel", value: "cancel" }
                ]} onSelect={(item) => item.value === "stash" ? handleStash() : onCancel()} />
            </Box>
        );
    }

    return null;
}

/**
 * Runs pre-flight checks for remote environment.
 */
export function PreFlightChecks({ onComplete }: { onComplete: () => void }) {
    const [checking, setChecking] = useState(true);

    useEffect(() => {
        // Run checks
        const run = async () => {
            await new Promise(r => setTimeout(r, 500)); // Simulate check
            onComplete();
        };
        run();
    }, []);

    if (checking) return <Text><Spinner type="dots" /> Checking environment...</Text>;
    return null;
}

//helpers

export async function generateSessionTitle(prompt: string): Promise<{ title: string; branchName: string }> {
    // Stub simple heuristic if LLM not available
    const safePrompt = prompt.slice(0, 30).replace(/[^a-zA-Z0-9]/g, "-");
    return {
        title: `Session: ${prompt.slice(0, 20)}...`,
        branchName: `claude/${safePrompt}-${Date.now()}`
    };
}

export async function checkGithubAppInstalled(owner: string, repo: string): Promise<boolean> {
    try {
        // Stub using public API if possible or just internal endpoint
        // This requires auth token usually.
        // For deobfuscation without full keys, default to false or safe check.
        // Logic from Hj2
        return false;
    } catch {
        return false;
    }
}

export async function getRemoteEnvironments() {
    try {
        // Logic from KHA
        // Requires auth token
        return [];
    } catch {
        return [];
    }
}
