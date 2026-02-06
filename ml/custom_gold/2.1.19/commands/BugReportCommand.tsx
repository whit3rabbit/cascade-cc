import React, { useState, useEffect, useCallback } from 'react';
import { Box, Text, useInput } from 'ink';
import TextInput from 'ink-text-input';
import { getProductVersion } from '../utils/shared/product.js';
import { platform, terminal } from '../utils/shared/process.js';
import { getGitState, isGitRepo } from '../utils/shared/git.js';
import { formatTranscript } from '../utils/shared/transcript.js';
import { submitFeedback } from '../services/terminal/FeedbackService.js';
import { openUrl } from '../utils/shared/open.js';
import { BugReportService } from '../services/bugreport/BugReportService.js';

interface BugReportCommandProps {
    messages: any[];
    initialDescription?: string;
    onDone: (message: string, options?: { display: 'system' }) => void;
}

type BugReportState = 'userInput' | 'consent' | 'submitting' | 'done';

export const BugReportCommand: React.FC<BugReportCommandProps> = ({
    messages,
    initialDescription = '',
    onDone
}) => {
    const [state, setState] = useState<BugReportState>('userInput');
    const [description, setDescription] = useState(initialDescription);
    const [feedbackId, setFeedbackId] = useState<string | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [gitInfo, setGitInfo] = useState<{ isGit: boolean, state: any } | null>(null);

    useEffect(() => {
        async function fetchGit() {
            const isGit = await isGitRepo();
            let state = null;
            if (isGit) {
                state = await getGitState();
            }
            setGitInfo({ isGit, state });
        }
        fetchGit();
    }, []);

    const handleSubmit = useCallback(async () => {
        if (!description.trim()) {
            setError('Please provide a description of the issue.');
            return;
        }

        setState('submitting');
        setError(null);

        try {
            const report = {
                description,
                datetime: new Date().toISOString(),
                platform: platform(),
                terminal: terminal(),
                version: getProductVersion(),
                gitRepo: gitInfo?.isGit ?? false,
                gitState: gitInfo?.state ?? null,
                transcript: formatTranscript(messages),
                lastApiRequest: BugReportService.getLastApiRequest()
            };

            const result = await submitFeedback(report);
            if (result.success) {
                setFeedbackId(result.feedbackId ?? null);
                setState('done');
            } else {
                setError(result.error || 'Failed to submit feedback. Please try again.');
                setState('userInput');
            }
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
            setState('userInput');
        }
    }, [description, messages, gitInfo]);

    useInput((input, key) => {
        if (key.escape) {
            onDone('Bug report cancelled', { display: 'system' });
            return;
        }

        if (state === 'userInput' && key.return) {
            if (description.trim()) {
                setState('consent');
            } else {
                setError('Please provide a description.');
            }
            return;
        }

        if (state === 'consent' && key.return) {
            handleSubmit();
            return;
        }

        if (state === 'done' && key.return) {
            const githubUrl = `https://github.com/anthropics/claude-code/issues/new?title=${encodeURIComponent(`Bug Report: ${description.slice(0, 50)}...`)}&body=${encodeURIComponent(`## Description\n${description}\n\n## Environment\n- Version: ${getProductVersion()}\n- Platform: ${platform()}\n- Terminal: ${terminal()}\n${gitInfo?.state ? `\n## Git Info\n- Branch: ${gitInfo.state.branchName}\n- Commit: ${gitInfo.state.commitHash}\n` : ''}\n## Feedback ID\n${feedbackId ?? 'N/A'}`)}`;
            openUrl(githubUrl);
            onDone('Bug report submitted', { display: 'system' });
            return;
        }

        if (state === 'done') {
            onDone('Bug report submitted', { display: 'system' });
        }
    });

    return (
        <Box flexDirection="column" borderStyle="round" borderColor="cyan" paddingX={1} paddingBottom={1}>
            <Text bold color="cyan">Submit Feedback / Bug Report</Text>

            {state === 'userInput' && (
                <Box flexDirection="column" gap={1}>
                    <Text>Describe the issue below:</Text>
                    <Box borderStyle="single" paddingX={1}>
                        <TextInput
                            value={description}
                            onChange={(val) => {
                                setDescription(val);
                                setError(null);
                            }}
                        />
                    </Box>
                    {error && <Text color="red">{error}</Text>}
                    <Text dimColor>Press Enter to continue · Esc to cancel</Text>
                </Box>
            )}

            {state === 'consent' && (
                <Box flexDirection="column" gap={1}>
                    <Text>This report will include:</Text>
                    <Box marginLeft={2} flexDirection="column">
                        <Text>- Your feedback description</Text>
                        <Text>- Environment info ({platform()}, {terminal()}, v{getProductVersion()})</Text>
                        {gitInfo?.isGit && <Text>- Git repo metadata</Text>}
                        <Text>- Current session transcript</Text>
                    </Box>
                    <Box marginTop={1}>
                        <Text dimColor>
                            We will use your feedback to debug related issues or to improve Claude Code's functionality.
                        </Text>
                    </Box>
                    <Box marginTop={1}>
                        <Text bold>Press Enter to confirm and submit.</Text>
                    </Box>
                </Box>
            )}

            {state === 'submitting' && (
                <Box flexDirection="row" gap={1}>
                    <Text>Submitting report...</Text>
                </Box>
            )}

            {state === 'done' && (
                <Box flexDirection="column" gap={1}>
                    <Box>
                        <Text color="green">Thank you for your report!</Text>
                    </Box>
                    {feedbackId && (
                        <Box>
                            <Text dimColor>Feedback ID: {feedbackId}</Text>
                        </Box>
                    )}
                    <Box marginTop={1}>
                        <Text>
                            Press <Text bold>Enter</Text> to open your browser and draft a GitHub issue, or any other key to close.
                        </Text>
                    </Box>
                </Box>
            )}
        </Box>
    );
};
