
import React from 'react';
import { render } from 'ink';
import { TerminalShell } from '../components/terminal/TerminalShell.js';
import { cliMain } from './main.js';

/**
 * Main entry point for the Claude Code CLI.
 */
async function run() {
    // 1. Handle command line arguments (e.g., --version, --chrome-native-host)
    // main.ts handles the subset of flags that exit early or start background processes.
    const result = await cliMain();

    // 2. Clear the screen or perform other terminal setup if needed
    // process.stdout.write('\x1Bc');

    // 3. Render the interactive TUI
    const { AppStateProvider, getAppState } = await import('../contexts/AppStateContext.js');
    render(
        <AppStateProvider initialState={getAppState()}>
            <TerminalShell initialPrompt={result?.prompt} initialMessages={result?.initialMessages} options={result?.options} />
        </AppStateProvider>
    );
}

run().catch((err) => {
    console.error('Fatal error during CLI startup:', err);
    process.exit(1);
});
