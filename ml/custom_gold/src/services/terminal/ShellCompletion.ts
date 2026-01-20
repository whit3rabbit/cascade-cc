
import { runBashCommand } from './BashExecutor.js';

// Based on chunk_496.ts:332-457

export interface ShellCompletion {
    id: string;
    displayText: string;
    description?: string;
    metadata?: any;
}

export async function getShellCompletions(input: string, cursorPosition: number, shellType: 'bash' | 'zsh' = 'bash'): Promise<ShellCompletion[]> {
    const prefix = input.substring(0, cursorPosition);
    // Rough logic to extract the word being completed
    const match = prefix.match(/\S+$/);
    const word = match ? match[0] : "";

    if (!word && !input.endsWith(" ")) return [];

    let command: string;
    if (shellType === 'bash') {
        if (word.startsWith('$')) {
            command = `compgen -v ${word.slice(1)}`;
        } else if (word.includes('/') || word.startsWith('.') || word.startsWith('~')) {
            command = `compgen -f ${word}`;
        } else {
            command = `compgen -c ${word}`;
        }
    } else {
        // Zsh simplified logic
        command = `print -rl -- ${(word || '')}*(N)`;
    }

    try {
        const bashExecution = await runBashCommand(command);
        const result = await (bashExecution.result as Promise<any>);
        if (result.code !== 0) return [];

        return result.stdout.split('\n')
            .filter((line: string) => line.trim())
            .slice(0, 15)
            .map((line: string) => ({
                id: line,
                displayText: line,
                metadata: { completionType: word.startsWith('$') ? 'variable' : 'file' }
            }));
    } catch {
        return [];
    }
}
