/**
 * File: src/utils/terminal/bashHistory.ts
 * Role: Provides utilities for reading and parsing shell history files (bash/zsh).
 */

import { readFileSync, existsSync } from 'node:fs';
import { join } from 'node:path';
import { homedir } from 'node:os';

/**
 * Retrieves a list of commands from common shell history files.
 * Supports .bash_history and .zsh_history.
 */
export async function getBashHistory(): Promise<string[]> {
    const historyFiles = [
        join(homedir(), '.zsh_history'),
        join(homedir(), '.bash_history')
    ];

    const commandsSet = new Set<string>();

    for (const file of historyFiles) {
        if (!existsSync(file)) continue;

        try {
            const content = readFileSync(file, 'utf8');
            const lines = content.split('\n');

            for (const line of lines) {
                if (!line.trim()) continue;

                let cmd = line;
                // Zsh history format: ": 1234567890:0;command"
                if (file.endsWith('.zsh_history') && line.startsWith(': ')) {
                    const semicolonIndex = line.indexOf(';');
                    if (semicolonIndex !== -1) {
                        cmd = line.substring(semicolonIndex + 1);
                    }
                }

                const trimmed = cmd.trim();
                if (trimmed && trimmed.length > 2) {
                    commandsSet.add(trimmed);
                }
            }
        } catch {
            // Silently fail for individual files
        }
    }

    // Convert to array and return the last 1000 items (most recent)
    return Array.from(commandsSet).slice(-1000).reverse();
}
