/**
 * File: src/services/terminal/ShellSnapshotService.ts
 * Role: Captures shell configuration (aliases, functions, options) to preserve user environment.
 */

import { join } from 'node:path';
import { existsSync, mkdirSync } from 'node:fs';
import { homedir } from 'node:os';
import { spawnBashCommand } from '../../utils/shared/bashUtils.js';
import { getBaseConfigDir } from '../../utils/shared/runtimeAndEnv.js';

const SHELL_SNAPSHOT_TIMEOUT = 10000;

/**
 * Captures functions, options, and aliases from the user's shell.
 */
function generateSnapshotScript(shellPath: string, snapshotFile: string): string {
    const isZsh = shellPath.includes('zsh');
    const shellConfigFile = join(homedir(), isZsh ? '.zshrc' : '.bashrc');
    const sourceCommand = existsSync(shellConfigFile) ? `source "${shellConfigFile}" < /dev/null` : "# No config";

    let script = `
        SNAPSHOT_FILE="${snapshotFile}"
        ${sourceCommand}
        echo "# Snapshot" >| "$SNAPSHOT_FILE"
        echo "unalias -a 2>/dev/null || true" >> "$SNAPSHOT_FILE"
    `;

    if (isZsh) {
        script += `
            typeset -f > /dev/null 2>&1
            typeset +f | grep -vE '^(_|__)' | while read func; do
                typeset -f "$func" >> "$SNAPSHOT_FILE"
            done
            setopt | sed 's/^/setopt /' >> "$SNAPSHOT_FILE"
        `;
    } else {
        script += `
            declare -f > /dev/null 2>&1
            declare -F | cut -d' ' -f3 | grep -vE '^(_|__)' | while read func; do
                declare -f "$func" >> "$SNAPSHOT_FILE"
            done
            shopt -p >> "$SNAPSHOT_FILE"
            echo "shopt -s expand_aliases" >> "$SNAPSHOT_FILE"
        `;
    }

    script += `
        alias | sed 's/^alias //g' | sed 's/^/alias -- /' >> "$SNAPSHOT_FILE"
    `;

    return script;
}

/**
 * Creates a shell snapshot file and returns its path.
 */
export async function createShellSnapshot(shellPath: string): Promise<string | undefined> {
    const shellType = shellPath.includes('zsh') ? 'zsh' : 'bash';
    const snapshotsDir = join(getBaseConfigDir(), 'shell-snapshots');
    if (!existsSync(snapshotsDir)) mkdirSync(snapshotsDir, { recursive: true });

    const snapshotPath = join(snapshotsDir, `snapshot-${shellType}-${Date.now()}.sh`);
    const script = generateSnapshotScript(shellPath, snapshotPath);

    return new Promise((resolve) => {
        spawnBashCommand(shellPath, ["-c", script], {
            timeout: SHELL_SNAPSHOT_TIMEOUT,
            env: { ...process.env, CLAUDECODE: "1" }
        }, (error) => {
            if (error) {
                console.error(`[ShellSnapshot] Failed to create shell snapshot: ${error.message}`);
                resolve(undefined);
            } else if (existsSync(snapshotPath)) {
                resolve(snapshotPath);
            } else {
                resolve(undefined);
            }
        });
    });
}
