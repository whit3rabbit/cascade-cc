
import { existsSync, readFileSync, writeFileSync } from 'fs';
import { homedir } from 'os';
import { join } from 'path';

const CLAUDE_ALIAS_REGEX = /^\s*alias\s+claude\s*=/;

export function getShellConfigFiles() {
    const zshDir = process.env.ZDOTDIR || homedir();
    return {
        zsh: join(zshDir, ".zshrc"),
        bash: join(homedir(), ".bashrc"),
        fish: join(homedir(), ".config/fish/config.fish")
    };
}

export function detectClaudeAlias(): string | null {
    const configs = getShellConfigFiles();
    for (const path of Object.values(configs)) {
        if (!existsSync(path)) continue;
        try {
            const content = readFileSync(path, 'utf8');
            const lines = content.split('\n');
            for (const line of lines) {
                if (CLAUDE_ALIAS_REGEX.test(line)) {
                    const match = line.match(/alias\s+claude=["']?([^"'\s]+)/);
                    if (match && match[1]) return match[1];
                }
            }
        } catch { }
    }
    return null;
}

export function resolveClaudeAliasPath(): string | null {
    const alias = detectClaudeAlias();
    if (!alias) return null;

    const resolvedPath = alias.startsWith('~') ? alias.replace('~', homedir()) : alias;
    if (existsSync(resolvedPath)) return alias;

    return null;
}
