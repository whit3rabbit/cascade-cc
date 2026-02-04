/**
 * File: src/utils/terminal/terminalDetection.ts
 * Role: Detection of terminal environments and leader pane identification.
 */

import { executeBashCommand } from "../shared/bashUtils.js";
import { EnvService } from "../../services/config/EnvService.js";

/**
 * Returns the name of the detected terminal environment.
 * Based on environment variables and process checks.
 */
export function getTerminalName(): string | null {
    const env = process.env;

    // Detected via specific env vars
    if (env.CURSOR_TRACE_ID) return "cursor";
    if (env.VSCODE_GIT_ASKPASS_MAIN?.includes("/.cursor-server/")) return "cursor";
    if (env.VSCODE_GIT_ASKPASS_MAIN?.includes("/.windsurf-server/")) return "windsurf";

    const bundleId = env.__CFBundleIdentifier?.toLowerCase();
    if (bundleId?.includes("vscodium")) return "codium";
    if (bundleId?.includes("windsurf")) return "windsurf";
    if (bundleId?.includes("com.google.android.studio")) return "androidstudio";

    // JetBrains detection (partial port from reference)
    if (env.TERMINAL_EMULATOR === "JetBrains-JediTerm") {
        return "jetbrains"; // simplified
    }

    if (env.TERM === "xterm-ghostty") return "ghostty";
    if (env.TERM?.includes("kitty")) return "kitty";
    if (env.KITTY_WINDOW_ID) return "kitty";

    if (env.TERM_PROGRAM) {
        return env.TERM_PROGRAM;
    }

    if (env.TMUX) return "tmux";
    if (env.STY) return "screen";
    if (env.KONSOLE_VERSION) return "konsole";
    if (env.GNOME_TERMINAL_SERVICE) return "gnome-terminal";
    if (env.XTERM_VERSION) return "xterm";
    if (env.VTE_VERSION) return "vte-based";
    if (env.TERMINATOR_UUID) return "terminator";
    if (env.ALACRITTY_LOG) return "alacritty";
    if (env.TILIX_ID) return "tilix";
    if (env.WT_SESSION) return "windows-terminal"; // Windows Terminal

    // Check for iTerm2 specific session ID if not caught by TERM_PROGRAM
    if (env.ITERM_SESSION_ID) return "iterm2";

    if (env.SSH_CONNECTION || env.SSH_CLIENT || env.SSH_TTY) {
        return "ssh-session";
    }

    return null;
}

/**
 * Detects the "Leader Pane ID".
 * This is the ID of the pane where the assistant was started.
 * Used to route background command outputs back to the correct context.
 */
export async function getLeaderPaneId(): Promise<string | null> {
    // 1. Check for TMUX
    if (EnvService.get("TMUX")) {
        try {
            const res = await executeBashCommand("tmux display-message -p '#{pane_id}'");
            if (res.exitCode === 0) {
                return res.stdout.trim();
            }
        } catch (error) {
            // ignore error
        }
    }

    // 2. Check for iTerm2
    const itermSession = EnvService.get("ITERM_SESSION_ID");
    if (itermSession) {
        // ITERM_SESSION_ID is typically something like "w0t0p0:xxxxxx"
        // We effectively treat this as the pane ID for iTerm.
        return itermSession;
    }

    // 3. Fallback or other terminals?
    // For now, only TMUX and iTerm2 have clear "pane" concepts we integrate with.
    return null;
}

/**
 * Checks if the current environment supports window splitting.
 */
export function supportsSplitting(): boolean {
    const term = getTerminalName();
    return term === "tmux" || term === "iterm2" || term === "vscode" || term === "cursor";
}
