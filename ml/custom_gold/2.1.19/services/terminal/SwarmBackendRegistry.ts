/**
 * File: src/services/terminal/SwarmBackendRegistry.ts
 * Role: Selects the swarm backend based on environment (tmux/iTerm2).
 */

import { TmuxBackend } from "./TmuxBackend.js";
import { isIt2Installed, isTmuxPreferredOverIterm2 } from "../../utils/terminal/iterm2Setup.js";

export interface SwarmBackendInfo {
    backend: TmuxBackend;
    isNative: boolean;
    needsIt2Setup: boolean;
}

let cachedBackend: SwarmBackendInfo | null = null;

function isInsideTmux(): boolean {
    return Boolean(process.env.TMUX);
}

function isInsideIterm2(): boolean {
    return Boolean(process.env.ITERM_SESSION_ID) ||
        (process.env.TERM_PROGRAM || "").toLowerCase().includes("iterm");
}

function isWsl(): boolean {
    return Boolean(process.env.WSL_DISTRO_NAME || process.env.WSL_INTEROP);
}

export function getSwarmSetupErrorMessage(): string {
    switch (true) {
        case process.platform === "darwin":
            return `To use agent swarms, install tmux:\n  brew install tmux\nThen start a tmux session with: tmux new-session -s claude`;
        case process.platform === "linux" && isWsl():
            return `To use agent swarms, you need tmux which requires WSL (Windows Subsystem for Linux).\nInstall WSL first, then inside WSL run:\n  sudo apt install tmux\nThen start a tmux session with: tmux new-session -s claude`;
        case process.platform === "linux":
            return `To use agent swarms, install tmux:\n  sudo apt install tmux    # Ubuntu/Debian\n  sudo dnf install tmux    # Fedora/RHEL\nThen start a tmux session with: tmux new-session -s claude`;
        case process.platform === "win32":
            return `To use agent swarms, you need tmux which requires WSL (Windows Subsystem for Linux).\nInstall WSL first, then inside WSL run:\n  sudo apt install tmux\nThen start a tmux session with: tmux new-session -s claude`;
        default:
            return `To use agent swarms, install tmux using your system's package manager.\nThen start a tmux session with: tmux new-session -s claude`;
    }
}

export async function detectSwarmBackend(): Promise<SwarmBackendInfo> {
    if (cachedBackend) return cachedBackend;

    const tmuxBackend = new TmuxBackend();
    const insideTmux = await tmuxBackend.isRunningInside();
    const inIterm2 = isInsideIterm2();

    if (insideTmux) {
        cachedBackend = { backend: tmuxBackend, isNative: true, needsIt2Setup: false };
        return cachedBackend;
    }

    if (inIterm2) {
        if (!isTmuxPreferredOverIterm2()) {
            const it2Available = await isIt2Installed();
            if (!it2Available) {
                const tmuxAvailable = await tmuxBackend.isAvailable();
                if (tmuxAvailable) {
                    cachedBackend = { backend: tmuxBackend, isNative: false, needsIt2Setup: true };
                    return cachedBackend;
                }
                throw new Error("iTerm2 detected but it2 CLI not installed. Install it2 with: pip install it2");
            }
        }

        const tmuxAvailable = await tmuxBackend.isAvailable();
        if (!tmuxAvailable) {
            throw new Error(getSwarmSetupErrorMessage());
        }
        cachedBackend = { backend: tmuxBackend, isNative: false, needsIt2Setup: false };
        return cachedBackend;
    }

    const tmuxAvailable = await tmuxBackend.isAvailable();
    if (!tmuxAvailable) {
        throw new Error(getSwarmSetupErrorMessage());
    }

    cachedBackend = { backend: tmuxBackend, isNative: false, needsIt2Setup: false };
    return cachedBackend;
}

export function resetSwarmBackendDetection(): void {
    cachedBackend = null;
}
