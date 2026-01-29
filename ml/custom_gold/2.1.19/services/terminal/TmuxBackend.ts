/**
 * File: src/services/terminal/TmuxBackend.ts
 * Role: tmux-backed swarm pane operations.
 */

import { executeBashCommand, BashResult } from "../../utils/shared/bashUtils.js";
import { HIDDEN_TMUX_SESSION, SWARM_SESSION_NAME, SWARM_WINDOW_NAME } from "./SwarmConstants.js";

export interface TmuxPaneInfo {
    paneId: string;
    isFirstTeammate: boolean;
}

export interface SwarmSessionInfo {
    windowTarget: string;
    paneId: string;
}

function shellEscape(value: string): string {
    if (/^[A-Za-z0-9_\-./:@]+$/.test(value)) return value;
    return JSON.stringify(value);
}

async function runTmuxCommand(args: string[], ignoreTmuxEnv = false): Promise<{ code: number; stdout: string; stderr: string }> {
    const command = ["tmux", ...args].map(shellEscape).join(" ");
    const result: BashResult = await executeBashCommand(command, {
        env: ignoreTmuxEnv ? { ...process.env, TMUX: "" } : process.env
    });
    return {
        code: result.exitCode,
        stdout: result.stdout ?? "",
        stderr: result.stderr ?? ""
    };
}

export class TmuxBackend {
    readonly type = "tmux";
    readonly displayName = "tmux";
    readonly supportsHideShow = true;

    async isAvailable(): Promise<boolean> {
        const res = await executeBashCommand("tmux -V");
        return res.exitCode === 0;
    }

    async isRunningInside(): Promise<boolean> {
        return Boolean(process.env.TMUX);
    }

    async sendCommandToPane(paneId: string, command: string, external = false): Promise<void> {
        const res = await runTmuxCommand(["send-keys", "-t", paneId, command, "Enter"], external);
        if (res.code !== 0) {
            throw new Error(`Failed to send command to pane ${paneId}: ${res.stderr}`);
        }
    }

    async setPaneBorderColor(paneId: string, color: string, external = false): Promise<void> {
        const colorValue = color || "default";
        await runTmuxCommand(["select-pane", "-t", paneId, "-P", `bg=default,fg=${colorValue}`], external);
        await runTmuxCommand(["set-option", "-p", "-t", paneId, "pane-border-style", `fg=${colorValue}`], external);
        await runTmuxCommand(["set-option", "-p", "-t", paneId, "pane-active-border-style", `fg=${colorValue}`], external);
    }

    async setPaneTitle(paneId: string, title: string, color: string, external = false): Promise<void> {
        const colorValue = color || "default";
        await runTmuxCommand(["select-pane", "-t", paneId, "-T", title], external);
        await runTmuxCommand([
            "set-option",
            "-p",
            "-t",
            paneId,
            "pane-border-format",
            `#[fg=${colorValue},bold] #{pane_title} #[default]`
        ], external);
    }

    async enablePaneBorderStatus(windowTarget?: string, external = false): Promise<void> {
        const target = windowTarget || (await this.getCurrentWindowTarget());
        if (!target) return;
        await runTmuxCommand(["set-option", "-w", "-t", target, "pane-border-status", "top"], external);
    }

    async rebalancePanes(windowTarget: string, external = false): Promise<void> {
        if (external) {
            await this.rebalancePanesTiled(windowTarget);
            return;
        }
        await this.rebalancePanesWithLeader(windowTarget);
    }

    async killPane(paneId: string, external = false): Promise<boolean> {
        const res = await runTmuxCommand(["kill-pane", "-t", paneId], external);
        return res.code === 0;
    }

    async hidePane(paneId: string, external = false): Promise<boolean> {
        const res = await runTmuxCommand(["new-session", "-d", "-s", HIDDEN_TMUX_SESSION], external);
        if (res.code !== 0) {
            return false;
        }
        const move = await runTmuxCommand(["break-pane", "-d", "-s", paneId, "-t", `${HIDDEN_TMUX_SESSION}:`], external);
        return move.code === 0;
    }

    async showPane(paneId: string, targetWindow: string, external = false): Promise<boolean> {
        const res = await runTmuxCommand(["join-pane", "-h", "-s", paneId, "-t", targetWindow], external);
        if (res.code !== 0) {
            return false;
        }
        await runTmuxCommand(["select-layout", "-t", targetWindow, "main-vertical"], external);
        const panes = await runTmuxCommand(["list-panes", "-t", targetWindow, "-F", "#{pane_id}"] , external);
        const firstPane = panes.stdout.trim().split("\n").filter(Boolean)[0];
        if (firstPane) {
            await runTmuxCommand(["resize-pane", "-t", firstPane, "-x", "30%"], external);
        }
        return true;
    }

    async getCurrentPaneId(): Promise<string | null> {
        const res = await runTmuxCommand(["display-message", "-p", "#{pane_id}"]);
        if (res.code !== 0) return null;
        return res.stdout.trim();
    }

    async getCurrentWindowTarget(): Promise<string | null> {
        const res = await runTmuxCommand(["display-message", "-p", "#{session_name}:#{window_index}"]);
        if (res.code !== 0) return null;
        return res.stdout.trim();
    }

    async getCurrentWindowPaneCount(windowTarget?: string, external = false): Promise<number | null> {
        const target = windowTarget || (await this.getCurrentWindowTarget());
        if (!target) return null;
        const res = await runTmuxCommand(["list-panes", "-t", target, "-F", "#{pane_id}"] , external);
        if (res.code !== 0) return null;
        return res.stdout.trim().split("\n").filter(Boolean).length;
    }

    async hasSessionInSwarm(sessionName: string): Promise<boolean> {
        const res = await runTmuxCommand(["has-session", "-t", sessionName], true);
        return res.code === 0;
    }

    async createExternalSwarmSession(): Promise<SwarmSessionInfo> {
        if (!(await this.hasSessionInSwarm(SWARM_SESSION_NAME))) {
            const create = await runTmuxCommand([
                "new-session",
                "-d",
                "-s",
                SWARM_SESSION_NAME,
                "-n",
                SWARM_WINDOW_NAME,
                "-P",
                "-F",
                "#{pane_id}"
            ], true);
            if (create.code !== 0) {
                throw new Error(`Failed to create swarm session: ${create.stderr || "Unknown error"}`);
            }
            const paneId = create.stdout.trim();
            return {
                windowTarget: `${SWARM_SESSION_NAME}:${SWARM_WINDOW_NAME}`,
                paneId
            };
        }

        const windows = await runTmuxCommand([
            "list-windows",
            "-t",
            SWARM_SESSION_NAME,
            "-F",
            "#{window_name}"
        ], true);
        const names = windows.stdout.trim().split("\n").filter(Boolean);
        const windowTarget = `${SWARM_SESSION_NAME}:${SWARM_WINDOW_NAME}`;
        if (names.includes(SWARM_WINDOW_NAME)) {
            const panes = await runTmuxCommand(["list-panes", "-t", windowTarget, "-F", "#{pane_id}"] , true);
            const paneId = panes.stdout.trim().split("\n").filter(Boolean)[0] || "";
            return { windowTarget, paneId };
        }

        const createWindow = await runTmuxCommand([
            "new-window",
            "-t",
            SWARM_SESSION_NAME,
            "-n",
            SWARM_WINDOW_NAME,
            "-P",
            "-F",
            "#{pane_id}"
        ], true);
        if (createWindow.code !== 0) {
            throw new Error(`Failed to create swarm-view window: ${createWindow.stderr || "Unknown error"}`);
        }
        return { windowTarget, paneId: createWindow.stdout.trim() };
    }

    async createTeammatePaneWithLeader(agentName: string, color: string): Promise<TmuxPaneInfo> {
        const currentPane = await this.getCurrentPaneId();
        const windowTarget = await this.getCurrentWindowTarget();
        if (!currentPane || !windowTarget) {
            throw new Error("Could not determine current tmux pane/window");
        }
        const paneCount = await this.getCurrentWindowPaneCount(windowTarget);
        if (paneCount === null) {
            throw new Error("Could not determine pane count for current window");
        }

        const isFirstTeammate = paneCount === 1;
        let result;
        if (isFirstTeammate) {
            result = await runTmuxCommand([
                "split-window",
                "-t",
                currentPane,
                "-h",
                "-p",
                "70",
                "-P",
                "-F",
                "#{pane_id}"
            ]);
        } else {
            const panes = await runTmuxCommand(["list-panes", "-t", windowTarget, "-F", "#{pane_id}"]);
            const secondaryPanes = panes.stdout.trim().split("\n").filter(Boolean).slice(1);
            const idx = Math.floor((secondaryPanes.length - 1) / 2);
            const targetPane = secondaryPanes[idx] || secondaryPanes[secondaryPanes.length - 1];
            const splitVertical = secondaryPanes.length % 2 === 1;
            result = await runTmuxCommand([
                "split-window",
                "-t",
                targetPane,
                splitVertical ? "-v" : "-h",
                "-P",
                "-F",
                "#{pane_id}"
            ]);
        }

        if (result.code !== 0) {
            throw new Error(`Failed to create teammate pane: ${result.stderr}`);
        }

        const paneId = result.stdout.trim();
        await this.setPaneBorderColor(paneId, color);
        await this.setPaneTitle(paneId, agentName, color);
        await this.rebalancePanesWithLeader(windowTarget);
        return { paneId, isFirstTeammate };
    }

    async createTeammatePaneExternal(agentName: string, color: string): Promise<TmuxPaneInfo> {
        const session = await this.createExternalSwarmSession();
        const paneCount = await this.getCurrentWindowPaneCount(session.windowTarget, true);
        if (paneCount === null) {
            throw new Error("Could not determine pane count for swarm window");
        }

        const isFirstTeammate = paneCount === 1;
        let paneId = session.paneId;

        if (isFirstTeammate) {
            await this.enablePaneBorderStatus(session.windowTarget, true);
        } else {
            const panes = await runTmuxCommand(["list-panes", "-t", session.windowTarget, "-F", "#{pane_id}"] , true);
            const ids = panes.stdout.trim().split("\n").filter(Boolean);
            const idx = Math.floor((ids.length - 1) / 2);
            const targetPane = ids[idx] || ids[ids.length - 1];
            const splitVertical = ids.length % 2 === 1;
            const split = await runTmuxCommand([
                "split-window",
                "-t",
                targetPane,
                splitVertical ? "-v" : "-h",
                "-P",
                "-F",
                "#{pane_id}"
            ], true);
            if (split.code !== 0) {
                throw new Error(`Failed to create teammate pane: ${split.stderr}`);
            }
            paneId = split.stdout.trim();
        }

        await this.setPaneBorderColor(paneId, color, true);
        await this.setPaneTitle(paneId, agentName, color, true);
        await this.rebalancePanesTiled(session.windowTarget);
        return { paneId, isFirstTeammate };
    }

    async createTeammatePaneInSwarmView(agentName: string, color: string): Promise<TmuxPaneInfo> {
        if (await this.isRunningInside()) {
            return this.createTeammatePaneWithLeader(agentName, color);
        }
        return this.createTeammatePaneExternal(agentName, color);
    }

    async rebalancePanesWithLeader(windowTarget: string): Promise<void> {
        const panes = await runTmuxCommand(["list-panes", "-t", windowTarget, "-F", "#{pane_id}"]);
        const paneIds = panes.stdout.trim().split("\n").filter(Boolean);
        if (paneIds.length === 0) return;
        await runTmuxCommand(["select-layout", "-t", windowTarget, "main-vertical"]);
        await runTmuxCommand(["resize-pane", "-t", paneIds[0], "-x", "30%"]);
    }

    async rebalancePanesTiled(windowTarget: string): Promise<void> {
        await runTmuxCommand(["select-layout", "-t", windowTarget, "tiled"], true);
    }
}
