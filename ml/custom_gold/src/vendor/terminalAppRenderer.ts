import { SHOW_CURSOR, HIDE_CURSOR } from "./terminalSequences.js";
import { createHyperlinkSequence } from "./terminalOscSequences.js";

export interface RenderCommand {
    type: string;
    content?: string;
    count?: number;
    x?: number;
    y?: number;
    uri?: string;
    codes?: number[];
    state?: any;
}

/**
 * Optimizes render commands by merging redundant ones.
 * Deobfuscated from xl1 in chunk_203.ts.
 */
export function optimizeRenderCommands(commands: RenderCommand[]): RenderCommand[] {
    if (commands.length <= 1) return commands;

    const optimized: RenderCommand[] = [];
    for (const cmd of commands) {
        if (isRedundant(cmd)) continue;
        if (tryMerge(optimized, cmd)) continue;
        optimized.push(cmd);
    }
    return optimized;
}

function isRedundant(cmd: RenderCommand): boolean {
    switch (cmd.type) {
        case "stdout": return cmd.content === "";
        case "cursorMove": return cmd.x === 0 && cmd.y === 0;
        case "clear": return (cmd.count ?? 0) === 0;
        default: return false;
    }
}

function tryMerge(list: RenderCommand[], next: RenderCommand): boolean {
    if (list.length === 0) return false;
    const last = list[list.length - 1];

    if (next.type === "cursorMove" && last.type === "cursorMove") {
        list[list.length - 1] = {
            type: "cursorMove",
            x: (last.x ?? 0) + (next.x ?? 0),
            y: (last.y ?? 0) + (next.y ?? 0)
        };
        return true;
    }

    if (next.type === "style" && last.type === "style") {
        list[list.length - 1] = next; // Overwrite style
        return true;
    }

    if (next.type === "hyperlink" && last.type === "hyperlink" && next.uri === last.uri) {
        return true; // Redundant hyperlink
    }

    if ((next.type === "cursorShow" && last.type === "cursorHide") ||
        (next.type === "cursorHide" && last.type === "cursorShow")) {
        list.pop(); // Cancel out
        return true;
    }

    return false;
}

/**
 * Final stage of rendering: applies commands to the terminal stream.
 * Deobfuscated from Sl1 in chunk_203.ts.
 */
export function applyRenderCommands(stdout: NodeJS.WriteStream, commands: RenderCommand[]): void {
    let buffer = "";
    // In chunk_203, there's a vWB (SYNCHRONIZED_UPDATE start)
    for (const cmd of commands) {
        switch (cmd.type) {
            case "stdout": buffer += cmd.content; break;
            case "clearTerminal": buffer += "\x1b[2J\x1b[3J\x1b[H"; break; // Simplified clear
            case "cursorHide": buffer += HIDE_CURSOR; break;
            case "cursorShow": buffer += SHOW_CURSOR; break;
            case "cursorMove":
                // Implementation of nJB (cursorMove sequence)
                buffer += `\x1b[${cmd.y};${cmd.x}H`;
                break;
            case "carriageReturn": buffer += "\r"; break;
            case "hyperlink": buffer += createHyperlinkSequence(cmd.uri!); break;
            // ... styles and other commands
        }
    }
    stdout.write(buffer);
}
