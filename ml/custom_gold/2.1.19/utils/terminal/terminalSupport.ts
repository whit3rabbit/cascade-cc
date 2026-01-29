/**
 * File: src/utils/terminal/terminalSupport.ts
 * Role: Detection of terminal emulator features and common UI symbols.
 */

/**
 * Checks if the current terminal environment supports advanced features.
 */
export function isTerminalSupported(): boolean {
    const { env, platform } = process;
    const { TERM, TERM_PROGRAM } = env;

    if (platform !== "win32") {
        return TERM !== "linux";
    }

    // Windows-specific checks
    return (
        Boolean(env.WT_SESSION) ||
        Boolean(env.TERMINUS_SUBLIME) ||
        env.ConEmuTask === "{cmd::Cmder}" ||
        TERM_PROGRAM === "Terminus-Sublime" ||
        TERM_PROGRAM === "vscode" ||
        TERM === "xterm-256color" ||
        TERM === "alacritty" ||
        TERM === "rxvt-unicode" ||
        TERM === "rxvt-unicode-256color" ||
        env.TERMINAL_EMULATOR === "JetBrains-JediTerm"
    );
}

export const unicodeSymbols = {
    tick: "✔",
    info: "ℹ",
    warning: "⚠",
    cross: "✘",
    circle: "◯",
    circleFilled: "◉",
    radioOn: "◉",
    radioOff: "◯",
    checkboxOn: "☒",
    checkboxOff: "☐",
    pointer: "❯",
    star: "★",
    play: "▶",
    ellipsis: "…",
    line: "─",
    lineBold: "━",
    lineVertical: "│",
    lineVerticalBold: "┃"
};

export const asciiSymbols = {
    tick: "√",
    info: "i",
    warning: "‼",
    cross: "×",
    circle: "( )",
    circleFilled: "(*)",
    radioOn: "(*)",
    radioOff: "( )",
    checkboxOn: "[×]",
    checkboxOff: "[ ]",
    pointer: ">",
    star: "✶",
    play: "►",
    ellipsis: "...",
    line: "-",
    lineBold: "=",
    lineVertical: "|",
    lineVerticalBold: "||"
};

const isSupported = isTerminalSupported();
export const symbols = isSupported ? unicodeSymbols : asciiSymbols;
