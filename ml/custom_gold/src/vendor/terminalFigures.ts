import * as process from "node:process";

/**
 * Checks if the current terminal supports Unicode.
 * Deobfuscated from il1 in chunk_205.ts.
 */
export function isUnicodeSupported(): boolean {
    const { env, platform } = process;
    const { TERM, TERM_PROGRAM } = env;

    if (platform !== "win32") return TERM !== "linux";

    return Boolean(env.WT_SESSION) ||
        Boolean(env.TERMINUS_SUBLIME) ||
        env.ConEmuTask === "{cmd::Cmder}" ||
        TERM_PROGRAM === "Terminus-Sublime" ||
        TERM_PROGRAM === "vscode" ||
        TERM === "xterm-256color" ||
        TERM === "alacritty" ||
        env.TERMINAL_EMULATOR === "JetBrains-JediTerm";
}

const mainSymbols = {
    tick: "✔",
    info: "ℹ",
    warning: "⚠",
    cross: "✘",
    squareSmall: "◻",
    squareSmallFilled: "◼",
    circle: "◯",
    circleFilled: "◉",
    radioOn: "◉",
    radioOff: "◯",
    checkboxOn: "☒",
    checkboxOff: "☐",
    pointer: "❯",
    triangleUpOutline: "△",
    triangleLeft: "◀",
    triangleRight: "▶",
    lozenge: "◆",
    lozengeOutline: "◇",
    hamburger: "☰",
    smiley: "㋡",
    star: "★",
    play: "▶",
    nodejs: "⬢",
    arrowUp: "↑",
    arrowDown: "↓",
    arrowLeft: "←",
    arrowRight: "→",
    circleEmpty: "◯",
    bullet: "•",
    ellipsis: "…"
};
const fallbackSymbols = {
    tick: "√",
    info: "i",
    warning: "‼",
    cross: "×",
    squareSmall: "□",
    squareSmallFilled: "■",
    circle: "( )",
    circleFilled: "(*)",
    radioOn: "(*)",
    radioOff: "( )",
    checkboxOn: "[×]",
    checkboxOff: "[ ]",
    pointer: ">",
    triangleUpOutline: "∆",
    triangleLeft: "◄",
    triangleRight: "►",
    lozenge: "♦",
    lozengeOutline: "◊",
    hamburger: "≡",
    smiley: "☺",
    star: "✶",
    play: "►",
    nodejs: "♦",
    arrowUp: "^",
    arrowDown: "v",
    arrowLeft: "<",
    arrowRight: ">",
    circleEmpty: "( )",
    bullet: "*",
    ellipsis: "..."
};

/**
 * Terminal symbols (figures) with automatic Unicode fallback.
 * Deobfuscated from G1 in chunk_205.ts.
 */
export const figures = isUnicodeSupported() ? mainSymbols : fallbackSymbols;
