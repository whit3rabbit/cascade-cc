import chalk from "chalk";

export type Theme = typeof darkTheme;

/**
 * Theme definitions for different terminal styles.
 * Deobfuscated from fs8, hs8, gs8, us8, ms8, ds8 in chunk_198.ts.
 */

export const lightTheme = {
    autoAccept: "rgb(135,0,255)",
    bashBorder: "rgb(255,0,135)",
    claude: "rgb(215,119,87)",
    claudeShimmer: "rgb(245,149,117)",
    claudeBlue_FOR_SYSTEM_SPINNER: "rgb(87,105,247)",
    claudeBlueShimmer_FOR_SYSTEM_SPINNER: "rgb(117,135,255)",
    permission: "rgb(87,105,247)",
    permissionShimmer: "rgb(137,155,255)",
    planMode: "rgb(0,102,102)",
    delegateMode: "rgb(138,43,226)",
    ide: "rgb(71,130,200)",
    promptBorder: "rgb(153,153,153)",
    promptBorderShimmer: "rgb(183,183,183)",
    text: "rgb(0,0,0)",
    inverseText: "rgb(255,255,255)",
    inactive: "rgb(102,102,102)",
    subtle: "rgb(175,175,175)",
    suggestion: "rgb(87,105,247)",
    remember: "rgb(0,0,255)",
    background: "rgb(0,153,153)",
    success: "rgb(44,122,57)",
    error: "rgb(171,43,63)",
    warning: "rgb(150,108,30)",
    warningShimmer: "rgb(200,158,80)",
    diffAdded: "rgb(105,219,124)",
    diffRemoved: "rgb(255,168,180)",
    diffAddedDimmed: "rgb(199,225,203)",
    diffRemovedDimmed: "rgb(253,210,216)",
    diffAddedWord: "rgb(47,157,68)",
    diffRemovedWord: "rgb(209,69,75)",
    red_FOR_SUBAGENTS_ONLY: "rgb(220,38,38)",
    blue_FOR_SUBAGENTS_ONLY: "rgb(37,99,235)",
    green_FOR_SUBAGENTS_ONLY: "rgb(22,163,74)",
    yellow_FOR_SUBAGENTS_ONLY: "rgb(202,138,4)",
    purple_FOR_SUBAGENTS_ONLY: "rgb(147,51,234)",
    orange_FOR_SUBAGENTS_ONLY: "rgb(234,88,12)",
    pink_FOR_SUBAGENTS_ONLY: "rgb(219,39,119)",
    cyan_FOR_SUBAGENTS_ONLY: "rgb(8,145,178)",
    professionalBlue: "rgb(106,155,204)",
    chromeYellow: "rgb(251,188,4)",
    rainbow_red: "rgb(235,95,87)",
    rainbow_orange: "rgb(245,139,87)",
    rainbow_yellow: "rgb(250,195,95)",
    rainbow_green: "rgb(145,200,130)",
    rainbow_blue: "rgb(130,170,220)",
    rainbow_indigo: "rgb(155,130,200)",
    rainbow_violet: "rgb(200,130,180)",
    rainbow_red_shimmer: "rgb(250,155,147)",
    rainbow_orange_shimmer: "rgb(255,185,137)",
    rainbow_yellow_shimmer: "rgb(255,225,155)",
    rainbow_green_shimmer: "rgb(185,230,180)",
    rainbow_blue_shimmer: "rgb(180,205,240)",
    rainbow_indigo_shimmer: "rgb(195,180,230)",
    rainbow_violet_shimmer: "rgb(230,180,210)",
    clawd_body: "rgb(215,119,87)",
    clawd_background: "rgb(0,0,0)",
    ice_blue: "rgb(173,216,230)",
    userMessageBackground: "rgb(240, 240, 240)",
    bashMessageBackgroundColor: "rgb(250, 245, 250)",
    memoryBackgroundColor: "rgb(230, 245, 250)",
    rate_limit_fill: "rgb(87,105,247)",
    rate_limit_empty: "rgb(39,47,111)"
};

export const darkTheme = {
    autoAccept: "rgb(175,135,255)",
    bashBorder: "rgb(253,93,177)",
    claude: "rgb(215,119,87)",
    claudeShimmer: "rgb(235,159,127)",
    claudeBlue_FOR_SYSTEM_SPINNER: "rgb(147,165,255)",
    claudeBlueShimmer_FOR_SYSTEM_SPINNER: "rgb(177,195,255)",
    permission: "rgb(177,185,249)",
    permissionShimmer: "rgb(207,215,255)",
    planMode: "rgb(72,150,140)",
    delegateMode: "rgb(186,85,255)",
    ide: "rgb(71,130,200)",
    promptBorder: "rgb(136,136,136)",
    promptBorderShimmer: "rgb(166,166,166)",
    text: "rgb(255,255,255)",
    inverseText: "rgb(0,0,0)",
    inactive: "rgb(153,153,153)",
    subtle: "rgb(80,80,80)",
    suggestion: "rgb(177,185,249)",
    remember: "rgb(177,185,249)",
    background: "rgb(0,204,204)",
    success: "rgb(78,186,101)",
    error: "rgb(255,107,128)",
    warning: "rgb(255,193,7)",
    warningShimmer: "rgb(255,223,57)",
    diffAdded: "rgb(34,92,43)",
    diffRemoved: "rgb(122,41,54)",
    diffAddedDimmed: "rgb(71,88,74)",
    diffRemovedDimmed: "rgb(105,72,77)",
    diffAddedWord: "rgb(56,166,96)",
    diffRemovedWord: "rgb(179,89,107)",
    red_FOR_SUBAGENTS_ONLY: "rgb(220,38,38)",
    blue_FOR_SUBAGENTS_ONLY: "rgb(37,99,235)",
    green_FOR_SUBAGENTS_ONLY: "rgb(22,163,74)",
    yellow_FOR_SUBAGENTS_ONLY: "rgb(202,138,4)",
    purple_FOR_SUBAGENTS_ONLY: "rgb(147,51,234)",
    orange_FOR_SUBAGENTS_ONLY: "rgb(234,88,12)",
    pink_FOR_SUBAGENTS_ONLY: "rgb(219,39,119)",
    cyan_FOR_SUBAGENTS_ONLY: "rgb(8,145,178)",
    professionalBlue: "rgb(106,155,204)",
    chromeYellow: "rgb(251,188,4)",
    rainbow_red: "rgb(235,95,87)",
    rainbow_orange: "rgb(245,139,87)",
    rainbow_yellow: "rgb(250,195,95)",
    rainbow_green: "rgb(145,200,130)",
    rainbow_blue: "rgb(130,170,220)",
    rainbow_indigo: "rgb(155,130,200)",
    rainbow_violet: "rgb(200,130,180)",
    rainbow_red_shimmer: "rgb(250,155,147)",
    rainbow_orange_shimmer: "rgb(255,185,137)",
    rainbow_yellow_shimmer: "rgb(255,225,155)",
    rainbow_green_shimmer: "rgb(185,230,180)",
    rainbow_blue_shimmer: "rgb(180,205,240)",
    rainbow_indigo_shimmer: "rgb(195,180,230)",
    rainbow_violet_shimmer: "rgb(230,180,210)",
    clawd_body: "rgb(215,119,87)",
    clawd_background: "rgb(0,0,0)",
    ice_blue: "rgb(173,216,230)",
    userMessageBackground: "rgb(55, 55, 55)",
    bashMessageBackgroundColor: "rgb(65, 60, 65)",
    memoryBackgroundColor: "rgb(55, 65, 70)",
    rate_limit_fill: "rgb(177,185,249)",
    rate_limit_empty: "rgb(80,83,112)"
};

export const themes: Record<string, Theme> = {
    light: lightTheme,
    dark: darkTheme
};

/**
 * Returns the theme object by style name.
 */
export function getThemeByStyle(style: string): any {
    switch (style) {
        case "light": return lightTheme;
        case "dark": return darkTheme;
        // ... add cases for light-ansi, dark-ansi, etc. if needed
        default: return darkTheme;
    }
}

/**
 * Resolves a color name or value to its theme-specific string value.
 * Deobfuscated from Lt8 in chunk_202.ts.
 */
export function getColorValue(color: string | undefined, theme: Theme): string | undefined {
    if (!color) return undefined;
    if (color.startsWith("rgb(") || color.startsWith("#") || color.startsWith("ansi256(") || color.startsWith("ansi:")) {
        return color;
    }
    return (theme as any)[color];
}

/**
 * Resolves a color string to a chalk function.
 * Deobfuscated from bNA in chunk_198.ts.
 */
export function resolveColor(color: string, mode: "foreground" | "background" = "foreground"): any {
    if (!color) return (s: string) => s;

    if (color.startsWith("ansi:")) {
        const ansi = color.substring(5);
        const method = mode === "background" ? `bg${ansi[0].toUpperCase()}${ansi.substring(1)}` : ansi;
        return (chalk as any)[method] || ((s: string) => s);
    }

    if (color.startsWith("#")) {
        return mode === "foreground" ? chalk.hex(color) : chalk.bgHex(color);
    }

    if (color.startsWith("rgb(")) {
        const match = color.match(/rgb\(\s?(\d+),\s?(\d+),\s?(\d+)\s?\)/);
        if (match) {
            const [r, g, b] = match.slice(1).map(Number);
            return mode === "foreground" ? chalk.rgb(r, g, b) : chalk.bgRgb(r, g, b);
        }
    }

    return (s: string) => s;
}

/**
 * Applies multiple text styles to a string.
 * Deobfuscated from fNA in chunk_198.ts.
 */
export function applyStyles(text: string, styles: any): string {
    let result = text;
    if (styles.bold) result = chalk.bold(result);
    if (styles.dim) result = chalk.dim(result);
    if (styles.italic) result = chalk.italic(result);
    if (styles.underline) result = chalk.underline(result);
    if (styles.strikethrough) result = chalk.strikethrough(result);
    if (styles.inverse) result = chalk.inverse(result);

    if (styles.color) {
        result = resolveColor(styles.color, "foreground")(result);
    }
    if (styles.backgroundColor) {
        result = resolveColor(styles.backgroundColor, "background")(result);
    }

    return result;
}

/**
 * Creates a higher-order colorizer function.
 * Deobfuscated from sQ in chunk_198.ts.
 */
export function createColorizer(color: string, themeStyle: string, mode: "foreground" | "background" = "foreground") {
    return (text: string) => {
        if (!color) return text;
        if (color.startsWith("rgb(") || color.startsWith("#") || color.startsWith("ansi256(") || color.startsWith("ansi:")) {
            return resolveColor(color, mode)(text);
        }
        const theme = getThemeByStyle(themeStyle);
        return resolveColor(theme[color], mode)(text);
    };
}
