/**
 * File: src/services/terminal/ThemeService.ts
 * Role: Manages TUI themes and persistence.
 */

import { useState, useEffect } from 'react';
import { getSettings, updateSettings } from '../config/SettingsService.js';

export type ThemeName = 'dark' | 'light' | 'light-daltonized' | 'dark-daltonized' | 'light-ansi' | 'dark-ansi';

export interface ThemeColors {
    autoAccept: string;
    bashBorder: string;
    claude: string;
    claudeShimmer: string;
    claudeBlue_FOR_SYSTEM_SPINNER: string;
    claudeBlueShimmer_FOR_SYSTEM_SPINNER: string;
    permission: string;
    permissionShimmer: string;
    planMode: string;
    delegateMode: string;
    ide: string;
    promptBorder: string;
    promptBorderShimmer: string;
    text: string;
    inverseText: string;
    inactive: string;
    subtle: string;
    suggestion: string;
    remember: string;
    background: string;
    success: string;
    error: string;
    warning: string;
    warningShimmer: string;
    diffAdded: string;
    diffRemoved: string;
    diffAddedDimmed: string;
    diffRemovedDimmed: string;
    diffAddedWord: string;
    diffRemovedWord: string;
    red_FOR_SUBAGENTS_ONLY: string;
    blue_FOR_SUBAGENTS_ONLY: string;
    green_FOR_SUBAGENTS_ONLY: string;
    yellow_FOR_SUBAGENTS_ONLY: string;
    purple_FOR_SUBAGENTS_ONLY: string;
    orange_FOR_SUBAGENTS_ONLY: string;
    pink_FOR_SUBAGENTS_ONLY: string;
    cyan_FOR_SUBAGENTS_ONLY: string;
    professionalBlue: string;
    chromeYellow: string;
    clawd_body: string;
    clawd_background: string;
    userMessageBackground: string;
    bashMessageBackgroundColor: string;
    memoryBackgroundColor: string;
    rate_limit_fill: string;
    rate_limit_empty: string;
}

const THEMES: Record<ThemeName, ThemeColors> = {
    dark: {
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
        clawd_body: "rgb(215,119,87)",
        clawd_background: "rgb(0,0,0)",
        userMessageBackground: "rgb(55, 55, 55)",
        bashMessageBackgroundColor: "rgb(65, 60, 65)",
        memoryBackgroundColor: "rgb(55, 65, 70)",
        rate_limit_fill: "rgb(177,185,249)",
        rate_limit_empty: "rgb(80,83,112)"
    },
    light: {
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
        clawd_body: "rgb(215,119,87)",
        clawd_background: "rgb(0,0,0)",
        userMessageBackground: "rgb(240, 240, 240)",
        bashMessageBackgroundColor: "rgb(250, 245, 250)",
        memoryBackgroundColor: "rgb(230, 245, 250)",
        rate_limit_fill: "rgb(87,105,247)",
        rate_limit_empty: "rgb(39,47,111)"
    },
    'light-ansi': {
        autoAccept: "ansi:magenta",
        bashBorder: "ansi:magenta",
        claude: "ansi:redBright",
        claudeShimmer: "ansi:yellowBright",
        claudeBlue_FOR_SYSTEM_SPINNER: "ansi:blue",
        claudeBlueShimmer_FOR_SYSTEM_SPINNER: "ansi:blueBright",
        permission: "ansi:blue",
        permissionShimmer: "ansi:blueBright",
        planMode: "ansi:cyan",
        delegateMode: "ansi:magenta",
        ide: "ansi:blueBright",
        promptBorder: "ansi:white",
        promptBorderShimmer: "ansi:whiteBright",
        text: "ansi:black",
        inverseText: "ansi:white",
        inactive: "ansi:blackBright",
        subtle: "ansi:blackBright",
        suggestion: "ansi:blue",
        remember: "ansi:blue",
        background: "ansi:cyan",
        success: "ansi:green",
        error: "ansi:red",
        warning: "ansi:yellow",
        warningShimmer: "ansi:yellowBright",
        diffAdded: "ansi:green",
        diffRemoved: "ansi:red",
        diffAddedDimmed: "ansi:green",
        diffRemovedDimmed: "ansi:red",
        diffAddedWord: "ansi:greenBright",
        diffRemovedWord: "ansi:redBright",
        red_FOR_SUBAGENTS_ONLY: "ansi:red",
        blue_FOR_SUBAGENTS_ONLY: "ansi:blue",
        green_FOR_SUBAGENTS_ONLY: "ansi:green",
        yellow_FOR_SUBAGENTS_ONLY: "ansi:yellow",
        purple_FOR_SUBAGENTS_ONLY: "ansi:magenta",
        orange_FOR_SUBAGENTS_ONLY: "ansi:redBright",
        pink_FOR_SUBAGENTS_ONLY: "ansi:magentaBright",
        cyan_FOR_SUBAGENTS_ONLY: "ansi:cyan",
        professionalBlue: "ansi:blueBright",
        chromeYellow: "ansi:yellow",
        clawd_body: "ansi:redBright",
        clawd_background: "ansi:black",
        userMessageBackground: "ansi:white",
        bashMessageBackgroundColor: "ansi:whiteBright",
        memoryBackgroundColor: "ansi:white",
        rate_limit_fill: "ansi:yellow",
        rate_limit_empty: "ansi:black"
    },
    'dark-ansi': {
        autoAccept: "ansi:magentaBright",
        bashBorder: "ansi:magentaBright",
        claude: "ansi:redBright",
        claudeShimmer: "ansi:yellowBright",
        claudeBlue_FOR_SYSTEM_SPINNER: "ansi:blueBright",
        claudeBlueShimmer_FOR_SYSTEM_SPINNER: "ansi:blueBright",
        permission: "ansi:blueBright",
        permissionShimmer: "ansi:blueBright",
        planMode: "ansi:cyanBright",
        delegateMode: "ansi:magentaBright",
        ide: "ansi:blue",
        promptBorder: "ansi:white",
        promptBorderShimmer: "ansi:whiteBright",
        text: "ansi:whiteBright",
        inverseText: "ansi:black",
        inactive: "ansi:white",
        subtle: "ansi:white",
        suggestion: "ansi:blueBright",
        remember: "ansi:blueBright",
        background: "ansi:cyanBright",
        success: "ansi:greenBright",
        error: "ansi:redBright",
        warning: "ansi:yellowBright",
        warningShimmer: "ansi:yellowBright",
        diffAdded: "ansi:green",
        diffRemoved: "ansi:red",
        diffAddedDimmed: "ansi:green",
        diffRemovedDimmed: "ansi:red",
        diffAddedWord: "ansi:greenBright",
        diffRemovedWord: "ansi:redBright",
        red_FOR_SUBAGENTS_ONLY: "ansi:redBright",
        blue_FOR_SUBAGENTS_ONLY: "ansi:blueBright",
        green_FOR_SUBAGENTS_ONLY: "ansi:greenBright",
        yellow_FOR_SUBAGENTS_ONLY: "ansi:yellowBright",
        purple_FOR_SUBAGENTS_ONLY: "ansi:magentaBright",
        orange_FOR_SUBAGENTS_ONLY: "ansi:redBright",
        pink_FOR_SUBAGENTS_ONLY: "ansi:magentaBright",
        cyan_FOR_SUBAGENTS_ONLY: "ansi:cyanBright",
        professionalBlue: "rgb(106,155,204)",
        chromeYellow: "ansi:yellowBright",
        clawd_body: "ansi:redBright",
        clawd_background: "ansi:black",
        userMessageBackground: "ansi:blackBright",
        bashMessageBackgroundColor: "ansi:black",
        memoryBackgroundColor: "ansi:blackBright",
        rate_limit_fill: "ansi:yellow",
        rate_limit_empty: "ansi:white"
    },
    'light-daltonized': {
        autoAccept: "rgb(135,0,255)",
        bashBorder: "rgb(0,102,204)",
        claude: "rgb(255,153,51)",
        claudeShimmer: "rgb(255,183,101)",
        claudeBlue_FOR_SYSTEM_SPINNER: "rgb(51,102,255)",
        claudeBlueShimmer_FOR_SYSTEM_SPINNER: "rgb(101,152,255)",
        permission: "rgb(51,102,255)",
        permissionShimmer: "rgb(101,152,255)",
        planMode: "rgb(51,102,102)",
        delegateMode: "rgb(138,43,226)",
        ide: "rgb(71,130,200)",
        promptBorder: "rgb(153,153,153)",
        promptBorderShimmer: "rgb(183,183,183)",
        text: "rgb(0,0,0)",
        inverseText: "rgb(255,255,255)",
        inactive: "rgb(102,102,102)",
        subtle: "rgb(175,175,175)",
        suggestion: "rgb(51,102,255)",
        remember: "rgb(51,102,255)",
        background: "rgb(0,153,153)",
        success: "rgb(0,102,153)",
        error: "rgb(204,0,0)",
        warning: "rgb(255,153,0)",
        warningShimmer: "rgb(255,183,50)",
        diffAdded: "rgb(153,204,255)",
        diffRemoved: "rgb(255,204,204)",
        diffAddedDimmed: "rgb(209,231,253)",
        diffRemovedDimmed: "rgb(255,233,233)",
        diffAddedWord: "rgb(51,102,204)",
        diffRemovedWord: "rgb(153,51,51)",
        red_FOR_SUBAGENTS_ONLY: "rgb(204,0,0)",
        blue_FOR_SUBAGENTS_ONLY: "rgb(0,102,204)",
        green_FOR_SUBAGENTS_ONLY: "rgb(0,204,0)",
        yellow_FOR_SUBAGENTS_ONLY: "rgb(255,204,0)",
        purple_FOR_SUBAGENTS_ONLY: "rgb(128,0,128)",
        orange_FOR_SUBAGENTS_ONLY: "rgb(255,128,0)",
        pink_FOR_SUBAGENTS_ONLY: "rgb(255,102,178)",
        cyan_FOR_SUBAGENTS_ONLY: "rgb(0,178,178)",
        professionalBlue: "rgb(106,155,204)",
        chromeYellow: "rgb(251,188,4)",
        clawd_body: "rgb(215,119,87)",
        clawd_background: "rgb(0,0,0)",
        userMessageBackground: "rgb(220, 220, 220)",
        bashMessageBackgroundColor: "rgb(250, 245, 250)",
        memoryBackgroundColor: "rgb(230, 245, 250)",
        rate_limit_fill: "rgb(51,102,255)",
        rate_limit_empty: "rgb(23,46,114)"
    },
    'dark-daltonized': {
        autoAccept: "rgb(175,135,255)",
        bashBorder: "rgb(51,153,255)",
        claude: "rgb(255,153,51)",
        claudeShimmer: "rgb(255,183,101)",
        claudeBlue_FOR_SYSTEM_SPINNER: "rgb(153,204,255)",
        claudeBlueShimmer_FOR_SYSTEM_SPINNER: "rgb(183,224,255)",
        permission: "rgb(153,204,255)",
        permissionShimmer: "rgb(183,224,255)",
        planMode: "rgb(102,153,153)",
        delegateMode: "rgb(186,85,255)",
        ide: "rgb(71,130,200)",
        promptBorder: "rgb(136,136,136)",
        promptBorderShimmer: "rgb(166,166,166)",
        text: "rgb(255,255,255)",
        inverseText: "rgb(0,0,0)",
        inactive: "rgb(153,153,153)",
        subtle: "rgb(80,80,80)",
        suggestion: "rgb(153,204,255)",
        remember: "rgb(153,204,255)",
        background: "rgb(0,204,204)",
        success: "rgb(51,153,255)",
        error: "rgb(255,102,102)",
        warning: "rgb(255,204,0)",
        warningShimmer: "rgb(255,234,50)",
        diffAdded: "rgb(0,68,102)",
        diffRemoved: "rgb(102,0,0)",
        diffAddedDimmed: "rgb(62,81,91)",
        diffRemovedDimmed: "rgb(62,44,44)",
        diffAddedWord: "rgb(0,119,179)",
        diffRemovedWord: "rgb(179,0,0)",
        red_FOR_SUBAGENTS_ONLY: "rgb(255,102,102)",
        blue_FOR_SUBAGENTS_ONLY: "rgb(102,178,255)",
        green_FOR_SUBAGENTS_ONLY: "rgb(102,255,102)",
        yellow_FOR_SUBAGENTS_ONLY: "rgb(255,255,102)",
        purple_FOR_SUBAGENTS_ONLY: "rgb(178,102,255)",
        orange_FOR_SUBAGENTS_ONLY: "rgb(255,178,102)",
        pink_FOR_SUBAGENTS_ONLY: "rgb(255,153,204)",
        cyan_FOR_SUBAGENTS_ONLY: "rgb(102,204,204)",
        professionalBlue: "rgb(106,155,204)",
        chromeYellow: "rgb(251,188,4)",
        clawd_body: "rgb(215,119,87)",
        clawd_background: "rgb(0,0,0)",
        userMessageBackground: "rgb(55, 55, 55)",
        bashMessageBackgroundColor: "rgb(65, 60, 65)",
        memoryBackgroundColor: "rgb(55, 65, 70)",
        rate_limit_fill: "rgb(153,204,255)",
        rate_limit_empty: "rgb(69,92,115)"
    }
};

class ThemeService {
    private currentTheme: ThemeName = 'dark';
    private listeners: Set<(theme: ThemeColors) => void> = new Set();

    constructor() {
        const settings = getSettings();
        if (settings.theme && THEMES[settings.theme as ThemeName]) {
            this.currentTheme = settings.theme as ThemeName;
        }
    }

    getThemeColors(): ThemeColors {
        return THEMES[this.currentTheme];
    }

    getThemeName(): ThemeName {
        return this.currentTheme;
    }

    setTheme(name: ThemeName) {
        if (THEMES[name]) {
            this.currentTheme = name;
            updateSettings({ theme: name });
            this.notify();
        }
    }

    subscribe(listener: (theme: ThemeColors) => void) {
        this.listeners.add(listener);
        listener(this.getThemeColors());
        return () => this.listeners.delete(listener);
    }

    private notify() {
        const colors = this.getThemeColors();
        this.listeners.forEach(l => l(colors));
    }
}

export const themeService = new ThemeService();

export function useTheme() {
    const [colors, setColors] = useState<ThemeColors>(() => themeService.getThemeColors());

    useEffect(() => {
        const unsubscribe = themeService.subscribe((newColors) => {
            setColors(newColors);
        });
        return () => { unsubscribe(); };
    }, []);


    return colors;
}
