/**
 * File: src/constants/theme.ts
 * Role: Central theme definitions for the terminal UI.
 */

import { themeService } from '../services/terminal/ThemeService.js';

export const THEME = {
    get colors() {
        return themeService.getThemeColors();
    }
};

