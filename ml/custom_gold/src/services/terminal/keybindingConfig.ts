import * as os from "node:os";
import semver from "semver";

/**
 * Centralized keybinding configuration.
 * Deobfuscated from Sg in chunk_216.ts.
 */

const platform = os.platform();
const isMacOS = platform === "darwin";
const isWindows = platform === "win32";

// Meta key text (Opt on macOS, Alt elsewhere)
export const META_KEY_TEXT = isMacOS ? "opt" : "alt";

// Check if current runtime supports advanced keybindings
// Node 22.17+ (excluding 23.x) or Bun 1.2.23+ required for some features
export const SUPPORTS_KEYBINDINGS = isWindows
    ? ((process.versions as any).bun
        ? semver.satisfies((process.versions as any).bun, ">=1.2.23")
        : semver.satisfies(process.versions.node, ">=22.17.0 <23.0.0 || >=24.2.0"))
    : true;

// Key combo definitions
export const KEY_COMBOS = {
    // Move focus (Shift+Tab preferred if supported, otherwise Alt+M)
    MOVE_FOCUS: !SUPPORTS_KEYBINDINGS ? {
        displayText: `${META_KEY_TEXT}+m`,
        check: (input: string, key: any) => key.meta && (input === "m" || input === "M")
    } : {
        displayText: "shift+tab",
        check: (input: string, key: any) => key.tab && key.shift
    },

    // Paste (Ctrl+V on Linux/macOS, Alt+V on Windows)
    PASTE: isWindows ? {
        displayText: `${META_KEY_TEXT}+v`,
        check: (input: string, key: any) => key.meta && (input === "v" || input === "V")
    } : {
        displayText: "ctrl+v",
        check: (input: string, key: any) => key.ctrl && (input === "v" || input === "V")
    },

    // Model picker
    MODEL_PICKER: {
        displayText: `${META_KEY_TEXT}+p`,
        check: (input: string, key: any) => key.meta && (input === "p" || input === "P")
    },

    // Thinking toggle
    THINKING_TOGGLE: {
        displayText: `${META_KEY_TEXT}+t`,
        check: (input: string, key: any) => key.meta && (input === "t" || input === "T")
    }
};

// Fallback mappings for certain terminals that don't send meta correctly
export const FALLBACK_KEY_MAP: Record<string, string> = {
    "†": "alt+t", // Option+T on some macOS setups
    "π": "alt+p"  // Option+P on some macOS setups
};
