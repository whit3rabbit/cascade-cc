/**
 * File: src/utils/shared/constants.ts
 * Role: Core application constants, signals, and shared enum-like objects.
 */

export const Signals = {
    SIGINT: 2,
    SIGQUIT: 3,
    SIGKILL: 9,
    SIGTERM: 15
} as const;

export const constants = {
    signals: Signals
};

// Aliases for compatibility with legacy obfuscated imports
export const signals = Signals;
