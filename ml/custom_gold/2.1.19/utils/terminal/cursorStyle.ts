/**
 * File: src/utils/terminal/cursorStyle.ts
 * Role: Utilities for changing terminal cursor shape using DECSCUSR escape sequences.
 */

export type CursorStyle = 'block' | 'beam' | 'underline';

/**
 * Sets the terminal cursor shape.
 * Based on DECSCUSR (CSI Ps q) standard.
 * 
 * Ps:
 * 0 or 1: Blinking block
 * 2: Steady block
 * 3: Blinking underline
 * 4: Steady underline
 * 5: Blinking bar (beam)
 * 6: Steady bar (beam)
 */
export function setCursorStyle(style: CursorStyle, blinking: boolean = true): void {
    if (!process.stdout.isTTY) return;

    let ps = 0;
    switch (style) {
        case 'block':
            ps = blinking ? 1 : 2;
            break;
        case 'underline':
            ps = blinking ? 3 : 4;
            break;
        case 'beam':
            ps = blinking ? 5 : 6;
            break;
    }

    process.stdout.write(`\x1b[${ps} q`);
}

/**
 * Resets the terminal cursor to its default shape.
 */
export function resetCursorStyle(): void {
    if (!process.stdout.isTTY) return;
    process.stdout.write('\x1b[0 q');
}
