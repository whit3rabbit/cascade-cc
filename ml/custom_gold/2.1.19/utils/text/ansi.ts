import stringWidth from 'string-width';
import sliceAnsi from 'slice-ansi';

export const ANSI_REGEX = /[\u001b\u009b][[()#;?]*(?:[0-9]{1,4}(?:;[0-9]{0,4})*)?[0-9A-ORZcf-nqry=><]/g;

/**
 * Strips ANSI escape codes from a string.
 */
export function stripAnsi(text: string): string {
    if (typeof text !== 'string') {
        return text;
    }
    return text.replace(ANSI_REGEX, '');
}

/**
 * Alias for legacy/obfuscated usage and compatibility.
 */
export const eU = stripAnsi;
export const AH = stripAnsi;

/**
 * Calculates the visual width of a string.
 */
export function getWidth(text: string): number {
    return stringWidth(text);
}

export const stringWidthFn = getWidth;

/**
 * Finds the nearest space to truncate at for cleaner text.
 * Aligned with y71() from chunk330.
 */
function findNearestSpace(text: string, index: number, forward: boolean = false): number {
    if (text.charAt(index) === ' ') {
        return index;
    }
    const direction = forward ? 1 : -1;
    for (let i = 1; i <= 3; i++) {
        const checkIndex = index + i * direction;
        if (checkIndex < 0 || checkIndex >= text.length) break;
        if (text.charAt(checkIndex) === ' ') {
            return checkIndex;
        }
    }
    return index;
}

/**
 * Truncates a string to a specific width, handling ANSI codes and position.
 * Aligned with k66() from chunk330.
 */
export const truncateString = (
    input: string,
    columns: number,
    options: {
        position?: 'start' | 'middle' | 'end';
        truncationCharacter?: string;
        space?: boolean;
        preferTruncationOnSpace?: boolean;
    } = {}
): string => {
    const {
        position = 'end',
        truncationCharacter = '…',
        space = false,
        preferTruncationOnSpace = false
    } = options;

    if (typeof input !== 'string') throw new TypeError(`Expected input to be a string, got ${typeof input}`);
    if (typeof columns !== 'number') throw new TypeError(`Expected columns to be a number, got ${typeof columns}`);
    if (columns < 1) return '';
    if (columns === 1) return truncationCharacter;

    const inputWidth = stringWidth(input);
    if (inputWidth <= columns) {
        return input;
    }

    let effectiveChar = truncationCharacter;
    if (position === 'start') {
        if (preferTruncationOnSpace) {
            const splitPos = findNearestSpace(input, inputWidth - columns + 1, true);
            return effectiveChar + sliceAnsi(input, splitPos, inputWidth).trim();
        }
        if (space) effectiveChar += ' ';
        return effectiveChar + sliceAnsi(input, inputWidth - (columns - stringWidth(effectiveChar)), inputWidth);
    }

    if (position === 'middle') {
        if (space) effectiveChar = ` ${effectiveChar} `;
        const leftWidth = Math.floor(columns / 2);
        if (preferTruncationOnSpace) {
            const leftPos = findNearestSpace(input, leftWidth, false);
            const rightPos = findNearestSpace(input, inputWidth - (columns - leftWidth) + 1, true);
            return sliceAnsi(input, 0, leftPos) + effectiveChar + sliceAnsi(input, rightPos, inputWidth).trim();
        }
        return sliceAnsi(input, 0, leftWidth) + effectiveChar + sliceAnsi(input, inputWidth - (columns - leftWidth - stringWidth(effectiveChar)), inputWidth);
    }

    // Default: 'end'
    if (preferTruncationOnSpace) {
        const splitPos = findNearestSpace(input, columns - 1, false);
        return sliceAnsi(input, 0, splitPos) + effectiveChar;
    }
    if (space) effectiveChar = ` ${effectiveChar}`;
    return sliceAnsi(input, 0, columns - stringWidth(effectiveChar)) + effectiveChar;
};

export default stripAnsi;
