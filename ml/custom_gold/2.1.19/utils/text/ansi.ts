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
 * Checks if a code point is a full-width character.
 * Mirrored from E66 in chunk328.
 */
export function isFullwidthCodePoint(codePoint: number): boolean {
    if (!Number.isInteger(codePoint)) {
        return false;
    }
    return (
        codePoint >= 4352 &&
        (codePoint <= 4447 ||
            codePoint === 9001 ||
            codePoint === 9002 ||
            (11904 <= codePoint && codePoint <= 12871 && codePoint !== 12351) ||
            (12880 <= codePoint && codePoint <= 19903) ||
            (19968 <= codePoint && codePoint <= 42182) ||
            (43360 <= codePoint && codePoint <= 43388) ||
            (44032 <= codePoint && codePoint <= 55203) ||
            (63744 <= codePoint && codePoint <= 64255) ||
            (65040 <= codePoint && codePoint <= 65049) ||
            (65072 <= codePoint && codePoint <= 65131) ||
            (65281 <= codePoint && codePoint <= 65376) ||
            (65504 <= codePoint && codePoint <= 65510) ||
            (110592 <= codePoint && codePoint <= 110593) ||
            (127488 <= codePoint && codePoint <= 127569) ||
            (131072 <= codePoint && codePoint <= 262141))
    );
}

/**
 * Calculates the visual width of a string, accounting for ANSI escape codes and full-width characters.
 * Mirrored from F0A in chunk330.
 */
export function getWidth(
    input: string,
    options: { countAnsiEscapeCodes?: boolean; ambiguousIsNarrow?: boolean } = {}
): number {
    if (typeof input !== 'string' || input.length === 0) {
        return 0;
    }
    const { countAnsiEscapeCodes = false } = options;
    const text = countAnsiEscapeCodes ? input : stripAnsi(input);
    if (text.length === 0) {
        return 0;
    }

    let width = 0;
    const segmenter = new Intl.Segmenter();
    for (const { segment } of segmenter.segment(text)) {
        const codePoint = segment.codePointAt(0);
        if (!codePoint) continue;

        // Skip non-printable and ignorable characters
        if (
            (codePoint <= 31) ||
            (codePoint >= 127 && codePoint <= 159) ||
            (codePoint >= 8203 && codePoint <= 8207) ||
            (codePoint === 65279) ||
            (codePoint >= 768 && codePoint <= 879) ||
            (codePoint >= 6832 && codePoint <= 6911) ||
            (codePoint >= 7616 && codePoint <= 7679) ||
            (codePoint >= 8400 && codePoint <= 8447) ||
            (codePoint >= 65056 && codePoint <= 65071) ||
            (codePoint >= 55296 && codePoint <= 57343) ||
            (codePoint >= 65024 && codePoint <= 65039)
        ) {
            continue;
        }

        // Check if character is full-width
        if (isFullwidthCodePoint(codePoint)) {
            width += 2;
        } else {
            width += 1;
        }
    }
    return width;
}

export const stringWidthFn = getWidth;

/**
 * Slices a string while preserving ANSI escape codes and accounting for character width.
 * Mirrored from rb in chunk328.
 */
export function sliceAnsi(input: string, start: number, end?: number): string {
    const chars = [...input];
    const totalWidth = typeof end === 'number' ? end : input.length * 2;
    let currentWidth = 0;
    let result = '';

    let i = 0;
    while (i < chars.length) {
        const char = chars[i];
        if (char === '\u001b') {
            // Find end of ANSI sequence
            let j = i;
            while (j < chars.length && !/[a-zA-Z]/.test(chars[j])) {
                j++;
            }
            const ansiSequence = chars.slice(i, j + 1).join('');
            if (currentWidth < totalWidth) {
                result += ansiSequence;
            }
            i = j + 1;
            continue;
        }

        const width = isFullwidthCodePoint(char.codePointAt(0) || 0) ? 2 : 1;
        if (currentWidth + width > start && currentWidth < totalWidth) {
            result += char;
        }
        currentWidth += width;
        if (currentWidth >= totalWidth) break;
        i++;
    }

    return result;
}

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

    const inputWidth = getWidth(input);
    if (inputWidth <= columns) {
        return input;
    }

    let effectiveChar = truncationCharacter;
    const truncWidth = getWidth(effectiveChar);

    if (position === 'start') {
        if (preferTruncationOnSpace) {
            const splitPos = findNearestSpace(input, inputWidth - columns + 1, true);
            return effectiveChar + sliceAnsi(input, splitPos, inputWidth).trim();
        }
        if (space) effectiveChar += ' ';
        return effectiveChar + sliceAnsi(input, inputWidth - (columns - truncWidth), inputWidth);
    }

    if (position === 'middle') {
        if (space) effectiveChar = ` ${effectiveChar} `;
        const leftWidth = Math.floor(columns / 2);
        if (preferTruncationOnSpace) {
            const leftPos = findNearestSpace(input, leftWidth, false);
            const rightPos = findNearestSpace(input, inputWidth - (columns - leftWidth) + 1, true);
            return sliceAnsi(input, 0, leftPos) + effectiveChar + sliceAnsi(input, rightPos, inputWidth).trim();
        }
        return sliceAnsi(input, 0, leftWidth) + effectiveChar + sliceAnsi(input, inputWidth - (columns - leftWidth - truncWidth), inputWidth);
    }

    // Default: 'end'
    if (preferTruncationOnSpace) {
        const splitPos = findNearestSpace(input, columns - 1, false);
        return sliceAnsi(input, 0, splitPos) + effectiveChar;
    }
    if (space) effectiveChar = ` ${effectiveChar}`;
    return sliceAnsi(input, 0, columns - getWidth(effectiveChar)) + effectiveChar;
};

export default stripAnsi;
