
// Ported from chunk_575.ts (asciichart logic)

export const black = "\x1B[30m";
export const red = "\x1B[31m";
export const green = "\x1B[32m";
export const yellow = "\x1B[33m";
export const blue = "\x1B[34m";
export const magenta = "\x1B[35m";
export const cyan = "\x1B[36m";
export const lightgray = "\x1B[37m";
export const defaultColor = "\x1B[39m";
export const darkgray = "\x1B[90m";
export const lightred = "\x1B[91m";
export const lightgreen = "\x1B[92m";
export const lightyellow = "\x1B[93m";
export const lightblue = "\x1B[94m";
export const lightmagenta = "\x1B[95m";
export const lightcyan = "\x1B[96m";
export const white = "\x1B[97m";
export const reset = "\x1B[0m";

export function colored(text: string, color?: string): string {
    return color === undefined ? text : color + text + reset;
}

export interface PlotConfig {
    offset?: number;
    padding?: string;
    height?: number;
    colors?: string[];
    min?: number;
    max?: number;
    symbols?: string[];
    format?: (x: number, i: number) => string;
}

export function plot(series: number[] | number[][], cfg: PlotConfig = {}): string {
    let data: number[][] = [];
    if (typeof series[0] === "number") {
        data = [series as number[]];
    } else {
        data = series as number[][];
    }

    let min = cfg.min ?? data[0][0];
    let max = cfg.max ?? data[0][0];

    for (let i = 0; i < data.length; i++) {
        for (let j = 0; j < data[i].length; j++) {
            min = Math.min(min, data[i][j]);
            max = Math.max(max, data[i][j]);
        }
    }

    const defaultSymbols = ["┼", "┤", "╶", "╴", "─", "╰", "╭", "╮", "╯", "│"];
    const symbols = cfg.symbols ?? defaultSymbols;

    const range = Math.abs(max - min);
    const offset = cfg.offset ?? 3;
    const padding = cfg.padding ?? "           ";
    const height = cfg.height ?? range;
    const colors = cfg.colors ?? [];
    const ratio = range !== 0 ? height / range : 1;

    const min2 = Math.round(min * ratio);
    const max2 = Math.round(max * ratio);
    const rows = Math.abs(max2 - min2);

    let width = 0;
    for (let i = 0; i < data.length; i++) {
        width = Math.max(width, data[i].length);
    }
    width = width + offset;

    const format = cfg.format ?? ((x) => (padding + x.toFixed(2)).slice(-padding.length));

    const result: string[][] = [];
    for (let i = 0; i <= rows; i++) {
        result[i] = new Array(width).fill(" ");
    }

    // Draw axis
    for (let y = min2; y <= max2; ++y) {
        let label = format(rows > 0 ? max - (y - min2) * range / rows : min, y - min2);
        result[y - min2][Math.max(offset - label.length, 0)] = label;
        result[y - min2][offset - 1] = y === 0 ? symbols[0] : symbols[1];
    }

    // Draw data
    for (let i = 0; i < data.length; i++) {
        const currentColor = colors[i % colors.length];
        const y0 = Math.round(data[i][0] * ratio) - min2;

        result[rows - y0][offset - 1] = colored(symbols[0], currentColor);

        for (let x = 0; x < data[i].length - 1; x++) {
            const y1 = Math.round(data[i][x] * ratio) - min2;
            const y2 = Math.round(data[i][x + 1] * ratio) - min2;

            if (y1 === y2) {
                result[rows - y1][x + offset] = colored(symbols[4], currentColor);
            } else {
                result[rows - y2][x + offset] = colored(y1 > y2 ? symbols[5] : symbols[6], currentColor);
                result[rows - y1][x + offset] = colored(y1 > y2 ? symbols[7] : symbols[8], currentColor);

                const start = Math.min(y1, y2);
                const end = Math.max(y1, y2);
                for (let y = start + 1; y < end; y++) {
                    result[rows - y][x + offset] = colored(symbols[9], currentColor);
                }
            }
        }
    }

    return result.map(row => row.join("")).join("\n");
}
