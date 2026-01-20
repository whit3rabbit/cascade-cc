
// Logic from chunk_576.ts (Analytics Streaks & Heatmap, Wasm)

// Logic from chunk_576.ts (Analytics Streaks & Heatmap, Wasm)


// --- Streak Calculation (mZ7) ---
export function calculateStreaks(activity: { date: string }[]) {
    if (activity.length === 0) {
        return {
            currentStreak: 0,
            longestStreak: 0,
            currentStreakStart: null,
            longestStreakStart: null,
            longestStreakEnd: null
        };
    }

    const today = new Date();
    today.setHours(0, 0, 0, 0);

    // Current Streak
    let currentStreak = 0;
    let currentStreakStart: string | null = null;
    let checkDate = new Date(today);

    // Create a Set of active dates for O(1) lookup
    const activeDates = new Set(activity.map(a => a.date));

    // Check backwards from today for current streak
    while (true) {
        const dateStr = checkDate.toISOString().split("T")[0];
        if (!activeDates.has(dateStr)) break;

        currentStreak++;
        currentStreakStart = dateStr;
        checkDate.setDate(checkDate.getDate() - 1);
    }

    // Longest Streak
    let longestStreak = 0;
    let longestStreakStart: string | null = null;
    let longestStreakEnd: string | null = null;

    if (activity.length > 0) {
        const sortedDates = Array.from(activeDates).sort();
        let tempStreak = 1;
        let tempStart = sortedDates[0];

        for (let i = 1; i < sortedDates.length; i++) {
            const d1 = new Date(sortedDates[i - 1]);
            const d2 = new Date(sortedDates[i]);
            const diffDays = Math.round((d2.getTime() - d1.getTime()) / 86400000);

            if (diffDays === 1) {
                tempStreak++;
            } else {
                if (tempStreak > longestStreak) {
                    longestStreak = tempStreak;
                    longestStreakStart = tempStart;
                    longestStreakEnd = sortedDates[i - 1];
                }
                tempStreak = 1;
                tempStart = sortedDates[i];
            }
        }

        // Check last streak
        if (tempStreak > longestStreak) {
            longestStreak = tempStreak;
            longestStreakStart = tempStart;
            longestStreakEnd = sortedDates[sortedDates.length - 1];
        }
    }

    return {
        currentStreak,
        longestStreak,
        currentStreakStart,
        longestStreakStart,
        longestStreakEnd
    };
}

// --- Heatmap Rendering (aO0) ---
const INTENSITY_SYMBOLS = ["·", "░", "▒", "▓", "█"];
const HEATMAP_COLOR = "\x1B[38;2;218;119;86m"; // #da7756
const RESET = "\x1B[0m";
const GRAY = "\x1B[90m";

export function renderContributionCalendar(activity: { date: string, messageCount: number }[], width = 80) {
    const activityMap = new Map(activity.map(a => [a.date, a]));
    const counts = activity.map(a => a.messageCount).filter(c => c > 0).sort((a, b) => a - b);

    // Percentiles for intensity
    const p25 = counts[Math.floor(counts.length * 0.25)] || 0;
    const p50 = counts[Math.floor(counts.length * 0.5)] || 0;
    const p75 = counts[Math.floor(counts.length * 0.75)] || 0;

    const getIntensity = (count: number) => {
        if (count === 0) return 0;
        if (count >= p75) return 4;
        if (count >= p50) return 3;
        if (count >= p25) return 2;
        return 1;
    };

    const getSymbol = (intensity: number) => {
        const symbol = INTENSITY_SYMBOLS[intensity];
        if (intensity === 0) return `${GRAY}${symbol}${RESET}`;
        return `${HEATMAP_COLOR}${symbol}${RESET}`;
    };

    // Calculate dimensions
    const labelWidth = 4; // "Mon " etc
    const availableWidth = width - labelWidth;
    const weeksToDisplay = Math.min(52, Math.floor(availableWidth));
    // chunk_576 logic uses (width - 4) for calc, min 52.

    const today = new Date();
    today.setHours(0, 0, 0, 0);

    // Start date is (Today - DayOfWeek) - (Weeks * 7)?
    // chunk_576 logic:
    // W = today (00:00)
    // K = W - W.day (Start of this week, Sunday)
    // V = K - (weeks - 1) * 7 (Start date)

    const startOfThisWeek = new Date(today);
    startOfThisWeek.setDate(today.getDate() - today.getDay());

    const startDate = new Date(startOfThisWeek);
    startDate.setDate(startDate.getDate() - (weeksToDisplay - 1) * 7);

    // Grid: 7 rows x weeksToDisplay columns
    const grid: string[][] = Array.from({ length: 7 }, () => Array(weeksToDisplay).fill(""));
    const monthLabels: { month: number, week: number }[] = [];
    let lastMonth = -1;

    let currentDate = new Date(startDate);

    for (let w = 0; w < weeksToDisplay; w++) {
        for (let d = 0; d < 7; d++) {
            if (currentDate > today) {
                grid[d][w] = " ";
                currentDate.setDate(currentDate.getDate() + 1);
                continue;
            }

            const dateStr = currentDate.toISOString().split("T")[0];
            const entry = activityMap.get(dateStr);
            const count = entry?.messageCount || 0;
            const intensity = getIntensity(count);

            // Check for month label change on Sunday (d=0) or just first appearance?
            // chunk_576 checks on Sunday if month changed from previous processing
            if (d === 0) {
                const m = currentDate.getMonth();
                if (m !== lastMonth) {
                    monthLabels.push({ month: m, week: w });
                    lastMonth = m;
                }
            }

            grid[d][w] = getSymbol(intensity);
            currentDate.setDate(currentDate.getDate() + 1);
        }
    }

    // Build Output
    const resultLines: string[] = [];

    // 1. Month Labels
    const monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
    // Spacing logic needs to be precise so labels align with weeks
    // chunk_576 pads labels based on week distance
    // Simplified: create a label string
    let header = "    "; // Pad for day labels
    let currentWeekIndex = 0;

    for (const label of monthLabels) {
        const gap = label.week - currentWeekIndex;
        if (gap > 0) header += " ".repeat(gap);
        const name = monthNames[label.month];
        if (header.length + name.length < width) {
            header += name;
            currentWeekIndex = label.week + name.length;
        } else {
            currentWeekIndex = label.week; // Skip if no space
        }
    }
    resultLines.push(header);

    // 2. Grid
    const dayNames = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    for (let d = 0; d < 7; d++) {
        const label = [1, 3, 5].includes(d) ? dayNames[d].padEnd(3) : "   ";
        resultLines.push(`${label} ${grid[d].join("")}`);
    }

    // 3. Legend
    const legend = `    Less ${getIntensity(0) === 0 ? getSymbol(0) : ""} ${getSymbol(1)} ${getSymbol(2)} ${getSymbol(3)} ${getSymbol(4)} More`;
    resultLines.push(""); // Empty line
    resultLines.push(legend);

    return resultLines.join("\n");
}

// --- Wasm Instantiation (eZ7) ---
export async function instantiateWasm(module: WebAssembly.Module | Buffer | Response, imports: any) {
    if (typeof Response === "function" && module instanceof Response) {
        if (typeof WebAssembly.instantiateStreaming === "function") {
            try {
                return await WebAssembly.instantiateStreaming(module, imports);
            } catch (e: any) {
                if (module.headers.get("Content-Type") !== "application/wasm") {
                    console.warn("`WebAssembly.instantiateStreaming` failed because MIME type is not application/wasm. Falling back.", e);
                } else {
                    throw e;
                }
            }
        }
        const buffer = await module.arrayBuffer();
        return await WebAssembly.instantiate(buffer, imports);
    } else {
        const result = await WebAssembly.instantiate(module as any, imports);
        if (result instanceof WebAssembly.Instance) {
            return { instance: result, module: module };
        }
        return result;
    }
}
