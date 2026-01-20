
// Logic from chunk_599.ts (Idle Exit Timer)

/**
 * Automatically exits the process if no activity occurs within a specified duration.
 */
export function createIdleTimer(onExit: () => boolean, defaultDelay = 300000) {
    let timer: NodeJS.Timeout | null = null;
    const delay = parseInt(process.env.CLAUDE_CODE_EXIT_AFTER_STOP_DELAY || "0") || defaultDelay;

    return {
        start: () => {
            if (timer) clearTimeout(timer);
            if (delay > 0) {
                timer = setTimeout(() => {
                    if (onExit()) {
                        process.exit(0);
                    }
                }, delay);
            }
        },
        stop: () => {
            if (timer) {
                clearTimeout(timer);
                timer = null;
            }
        }
    };
}
