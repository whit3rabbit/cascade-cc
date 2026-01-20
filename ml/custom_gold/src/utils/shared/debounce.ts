/**
 * Full-featured debounce implementation.
 * Deobfuscated from je8 (OKB) in chunk_205.ts.
 */

export function debounce<T extends (...args: any[]) => any>(
    func: T,
    wait: number = 0,
    options: { leading?: boolean; trailing?: boolean; maxWait?: number } = {}
): T & { cancel(): void; flush(): any } {
    let lastArgs: any;
    let lastThis: any;
    let maxWait: number | undefined = options.maxWait;
    let result: any;
    let timerId: NodeJS.Timeout | undefined;
    let lastCallTime: number | undefined;
    let lastInvokeTime: number = 0;
    let leading = !!options.leading;
    let trailing = "trailing" in options ? !!options.trailing : true;

    function invokeFunc(time: number) {
        const args = lastArgs;
        const thisArg = lastThis;
        lastArgs = lastThis = undefined;
        lastInvokeTime = time;
        result = func.apply(thisArg, args);
        return result;
    }

    function startTimer(pendingFunc: () => void, wait: number) {
        return setTimeout(pendingFunc, wait);
    }

    function leadingEdge(time: number) {
        lastInvokeTime = time;
        timerId = startTimer(timerTick, wait);
        return leading ? invokeFunc(time) : result;
    }

    function remainingWait(time: number) {
        const timeSinceLastCall = time - (lastCallTime || 0);
        const timeSinceLastInvoke = time - lastInvokeTime;
        const timeWaiting = wait - timeSinceLastCall;
        return maxWait === undefined
            ? timeWaiting
            : Math.min(timeWaiting, maxWait - timeSinceLastInvoke);
    }

    function shouldInvoke(time: number) {
        const timeSinceLastCall = time - (lastCallTime || 0);
        const timeSinceLastInvoke = time - lastInvokeTime;
        return (
            lastCallTime === undefined ||
            timeSinceLastCall >= wait ||
            timeSinceLastCall < 0 ||
            (maxWait !== undefined && timeSinceLastInvoke >= maxWait)
        );
    }

    function timerTick() {
        const time = Date.now();
        if (shouldInvoke(time)) {
            return trailingEdge(time);
        }
        timerId = startTimer(timerTick, remainingWait(time));
    }

    function trailingEdge(time: number) {
        timerId = undefined;
        if (trailing && lastArgs) {
            return invokeFunc(time);
        }
        lastArgs = lastThis = undefined;
        return result;
    }

    function debounced(this: any, ...args: any[]) {
        const time = Date.now();
        const isInvoking = shouldInvoke(time);
        lastArgs = args;
        lastThis = this;
        lastCallTime = time;

        if (isInvoking) {
            if (timerId === undefined) {
                return leadingEdge(lastCallTime);
            }
            if (maxWait !== undefined) {
                clearTimeout(timerId);
                timerId = startTimer(timerTick, wait);
                return invokeFunc(lastCallTime);
            }
        }
        if (timerId === undefined) {
            timerId = startTimer(timerTick, wait);
        }
        return result;
    }

    debounced.cancel = () => {
        if (timerId !== undefined) clearTimeout(timerId);
        lastInvokeTime = 0;
        lastArgs = lastCallTime = lastThis = timerId = undefined;
    };

    debounced.flush = () => {
        return timerId === undefined ? result : trailingEdge(Date.now());
    };

    return debounced as any;
}
