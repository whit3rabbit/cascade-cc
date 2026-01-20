
// Logic from chunk_502.ts (Gn2)

export class Emitter {
    private callbacks: Record<string, Function[]> = {};
    private warned = false;
    private maxListeners: number;

    constructor(options?: { maxListeners?: number }) {
        this.maxListeners = options?.maxListeners ?? 10;
    }

    private warnIfPossibleMemoryLeak(event: string) {
        if (this.warned) return;
        if (this.maxListeners && this.callbacks[event].length > this.maxListeners) {
            console.warn(`Event Emitter: Possible memory leak detected; ${String(event)} has exceeded ${this.maxListeners} listeners.`);
            this.warned = true;
        }
    }

    on(event: string, fn: Function): this {
        if (!this.callbacks[event]) {
            this.callbacks[event] = [fn];
        } else {
            this.callbacks[event].push(fn);
            this.warnIfPossibleMemoryLeak(event);
        }
        return this;
    }

    once(event: string, fn: Function): this {
        const on = (...args: any[]) => {
            this.off(event, on);
            fn.apply(this, args);
        };
        this.on(event, on);
        return this;
    }

    off(event: string, fn: Function): this {
        const callbacks = this.callbacks[event] ?? [];
        const newCallbacks = callbacks.filter((c) => c !== fn);
        this.callbacks[event] = newCallbacks;
        return this;
    }

    emit(event: string, ...args: any[]): this {
        const callbacks = this.callbacks[event] ?? [];
        callbacks.forEach((fn) => {
            fn.apply(this, args);
        });
        return this;
    }
}
