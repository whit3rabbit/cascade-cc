import { EventEmitter } from "events";

/**
 * Event object for terminal interactions.
 * Deobfuscated from NYA in chunk_201.ts.
 */
export class TerminalEvent {
    private _didStopImmediatePropagation = false;

    didStopImmediatePropagation(): boolean {
        return this._didStopImmediatePropagation;
    }

    stopImmediatePropagation(): void {
        this._didStopImmediatePropagation = true;
    }
}

/**
 * Custom EventEmitter that supports stopImmediatePropagation.
 * Deobfuscated from _i in chunk_201.ts.
 */
export class TerminalEventEmitter extends EventEmitter {
    override emit(eventName: string | symbol, ...args: any[]): boolean {
        if (eventName === "error") {
            return super.emit(eventName, ...args);
        }

        const listeners = this.rawListeners(eventName);
        if (listeners.length === 0) {
            return false;
        }

        const eventObj = args[0] instanceof TerminalEvent ? args[0] : null;

        for (const listener of listeners) {
            (listener as Function).apply(this, args);
            if (eventObj?.didStopImmediatePropagation()) {
                break;
            }
        }

        return true;
    }
}
