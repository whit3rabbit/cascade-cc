import { onExit } from 'signal-exit';
import { EventEmitter } from 'events';

export function platform(): string {
    return process.platform;
}

export function terminal(): string {
    return process.env.TERM || 'unknown';
}

class ProcessManager extends EventEmitter {
    constructor() {
        super();
        // Uses signal-exit to reliably capture exit events (SIGINT, SIGTERM, normal exit)
        onExit((code: number | null | undefined, signal: string | null) => {
            // Emit 'afterexit' to allow listeners (like Logger) to perform final cleanup
            // Note: This must be synchronous or "fire and forget" as the process is exiting.
            // If syncToR2 is async, we might not complete it, but we can try.
            this.emit('afterexit', { code, signal });
        }, { alwaysLast: true });
    }
}

export const processManager = new ProcessManager();
