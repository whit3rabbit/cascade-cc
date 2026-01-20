
// Logic from chunk_4.ts (Observable / Subscription minimal polyfill)

export type TeardownLogic = Subscription | Unsubscribable | (() => void) | void;

export interface Unsubscribable {
    unsubscribe(): void;
}

export interface Observer<T> {
    next: (value: T) => void;
    error: (err: any) => void;
    complete: () => void;
}

export class Subscription implements Unsubscribable {
    public static EMPTY = (function () {
        const empty = new Subscription();
        empty.closed = true;
        return empty;
    })();

    public closed = false;
    private _parentage: Subscription[] | Subscription | null = null;
    private _finalizers: TeardownLogic[] | null = null;

    constructor(private initialTeardown?: () => void) { }

    unsubscribe(): void {
        if (this.closed) return;
        this.closed = true;

        // ... logic from RE1 ...
        const { _parentage } = this;
        if (_parentage) {
            this._parentage = null;
            if (Array.isArray(_parentage)) {
                for (const parent of _parentage) {
                    parent.remove(this);
                }
            } else {
                _parentage.remove(this);
            }
        }

        const { initialTeardown } = this;
        if (typeof initialTeardown === 'function') {
            try {
                initialTeardown();
            } catch (e) {
                console.error(e);
            }
        }

        const { _finalizers } = this;
        if (_finalizers) {
            this._finalizers = null;
            for (const finalizer of _finalizers) {
                try {
                    execTeardown(finalizer);
                } catch (e) {
                    console.error(e);
                }
            }
        }
    }

    add(teardown: TeardownLogic): void {
        if (teardown && teardown !== this) {
            if (this.closed) {
                execTeardown(teardown);
            } else {
                if (teardown instanceof Subscription) {
                    if (teardown.closed || teardown._hasParent(this)) {
                        return;
                    }
                    teardown._addParent(this);
                }
                (this._finalizers = this._finalizers || []).push(teardown);
            }
        }
    }

    remove(teardown: TeardownLogic): void {
        const { _finalizers } = this;
        if (_finalizers && _finalizers.length > 0) { // Simplified remove logic
            const index = _finalizers.indexOf(teardown);
            if (index > -1) _finalizers.splice(index, 1);
        }
        if (teardown instanceof Subscription) {
            teardown._removeParent(this);
        }
    }

    private _hasParent(parent: Subscription) {
        const { _parentage } = this;
        return _parentage === parent || (Array.isArray(_parentage) && _parentage.includes(parent));
    }

    private _addParent(parent: Subscription) {
        const { _parentage } = this;
        this._parentage = Array.isArray(_parentage) ? (_parentage.push(parent), _parentage) : _parentage ? [_parentage, parent] : parent;
    }

    private _removeParent(parent: Subscription) {
        const { _parentage } = this;
        if (_parentage === parent) {
            this._parentage = null;
        } else if (Array.isArray(_parentage)) {
            const index = _parentage.indexOf(parent);
            if (index > -1) _parentage.splice(index, 1);
        }
    }
}

function execTeardown(teardown: TeardownLogic) {
    if (typeof teardown === 'function') teardown();
    else if (typeof teardown === 'object' && teardown !== null && typeof (teardown as any).unsubscribe === 'function') (teardown as any).unsubscribe();
}


// Logic from UE1, t, pY (Logger)
import fs from "fs";
import path from "path";
// import { getSessionId } from "../../state/session";
// import { homedir, tmpdir } from "os";

export const LOG_DIR = path.join(process.cwd(), ".claude_logs"); // Placeholder

export function logError(error: any) { // t
    // ... logic from t ...
    console.error(error);
}

export function logMcpError(serverName: string, error: any) { // pY
    console.error(`MCP Server ${serverName} error:`, error);
}

export function logMcpDebug(serverName: string, message: string) { // QQ
    console.log(`MCP Server ${serverName}: ${message}`);
}
