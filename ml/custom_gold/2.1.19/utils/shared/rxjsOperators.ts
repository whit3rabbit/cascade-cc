/**
 * File: src/utils/shared/rxjsOperators.ts
 * Role: Custom, lightweight RxJS-like operators and observable creation helpers.
 * Note: This is an internal implementation to avoid external dependencies.
 */

export enum NotificationKind {
    NEXT = 'N',
    ERROR = 'E',
    COMPLETE = 'C'
}

/**
 * Minimal Observable implementation.
 */
export class Observable<T> {
    constructor(private _subscribe: (subscriber: Subscriber<T>) => any) { }

    subscribe(observer: Partial<Observer<T>> | ((value: T) => void)): Subscription {
        const subscriber = new Subscriber<T>(observer);
        subscriber.add(this._subscribe(subscriber));
        return subscriber;
    }

    pipe<R>(...operators: Array<(source: Observable<any>) => Observable<any>>): Observable<R> {
        return operators.reduce((prev, fn) => fn(prev), this as any);
    }
}

export interface Observer<T> {
    next(value: T): void;
    error(err: any): void;
    complete(): void;
}

export class Subscriber<T> implements Observer<T> {
    protected _closed = false;
    private _teardowns: Array<() => void> = [];

    constructor(private destination: Partial<Observer<T>> | ((value: T) => void)) { }

    next(value: T): void {
        if (this._closed) return;
        if (typeof this.destination === 'function') {
            this.destination(value);
        } else {
            this.destination.next?.(value);
        }
    }

    error(err: any): void {
        if (this._closed) return;
        this._closed = true;
        if (typeof this.destination !== 'function') {
            this.destination.error?.(err);
        }
        this._teardown();
    }

    complete(): void {
        if (this._closed) return;
        this._closed = true;
        if (typeof this.destination !== 'function') {
            this.destination.complete?.();
        }
        this._teardown();
    }

    add(teardown: any): void {
        if (typeof teardown === 'function') this._teardowns.push(teardown);
        else if (teardown && typeof teardown.unsubscribe === 'function') this._teardowns.push(() => teardown.unsubscribe());
    }

    unsubscribe(): void {
        this._closed = true;
        this._teardown();
    }

    get closed() { return this._closed; }

    private _teardown(): void {
        this._teardowns.forEach(fn => fn());
        this._teardowns = [];
    }
}

export interface Subscription {
    unsubscribe(): void;
    readonly closed: boolean;
}

/**
 * Notification class.
 */
export class Notification<T> {
    constructor(
        public readonly kind: NotificationKind,
        public readonly value?: T,
        public readonly error?: any
    ) { }

    static createNext<T>(value: T): Notification<T> {
        return new Notification(NotificationKind.NEXT, value);
    }

    static createError(error: any): Notification<any> {
        return new Notification(NotificationKind.ERROR, undefined, error);
    }

    static createComplete(): Notification<any> {
        return new Notification(NotificationKind.COMPLETE);
    }
}

/**
 * Creates an observable from an array or iterable.
 */
export function from<T>(input: T[] | Iterable<T>): Observable<T> {
    return new Observable<T>(subscriber => {
        for (const item of input) {
            if (subscriber.closed) break;
            subscriber.next(item);
        }
        subscriber.complete();
    });
}

/**
 * Creates an observable from arguments.
 */
export function of<T>(...args: T[]): Observable<T> {
    return from(args);
}

/**
 * Custom operator for mapping values.
 */
export function map<T, R>(project: (value: T, index: number) => R) {
    return (source: Observable<T>) => new Observable<R>(subscriber => {
        let index = 0;
        return source.subscribe({
            next: (value) => subscriber.next(project(value, index++)),
            error: (err) => subscriber.error(err),
            complete: () => subscriber.complete(),
        });
    });
}

/**
 * Converts an observable to a promise for the first value.
 */
export function firstValueFrom<T>(source: Observable<T>, config?: { defaultValue: T }): Promise<T> {
    return new Promise((resolve, reject) => {
        let hasValue = false;
        const sub = source.subscribe({
            next: (v) => {
                hasValue = true;
                resolve(v);
                sub.unsubscribe();
            },
            error: (e) => reject(e),
            complete: () => {
                if (!hasValue) {
                    if (config && 'defaultValue' in config) resolve(config.defaultValue);
                    else reject(new Error("EmptyError: no elements in sequence"));
                }
            }
        });
    });
}
