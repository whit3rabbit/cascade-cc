
export namespace Touch {
    export const None = 0;
    export const First = 1;
    export const AsOld = First;
    export const Last = 2;
    export const AsNew = Last;
}

export type Touch = 0 | 1 | 2;

interface Item<K, V> {
    key: K;
    value: V;
    next?: Item<K, V>;
    previous?: Item<K, V>;
}

export class LinkedMap<K, V> {
    private _map: Map<K, Item<K, V>>;
    private _head?: Item<K, V>;
    private _tail?: Item<K, V>;
    private _size: number;
    private _state: number;

    constructor() {
        this._map = new Map<K, Item<K, V>>();
        this._head = undefined;
        this._tail = undefined;
        this._size = 0;
        this._state = 0;
    }

    clear(): void {
        this._map.clear();
        this._head = undefined;
        this._tail = undefined;
        this._size = 0;
        this._state++;
    }

    isEmpty(): boolean {
        return !this._head && !this._tail;
    }

    get size(): number {
        return this._size;
    }

    get first(): V | undefined {
        return this._head?.value;
    }

    get last(): V | undefined {
        return this._tail?.value;
    }

    has(key: K): boolean {
        return this._map.has(key);
    }

    get(key: K, touch: Touch = Touch.None): V | undefined {
        const item = this._map.get(key);
        if (!item) {
            return undefined;
        }
        if (touch !== Touch.None) {
            this.touch(item, touch);
        }
        return item.value;
    }

    set(key: K, value: V, touch: Touch = Touch.None): this {
        let item = this._map.get(key);
        if (item) {
            item.value = value;
            if (touch !== Touch.None) {
                this.touch(item, touch);
            }
        } else {
            item = { key, value, next: undefined, previous: undefined };
            switch (touch) {
                case Touch.None:
                    this.addItemLast(item);
                    break;
                case Touch.First:
                    this.addItemFirst(item);
                    break;
                case Touch.Last:
                    this.addItemLast(item);
                    break;
                default:
                    this.addItemLast(item);
                    break;
            }
            this._map.set(key, item);
            this._size++;
        }
        return this;
    }

    delete(key: K): boolean {
        return !!this.remove(key);
    }

    remove(key: K): V | undefined {
        const item = this._map.get(key);
        if (!item) {
            return undefined;
        }
        this._map.delete(key);
        this.removeItem(item);
        this._size--;
        return item.value;
    }

    shift(): V | undefined {
        if (!this._head && !this._tail) {
            return undefined;
        }
        if (!this._head || !this._tail) {
            throw new Error("Invalid list");
        }
        const item = this._head;
        this._map.delete(item.key);
        this.removeItem(item);
        this._size--;
        return item.value;
    }

    forEach(callback: (value: V, key: K, map: LinkedMap<K, V>) => void, thisArg?: any): void {
        const state = this._state;
        let current = this._head;
        while (current) {
            if (thisArg) {
                callback.bind(thisArg)(current.value, current.key, this);
            } else {
                callback(current.value, current.key, this);
            }
            if (this._state !== state) {
                throw new Error("LinkedMap got modified during iteration.");
            }
            current = current.next;
        }
    }

    keys(): IterableIterator<K> {
        const state = this._state;
        let current = this._head;
        const iterator: IterableIterator<K> = {
            [Symbol.iterator]: () => {
                return iterator;
            },
            next: (): IteratorResult<K> => {
                if (this._state !== state) {
                    throw new Error("LinkedMap got modified during iteration.");
                }
                if (current) {
                    const result = { value: current.key, done: false };
                    current = current.next;
                    return result;
                } else {
                    return { value: undefined as any, done: true };
                }
            }
        };
        return iterator;
    }

    values(): IterableIterator<V> {
        const state = this._state;
        let current = this._head;
        const iterator: IterableIterator<V> = {
            [Symbol.iterator]: () => {
                return iterator;
            },
            next: (): IteratorResult<V> => {
                if (this._state !== state) {
                    throw new Error("LinkedMap got modified during iteration.");
                }
                if (current) {
                    const result = { value: current.value, done: false };
                    current = current.next;
                    return result;
                } else {
                    return { value: undefined as any, done: true };
                }
            }
        };
        return iterator;
    }

    entries(): IterableIterator<[K, V]> {
        const state = this._state;
        let current = this._head;
        const iterator: IterableIterator<[K, V]> = {
            [Symbol.iterator]: () => {
                return iterator;
            },
            next: (): IteratorResult<[K, V]> => {
                if (this._state !== state) {
                    throw new Error("LinkedMap got modified during iteration.");
                }
                if (current) {
                    const result: IteratorResult<[K, V]> = { value: [current.key, current.value], done: false };
                    current = current.next;
                    return result;
                } else {
                    return { value: undefined as any, done: true };
                }
            }
        };
        return iterator;
    }

    [Symbol.iterator](): IterableIterator<[K, V]> {
        return this.entries();
    }

    protected trimOld(newSize: number): void {
        if (newSize >= this.size) {
            return;
        }
        if (newSize === 0) {
            this.clear();
            return;
        }
        let current = this._head;
        let currentSize = this.size;
        while (current && currentSize > newSize) {
            this._map.delete(current.key);
            current = current.next;
            currentSize--;
        }
        this._head = current;
        this._size = currentSize;
        if (current) {
            current.previous = undefined;
        }
        this._state++;
    }

    private addItemFirst(item: Item<K, V>): void {
        if (!this._head && !this._tail) {
            this._tail = item;
        } else if (!this._head) {
            throw new Error("Invalid list");
        } else {
            item.next = this._head;
            this._head.previous = item;
        }
        this._head = item;
        this._state++;
    }

    private addItemLast(item: Item<K, V>): void {
        if (!this._head && !this._tail) {
            this._head = item;
        } else if (!this._tail) {
            throw new Error("Invalid list");
        } else {
            item.previous = this._tail;
            this._tail.next = item;
        }
        this._tail = item;
        this._state++;
    }

    private removeItem(item: Item<K, V>): void {
        if (item === this._head && item === this._tail) {
            this._head = undefined;
            this._tail = undefined;
        } else if (item === this._head) {
            if (!item.next) {
                throw new Error("Invalid list");
            }
            item.next.previous = undefined;
            this._head = item.next;
        } else if (item === this._tail) {
            if (!item.previous) {
                throw new Error("Invalid list");
            }
            item.previous.next = undefined;
            this._tail = item.previous;
        } else {
            const next = item.next;
            const previous = item.previous;
            if (!next || !previous) {
                throw new Error("Invalid list");
            }
            next.previous = previous;
            previous.next = next;
        }
        item.next = undefined;
        item.previous = undefined;
        this._state++;
    }

    private touch(item: Item<K, V>, touch: Touch): void {
        if (!this._head || !this._tail) {
            throw new Error("Invalid list");
        }
        if (touch !== Touch.First && touch !== Touch.Last) {
            return;
        }
        if (touch === Touch.First) {
            if (item === this._head) {
                return;
            }
            const next = item.next;
            const previous = item.previous;
            if (item === this._tail) {
                // previous must be defined since item is tail and not head
                if (previous) {
                    previous.next = undefined;
                    this._tail = previous;
                }
            } else {
                // both next and previous must be defined
                if (next) next.previous = previous;
                if (previous) previous.next = next;
            }
            item.previous = undefined;
            item.next = this._head;
            this._head.previous = item;
            this._head = item;
            this._state++;
        } else if (touch === Touch.Last) {
            if (item === this._tail) {
                return;
            }
            const next = item.next;
            const previous = item.previous;
            if (item === this._head) {
                // next must be defined since item is head and not tail
                if (next) {
                    next.previous = undefined;
                    this._head = next;
                }
            } else {
                if (next) next.previous = previous;
                if (previous) previous.next = next;
            }
            item.next = undefined;
            item.previous = this._tail;
            this._tail.next = item;
            this._tail = item;
            this._state++;
        }
    }
}
