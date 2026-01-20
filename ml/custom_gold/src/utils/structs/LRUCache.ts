
import { LinkedMap, Touch } from "./LinkedMap.js";

export class LRUCache<K, V> extends LinkedMap<K, V> {
    private _limit: number;
    private _ratio: number;

    constructor(limit: number, ratio: number = 1) {
        super();
        this._limit = limit;
        this._ratio = Math.min(Math.max(0, ratio), 1);
    }

    get limit(): number {
        return this._limit;
    }

    set limit(limit: number) {
        this._limit = limit;
        this.checkTrim();
    }

    get ratio(): number {
        return this._ratio;
    }

    set ratio(ratio: number) {
        this._ratio = Math.min(Math.max(0, ratio), 1);
        this.checkTrim();
    }

    get(key: K, touch: Touch = Touch.AsNew): V | undefined {
        return super.get(key, touch);
    }

    peek(key: K): V | undefined {
        return super.get(key, Touch.None);
    }

    set(key: K, value: V): this {
        super.set(key, value, Touch.Last);
        this.checkTrim();
        return this;
    }

    has(key: K): boolean {
        return super.has(key);
    }

    delete(key: K): boolean {
        return super.delete(key);
    }

    protected checkTrim(): void {
        if (this.size > this._limit) {
            this.trimOld(Math.round(this._limit * this._ratio));
        }
    }
}
