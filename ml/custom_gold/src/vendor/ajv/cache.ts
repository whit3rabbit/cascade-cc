export default class Cache {
    private _cache: { [key: string]: any } = {};

    put(key: string, value: any) {
        this._cache[key] = value;
    }

    get(key: string) {
        return this._cache[key];
    }

    del(key: string) {
        delete this._cache[key];
    }

    clear() {
        this._cache = {};
    }
}
