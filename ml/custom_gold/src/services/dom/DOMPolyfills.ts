
// Logic from chunk_469.ts (DOM Polyfills, TokenList)

// --- DOM Token List implementation (mb2) ---
export class DOMTokenList {
    private _tokens: string[] = [];
    private _getString: () => string;
    private _setString: (v: string) => void;

    constructor(getString: () => string, setString: (v: string) => void) {
        this._getString = getString;
        this._setString = setString;
        this.updateFromSource();
    }

    private updateFromSource() {
        const s = this._getString() || "";
        this._tokens = s.trim().split(/\s+/).filter(t => t.length > 0);
    }

    add(...tokens: string[]) {
        tokens.forEach(t => {
            if (!this._tokens.includes(t)) this._tokens.push(t);
        });
        this.sync();
    }

    remove(...tokens: string[]) {
        this._tokens = this._tokens.filter(t => !tokens.includes(t));
        this.sync();
    }

    contains(token: string) {
        return this._tokens.includes(token);
    }

    toggle(token: string, force?: boolean) {
        const exists = this.contains(token);
        const shouldExist = force !== undefined ? force : !exists;
        if (shouldExist) this.add(token);
        else this.remove(token);
        return shouldExist;
    }

    private sync() {
        this._setString(this._tokens.join(" "));
    }

    toString() {
        return this._tokens.join(" ");
    }

    get length() {
        return this._tokens.length;
    }
}

// --- Collection implementation (fb2) ---
export class DOMCollection {
    // Stub for live node collection
    private items: any[] = [];
    constructor(items: any[]) {
        this.items = items;
    }
    item(i: number) { return this.items[i]; }
    get length() { return this.items.length; }
}

// --- Property Helpers (rE0) ---
export function createDOMProperty(name: string, type: any) {
    // Stub for defineProperty logic
    return {
        get() { return (this as any).getAttribute(name); },
        set(v: any) { (this as any).setAttribute(name, v); }
    }
}
