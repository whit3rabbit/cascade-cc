export default function stringify(obj: any, opts: any = {}): string | undefined {
    if (typeof opts === 'function') opts = { cmp: opts };
    const cycles = typeof opts.cycles === 'boolean' ? opts.cycles : false;
    const cmp = opts.cmp && (function (f) {
        return function (node: any) {
            return function (a: string, b: string) {
                const aObj = { key: a, value: node[a] };
                const bObj = { key: b, value: node[b] };
                return f(aObj, bObj);
            };
        };
    })(opts.cmp);

    const stack: any[] = [];

    return (function _stringify(node: any): string | undefined {
        if (node && node.toJSON && typeof node.toJSON === 'function') node = node.toJSON();
        if (node === undefined) return;
        if (typeof node === 'number') return isFinite(node) ? '' + node : 'null';
        if (typeof node !== 'object') return JSON.stringify(node);

        if (Array.isArray(node)) {
            let res = '[';
            for (let i = 0; i < node.length; i++) {
                if (i) res += ',';
                res += _stringify(node[i]) || 'null';
            }
            return res + ']';
        }

        if (node === null) return 'null';

        if (stack.indexOf(node) !== -1) {
            if (cycles) return JSON.stringify('__cycle__');
            throw new TypeError('Converting circular structure to JSON');
        }

        const index = stack.push(node) - 1;
        const keys = Object.keys(node).sort(cmp && cmp(node));
        let res = '';
        for (let i = 0; i < keys.length; i++) {
            const key = keys[i];
            const value = _stringify(node[key]);
            if (!value) continue;
            if (res) res += ',';
            res += JSON.stringify(key) + ':' + value;
        }
        stack.splice(index, 1);
        return '{' + res + '}';
    })(obj);
}
