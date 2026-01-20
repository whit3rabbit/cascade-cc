
// fast-safe-stringify (Vendored from chunk_478.ts)

export default function stringify(data: any, options?: any): string {
    if (!options) options = {};
    if (typeof options === "function") options = { cmp: options };
    var cycles = typeof options.cycles === "boolean" ? options.cycles : false;
    var cmp = options.cmp && function (f: any) {
        return function (node: any) {
            return function (a: any, b: any) {
                var aobj = { key: a, value: node[a] };
                var bobj = { key: b, value: node[b] };
                return f(aobj, bobj);
            };
        };
    }(options.cmp);
    var seen: any[] = [];

    return (function stringify(node: any): any {
        if (node && node.toJSON && typeof node.toJSON === "function") node = node.toJSON();
        if (node === undefined) return;
        if (typeof node === "number") return isFinite(node) ? "" + node : "null";
        if (typeof node !== "object") return JSON.stringify(node);

        var i, out;
        if (Array.isArray(node)) {
            out = "[";
            for (i = 0; i < node.length; i++) {
                if (i) out += ",";
                out += stringify(node[i]) || "null";
            }
            return out + "]";
        }

        if (node === null) return "null";

        if (seen.indexOf(node) !== -1) {
            if (cycles) return JSON.stringify("__cycle__");
            throw TypeError("Converting circular structure to JSON");
        }

        var seenIndex = seen.push(node) - 1;
        var keys = Object.keys(node).sort(cmp && cmp(node));
        out = "";
        for (i = 0; i < keys.length; i++) {
            var key = keys[i];
            var value = stringify(node[key]);
            if (!value) continue;
            if (out) out += ",";
            out += JSON.stringify(key) + ":" + value;
        }
        seen.splice(seenIndex, 1);
        return "{" + out + "}";
    })(data);
}
