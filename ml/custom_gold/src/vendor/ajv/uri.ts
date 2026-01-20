const SCHEMES: any = {};

function pctEncChar(c: string): string {
    const n = c.charCodeAt(0);
    if (n < 16) return "%0" + n.toString(16).toUpperCase();
    if (n < 128) return "%" + n.toString(16).toUpperCase();
    if (n < 2048) return "%" + (n >> 6 | 192).toString(16).toUpperCase() + "%" + (n & 63 | 128).toString(16).toUpperCase();
    return "%" + (n >> 12 | 224).toString(16).toUpperCase() + "%" + (n >> 6 & 63 | 128).toString(16).toUpperCase() + "%" + (n & 63 | 128).toString(16).toUpperCase();
}

function pctDecChars(str: string): string {
    let res = "";
    let i = 0;
    while (i < str.length) {
        const c = parseInt(str.substr(i + 1, 2), 16);
        if (c < 128) {
            res += String.fromCharCode(c);
            i += 3;
        } else if (c >= 194 && c < 224) {
            if (str.length - i >= 6) {
                const c2 = parseInt(str.substr(i + 4, 2), 16);
                res += String.fromCharCode((c & 31) << 6 | (c2 & 63));
            } else res += str.substr(i, 6);
            i += 6;
        } else if (c >= 224) {
            if (str.length - i >= 9) {
                const c2 = parseInt(str.substr(i + 4, 2), 16);
                const c3 = parseInt(str.substr(i + 7, 2), 16);
                res += String.fromCharCode((c & 15) << 12 | (c2 & 63) << 6 | (c3 & 63));
            } else res += str.substr(i, 9);
            i += 9;
        } else {
            res += str.substr(i, 3);
            i += 3;
        }
    }
    return res;
}

const parseRegExp = /^(?:([^:\/?#]+):)?(?:\/\/((?:([^\/?#@]*)@)?(\[[^\/?#\]]+\]|[^\/?#:]*)(?:\:(\d*))?))?([^?#]*)(?:\?([^#]*))?(?:#((?:.|\n|\r)*))?/i;

export function parse(uri: string, opts: any = {}): any {
    const res: any = {};
    const match = uri.match(parseRegExp);
    if (match) {
        res.scheme = match[1];
        res.userinfo = match[3];
        res.host = match[4];
        res.port = parseInt(match[5], 10);
        res.path = match[6] || "";
        res.query = match[7];
        res.fragment = match[8];
        if (isNaN(res.port)) res.port = match[5];

        if (res.scheme === undefined && res.userinfo === undefined && res.host === undefined && res.port === undefined && !res.path && res.query === undefined) {
            res.reference = "same-document";
        } else if (res.scheme === undefined) {
            res.reference = "relative";
        } else if (res.fragment === undefined) {
            res.reference = "absolute";
        } else {
            res.reference = "uri";
        }
    }
    return res;
}

export function serialize(obj: any, opts: any = {}): string {
    const res: string[] = [];
    if (obj.scheme) {
        res.push(obj.scheme);
        res.push(":");
    }
    if (obj.host !== undefined || obj.userinfo !== undefined || obj.port !== undefined) {
        res.push("//");
        if (obj.userinfo !== undefined) {
            res.push(obj.userinfo);
            res.push("@");
        }
        if (obj.host !== undefined) {
            res.push(obj.host);
        }
        if (obj.port !== undefined && obj.port !== "") {
            res.push(":");
            res.push(String(obj.port));
        }
    }
    if (obj.path !== undefined) {
        let path = obj.path;
        if (obj.host !== undefined && path && path.charAt(0) !== "/") {
            res.push("/");
        }
        res.push(removeDotSegments(path));
    }
    if (obj.query !== undefined) {
        res.push("?");
        res.push(obj.query);
    }
    if (obj.fragment !== undefined) {
        res.push("#");
        res.push(obj.fragment);
    }
    return res.join("");
}

export function resolve(base: string, relative: string, opts: any = {}): string {
    const baseObj = parse(base, opts);
    const relObj = parse(relative, opts);
    return serialize(resolveComponents(baseObj, relObj, opts, true), opts);
}

function resolveComponents(base: any, rel: any, opts: any, skipNormalization?: boolean): any {
    const res: any = {};
    if (!rel.scheme) {
        res.scheme = base.scheme;
        if (rel.userinfo !== undefined || rel.host !== undefined || rel.port !== undefined) {
            res.userinfo = rel.userinfo;
            res.host = rel.host;
            res.port = rel.port;
            res.path = removeDotSegments(rel.path || "");
            res.query = rel.query;
        } else {
            if (!rel.path) {
                res.path = base.path;
                res.query = rel.query !== undefined ? rel.query : base.query;
            } else {
                if (rel.path.charAt(0) === "/") {
                    res.path = removeDotSegments(rel.path);
                } else {
                    if ((base.userinfo !== undefined || base.host !== undefined || base.port !== undefined) && !base.path) {
                        res.path = "/" + rel.path;
                    } else if (!base.path) {
                        res.path = rel.path;
                    } else {
                        res.path = base.path.slice(0, base.path.lastIndexOf("/") + 1) + rel.path;
                    }
                    res.path = removeDotSegments(res.path);
                }
                res.query = rel.query;
            }
            res.userinfo = base.userinfo;
            res.host = base.host;
            res.port = base.port;
        }
    } else {
        res.scheme = rel.scheme;
        res.userinfo = rel.userinfo;
        res.host = rel.host;
        res.port = rel.port;
        res.path = removeDotSegments(rel.path || "");
        res.query = rel.query;
    }
    res.fragment = rel.fragment;
    return res;
}

function removeDotSegments(path: string): string {
    const segments: string[] = [];
    while (path.length) {
        if (path.startsWith("./")) {
            path = path.slice(2);
        } else if (path.startsWith("../")) {
            path = path.slice(3);
        } else if (path.startsWith("/./")) {
            path = "/" + path.slice(3);
        } else if (path === "/.") {
            path = "/";
        } else if (path.startsWith("/../")) {
            path = "/" + path.slice(4);
            segments.pop();
        } else if (path === "/..") {
            path = "/";
            segments.pop();
        } else if (path === "." || path === "..") {
            path = "";
        } else {
            const match = path.match(/^\/?(?:.|\n|\r)*?(?=\/|$)/);
            if (match) {
                const seg = match[0];
                path = path.slice(seg.length);
                segments.push(seg);
            } else break;
        }
    }
    return segments.join("");
}
