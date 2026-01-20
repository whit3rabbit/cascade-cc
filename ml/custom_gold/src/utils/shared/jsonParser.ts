/**
 * Enhanced JSON parser with support for BigInt and prototype pollution protection.
 */

interface JsonParserOptions {
    strict?: boolean;
    storeAsString?: boolean;
    alwaysParseAsBig?: boolean;
    useNativeBigInt?: boolean;
    protoAction?: "error" | "ignore" | "preserve";
    constructorAction?: "error" | "ignore" | "preserve";
}

const PROT_REGEX = /(?:_|\\u005[Ff])(?:_|\\u005[Ff])(?:p|\\u0070)(?:r|\\u0072)(?:o|\\u006[Ff])(?:t|\\u0074)(?:o|\\u006[Ff])(?:_|\\u005[Ff])(?:_|\\u005[Ff])/;
const CONS_REGEX = /(?:c|\\u0063)(?:o|\\u006[Ff])(?:n|\\u006[Ee])(?:s|\\u0073)(?:t|\\u0074)(?:r|\\u0072)(?:u|\\u0075)(?:c|\\u0063)(?:t|\\u0074)(?:o|\\u006[Ff])(?:r|\\u0072)/;

export function createJsonParser(options: JsonParserOptions = {}) {
    const config = {
        strict: options.strict === true,
        storeAsString: options.storeAsString === true,
        alwaysParseAsBig: options.alwaysParseAsBig === true,
        useNativeBigInt: options.useNativeBigInt === true,
        protoAction: options.protoAction || "error",
        constructorAction: options.constructorAction || "error"
    };

    const escapeChars: Record<string, string> = {
        '"': '"',
        "\\": "\\",
        "/": "/",
        b: "\b",
        f: "\f",
        n: "\n",
        r: "\r",
        t: "\t"
    };

    let at: number;
    let ch: string;
    let textRef: string;

    const error = (m: string) => {
        throw {
            name: "SyntaxError",
            message: m,
            at: at,
            text: textRef
        };
    };

    const next = (c?: string) => {
        if (c && c !== ch) {
            error(`Expected '${c}' instead of '${ch}'`);
        }
        ch = textRef.charAt(at);
        at += 1;
        return ch;
    };

    const parseNumber = () => {
        let value: any;
        let string = "";

        if (ch === "-") {
            string = "-";
            next("-");
        }
        while (ch >= "0" && ch <= "9") {
            string += ch;
            next();
        }
        if (ch === ".") {
            string += ".";
            while (next() && ch >= "0" && ch <= "9") {
                string += ch;
            }
        }
        if (ch === "e" || ch === "E") {
            string += ch;
            next();
            if ((ch as string) === "-" || (ch as string) === "+") {
                string += ch;
                next();
            }
            while (ch >= "0" && ch <= "9") {
                string += ch;
                next();
            }
        }

        value = +string;
        if (!isFinite(value)) {
            error("Bad number");
        } else {
            if (string.length > 15) {
                if (config.storeAsString) return string;
                if (config.useNativeBigInt) return BigInt(string);
                return string; // Fallback or use a Decimal lib if available
            }
            if (config.alwaysParseAsBig) {
                return config.useNativeBigInt ? BigInt(value) : string;
            }
            return value;
        }
    };

    const parseString = () => {
        let hex: number;
        let i: number;
        let string = "";
        let uffff: number;

        if (ch === '"') {
            let start = at;
            while (next()) {
                if (ch === '"') {
                    if (at - 1 > start) {
                        string += textRef.substring(start, at - 1);
                    }
                    next();
                    return string;
                }
                if (ch === "\\") {
                    if (at - 1 > start) {
                        string += textRef.substring(start, at - 1);
                    }
                    next();
                    if (ch === "u") {
                        uffff = 0;
                        for (i = 0; i < 4; i += 1) {
                            hex = parseInt(next(), 16);
                            if (!isFinite(hex)) break;
                            uffff = uffff * 16 + hex;
                        }
                        string += String.fromCharCode(uffff);
                    } else if (typeof escapeChars[ch] === "string") {
                        string += escapeChars[ch];
                    } else {
                        break;
                    }
                    start = at;
                }
            }
        }
        error("Bad string");
    };

    const white = () => {
        while (ch && ch <= " ") {
            next();
        }
    };

    const word = () => {
        switch (ch) {
            case "t":
                next("t"); next("r"); next("u"); next("e");
                return true;
            case "f":
                next("f"); next("a"); next("l"); next("s"); next("e");
                return false;
            case "n":
                next("n"); next("u"); next("l"); next("l");
                return null;
        }
        error(`Unexpected '${ch}'`);
    };

    let value: () => any;

    const parseArray = () => {
        const array: any[] = [];
        if (ch === "[") {
            next("[");
            white();
            if ((ch as string) === "]") {
                next("]");
                return array;
            }
            while (ch) {
                array.push(value());
                white();
                if ((ch as string) === "]") {
                    next("]");
                    return array;
                }
                next(",");
                white();
            }
        }
        error("Bad array");
    };

    const parseObject = () => {
        let key: string;
        const object = Object.create(null);

        if (ch === "{") {
            next("{");
            white();
            if ((ch as string) === "}") {
                next("}");
                return object;
            }
            while (ch) {
                key = parseString() as string;
                white();
                next(":");

                if (config.strict && Object.prototype.hasOwnProperty.call(object, key)) {
                    error(`Duplicate key "${key}"`);
                }

                if (PROT_REGEX.test(key)) {
                    if (config.protoAction === "error") {
                        error("Object contains forbidden prototype property");
                    } else if (config.protoAction === "ignore") {
                        value();
                    } else {
                        object[key] = value();
                    }
                } else if (CONS_REGEX.test(key)) {
                    if (config.constructorAction === "error") {
                        error("Object contains forbidden constructor property");
                    } else if (config.constructorAction === "ignore") {
                        value();
                    } else {
                        object[key] = value();
                    }
                } else {
                    object[key] = value();
                }

                white();
                if ((ch as string) === "}") {
                    next("}");
                    return object;
                }
                next(",");
                white();
            }
        }
        error("Bad object");
    };

    value = () => {
        white();
        switch (ch) {
            case "{": return parseObject();
            case "[": return parseArray();
            case '"': return parseString();
            case "-": return parseNumber();
            default:
                return ch >= "0" && ch <= "9" ? parseNumber() : word();
        }
    };

    return (source: string, reviver?: (this: any, key: string, value: any) => any) => {
        let result: any;
        textRef = String(source);
        at = 0;
        ch = " ";
        result = value();
        white();
        if (ch) {
            error("Syntax error");
        }

        if (typeof reviver === "function") {
            const walk = (holder: any, key: string): any => {
                let v = holder[key];
                if (v && typeof v === "object") {
                    Object.keys(v).forEach((k) => {
                        const res = walk(v, k);
                        if (res !== undefined) {
                            v[k] = res;
                        } else {
                            delete v[k];
                        }
                    });
                }
                return reviver.call(holder, key, v);
            };
            return walk({ "": result }, "");
        }
        return result;
    };
}

export const jsonParse = createJsonParser();
export const jsonStringify = JSON.stringify;

export default {
    parse: jsonParse,
    stringify: jsonStringify
};
