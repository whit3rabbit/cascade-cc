import { ZodErrorMap } from "zod";

const jsonStringifyReplacer = (_: string, value: any): any => {
    if (typeof value === "bigint") {
        return value.toString();
    }
    return value;
};

function joinValues(array: any[], separator = " | "): string {
    return array.map((val) => JSON.stringify(val, jsonStringifyReplacer)).join(separator);
}

function getType(value: any): string {
    const type = typeof value;
    switch (type) {
        case "number":
            return Number.isNaN(value) ? "NaN" : "number";
        case "object":
            if (Array.isArray(value)) return "array";
            if (value === null) return "null";
            if (Object.getPrototypeOf(value) !== Object.prototype && value.constructor) {
                return value.constructor.name;
            }
            return "object";
        default:
            return type;
    }
}

export const customZodErrorMap = (issue: any, ctx: any): { message: string } => {
    let message: string;

    const widthMap: Record<string, { unit: string; verb: string }> = {
        string: { unit: "characters", verb: "to have" },
        file: { unit: "bytes", verb: "to have" },
        array: { unit: "items", verb: "to have" },
        set: { unit: "items", verb: "to have" }
    };

    const formatMap: Record<string, string> = {
        regex: "input",
        email: "email address",
        url: "URL",
        emoji: "emoji",
        uuid: "UUID",
        uuidv4: "UUIDv4",
        uuidv6: "UUIDv6",
        nanoid: "nanoid",
        guid: "GUID",
        cuid: "cuid",
        cuid2: "cuid2",
        ulid: "ULID",
        xid: "XID",
        ksuid: "KSUID",
        datetime: "ISO datetime",
        date: "ISO date",
        time: "ISO time",
        duration: "ISO duration",
        ipv4: "IPv4 address",
        ipv6: "IPv6 address",
        cidrv4: "IPv4 range",
        cidrv6: "IPv6 range",
        base64: "base64-encoded string",
        base64url: "base64url-encoded string",
        json_string: "JSON string",
        e164: "E.164 number",
        jwt: "JWT",
        template_literal: "input"
    };

    switch (issue.code) {
        case "invalid_type":
            message = `Invalid input: expected ${issue.expected}, received ${getType(ctx.data)}`;
            break;
        case "invalid_value":
            // Originally "invalid_value" handled single value or list (enum)
            if (issue.values && issue.values.length === 1) {
                message = `Invalid input: expected ${JSON.stringify(issue.values[0], jsonStringifyReplacer)}`;
            } else if (issue.values) {
                message = `Invalid option: expected one of ${joinValues(issue.values)}`;
            } else {
                message = "Invalid value";
            }
            break;
        case "invalid_literal": // Keeping this just in case, though chunk didn't explicitly show it
            message = `Invalid input: expected ${JSON.stringify(issue.expected, jsonStringifyReplacer)}`;
            break;
        case "invalid_union":
            message = "Invalid input";
            break;
        case "invalid_union_discriminator":
            message = `Invalid discriminator value. Expected ${joinValues(issue.options)}`;
            break;
        case "unrecognized_keys":
            message = `Unrecognized key${issue.keys.length > 1 ? "s" : ""}: ${issue.keys.join(", ")}`;
            break;
        case "invalid_arguments":
            message = "Invalid function arguments";
            break;
        case "invalid_return_type":
            message = "Invalid function return type";
            break;
        case "invalid_date":
            message = "Invalid date";
            break;
        case "invalid_format":
        case "invalid_string": // Fallback
            const format = issue.format || (typeof issue.validation === "string" ? issue.validation : "video"); // fallback

            if (format === "starts_with" || issue.validation?.startsWith) {
                message = `Invalid string: must start with "${issue.prefix || issue.validation?.startsWith}"`;
            } else if (format === "ends_with" || issue.validation?.endsWith) {
                message = `Invalid string: must end with "${issue.suffix || issue.validation?.endsWith}"`;
            } else if (format === "includes" || issue.validation?.includes) {
                message = `Invalid string: must include "${issue.includes || issue.validation?.includes}"`;
            } else if (format === "regex" || issue.validation === "regex") {
                message = `Invalid string: must match pattern ${issue.pattern || "regex"}`;
            } else {
                message = `Invalid ${formatMap[format] ?? format}`;
            }
            break;
        case "too_small":
            {
                const unit = widthMap[issue.type]?.unit ?? "elements";
                const inclusive = issue.inclusive ? ">=" : ">";
                message = `Too small: expected ${issue.type} to have ${inclusive}${issue.minimum} ${unit}`;
            }
            break;
        case "too_big":
            {
                const unit = widthMap[issue.type]?.unit ?? "elements";
                const inclusive = issue.inclusive ? "<=" : "<";
                message = `Too big: expected ${issue.type} to have ${inclusive}${issue.maximum} ${unit}`;
            }
            break;
        case "custom":
            message = issue.message ?? "Invalid input";
            break;
        case "invalid_intersection_types":
            message = "Intersection results could not be merged";
            break;
        case "not_multiple_of":
            message = `Invalid number: must be a multiple of ${issue.multipleOf || issue.divisor}`;
            break;
        case "invalid_key":
            message = `Invalid key in ${issue.origin}`;
            break;
        case "invalid_element":
            message = `Invalid value in ${issue.origin}`;
            break;
        default:
            message = ctx.defaultError;
            break;
    }
    return { message };
};
