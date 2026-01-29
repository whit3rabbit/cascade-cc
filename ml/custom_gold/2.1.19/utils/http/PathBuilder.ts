/**
 * File: src/utils/http/PathBuilder.ts
 * Role: Template tag for building URL paths with encoded parameters reliably.
 */

import { encodeHeaderValue } from './HeaderUtils.js';

export interface PathError {
    start: number;
    length: number;
    error: string;
}

/**
 * Creates a path builder function using the specified encoding function.
 */
export function createPathBuilder(encodeFn: (val: string) => string = encodeHeaderValue) {
    return (pathTemplate: TemplateStringsArray, ...params: any[]): string => {
        if (pathTemplate.length === 1) {
            return pathTemplate[0];
        }

        let hasQueryOrHash = false;
        const errors: PathError[] = [];

        const result = pathTemplate.reduce((acc, part, index) => {
            if (/[?#]/.test(part)) {
                hasQueryOrHash = true;
            }

            if (index === params.length) {
                return acc + part;
            }

            const param = params[index];
            let encoded = (hasQueryOrHash ? encodeURIComponent : encodeFn)("" + param);

            // Validation logic for objects or nulls in path
            if (param == null || (typeof param === "object" && param.toString === Object.prototype.toString)) {
                encoded = String(param);
                errors.push({
                    start: acc.length + part.length,
                    length: encoded.length,
                    error: `Value of type ${Object.prototype.toString.call(param).slice(8, -1)} is not a valid path parameter`
                });
            }

            return acc + part + encoded;
        }, "");

        // Find and report path parameter errors (like .. or .)
        const pathSegmentErrorsRegex = /(?<=^|\/)(?:\.|%2e){1,2}(?=\/|$)/gi;
        let match: RegExpExecArray | null;
        while ((match = pathSegmentErrorsRegex.exec(result)) !== null) {
            errors.push({
                start: match.index,
                length: match[0].length,
                error: `Value "${match[0]}" can't be safely passed as a path parameter`
            });
        }

        if (errors.length > 0) {
            throw new Error(`Path parameters result in invalid segments: \n${errors.map(e => e.error).join('\n')}\n${result}`);
        }

        return result;
    };
}

/**
 * Default path builder implementation.
 */
export const path = createPathBuilder();
