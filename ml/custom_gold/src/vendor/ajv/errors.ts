import resolver from './resolver.js';

export class ValidationError extends Error {
    errors: any;
    ajv: boolean;
    validation: boolean;

    constructor(errors: any) {
        super('validation failed');
        this.errors = errors;
        this.ajv = this.validation = true;
    }
}

export class MissingRefError extends Error {
    missingRef: string;
    missingSchema: string;

    constructor(baseId: string, ref: string, message?: string) {
        super(message || MissingRefError.message(baseId, ref));
        this.missingRef = resolver.url(baseId, ref);
        this.missingSchema = resolver.normalizeId(resolver.fullPath(this.missingRef));
    }

    static message(baseId: string, ref: string) {
        return "can't resolve reference " + ref + " from id " + baseId;
    }
}
