export const allOf = (it: any, keyword: string): string => {
    let out = ' ';
    const schema = it.schema[keyword];
    const schemaPath = it.schemaPath + it.util.getProperty(keyword);
    const errSchemaPath = it.errSchemaPath + '/' + keyword;
    const allErrors = !it.opts.allErrors;
    const itCopy = it.util.copy(it);
    let closing = '';
    itCopy.level++;
    const valid = 'valid' + itCopy.level;
    const baseId = itCopy.baseId;
    let empty = true;

    if (schema) {
        schema.forEach((sch: any, i: number) => {
            if (it.opts.strictKeywords ? typeof sch === 'object' && Object.keys(sch).length > 0 || sch === false : it.util.schemaHasRules(sch, it.RULES.all)) {
                empty = false;
                itCopy.schema = sch;
                itCopy.schemaPath = schemaPath + '[' + i + ']';
                itCopy.errSchemaPath = errSchemaPath + '/' + i;
                out += '  ' + it.validate(itCopy) + ' ';
                itCopy.baseId = baseId;
                if (allErrors) {
                    out += ' if (' + valid + ') { ';
                    closing += '}';
                }
            }
        });
    }

    if (allErrors) {
        if (empty) out += ' if (true) { ';
        else out += ' ' + closing.slice(0, -1) + ' ';
    }
    return out;
};

export const anyOf = (it: any, keyword: string): string => {
    let out = ' ';
    const level = it.level;
    const dataLevel = it.dataLevel;
    const schema = it.schema[keyword];
    const schemaPath = it.schemaPath + it.util.getProperty(keyword);
    const errSchemaPath = it.errSchemaPath + '/' + keyword;
    const allErrors = !it.opts.allErrors;
    const data = 'data' + (dataLevel || '');
    const valid = 'valid' + level;
    const errs = 'errs__' + level;
    const itCopy = it.util.copy(it);
    let closing = '';
    itCopy.level++;
    const nextValid = 'valid' + itCopy.level;
    const hasRules = schema.every((sch: any) => {
        return it.opts.strictKeywords ? typeof sch === 'object' && Object.keys(sch).length > 0 || sch === false : it.util.schemaHasRules(sch, it.RULES.all);
    });

    if (hasRules) {
        const baseId = itCopy.baseId;
        out += ' var ' + errs + ' = errors; var ' + valid + ' = false; ';
        const compositeRule = it.compositeRule;
        it.compositeRule = itCopy.compositeRule = true;

        schema.forEach((sch: any, i: number) => {
            itCopy.schema = sch;
            itCopy.schemaPath = schemaPath + '[' + i + ']';
            itCopy.errSchemaPath = errSchemaPath + '/' + i;
            out += '  ' + it.validate(itCopy) + ' ';
            itCopy.baseId = baseId;
            out += ' ' + valid + ' = ' + valid + ' || ' + nextValid + '; if (!' + valid + ') { ';
            closing += '}';
        });

        it.compositeRule = itCopy.compositeRule = compositeRule;
        out += ' ' + closing + ' if (!' + valid + ') { ';

        const err = generateError(it, 'anyOf', '', 'should match some schema in anyOf', schemaPath, data);
        out += ' var err = ' + err + '; if (vErrors === null) vErrors = [err]; else vErrors.push(err); errors++; ';

        if (!it.compositeRule && allErrors) {
            if (it.async) out += ' throw new ValidationError(vErrors); ';
            else out += ' validate.errors = vErrors; return false; ';
        }

        out += ' } else { errors = ' + errs + '; if (vErrors !== null) { if (' + errs + ') vErrors.length = ' + errs + '; else vErrors = null; } ';
        if (it.opts.allErrors) out += ' } ';
    } else if (allErrors) {
        out += ' if (true) { ';
    }

    return out;
};

export const $comment = (it: any, keyword: string): string => {
    let out = ' ';
    const schema = it.schema[keyword];
    const errSchemaPath = it.errSchemaPath + '/' + keyword;
    const val = it.util.toQuotedString(schema);
    if (it.opts.$comment === true) {
        out += ' console.log(' + val + ');';
    } else if (typeof it.opts.$comment === 'function') {
        out += ' self._opts.$comment(' + val + ', ' + it.util.toQuotedString(errSchemaPath) + ', validate.root.schema);';
    }
    return out;
};

export const constant = (it: any, keyword: string): string => {
    let out = ' ';
    const level = it.level;
    const dataLevel = it.dataLevel;
    const schema = it.schema[keyword];
    const schemaPath = it.schemaPath + it.util.getProperty(keyword);
    const errSchemaPath = it.errSchemaPath + '/' + keyword;
    const allErrors = !it.opts.allErrors;
    const data = 'data' + (dataLevel || '');
    const valid = 'valid' + level;
    const dataVar = it.opts.$data && schema && schema.$data;

    let schemaVar;
    if (dataVar) {
        out += ' var schema' + level + ' = ' + it.util.getData(schema.$data, dataLevel, it.dataPathArr) + '; ';
        schemaVar = 'schema' + level;
    } else {
        schemaVar = 'schema' + level;
        out += ' var ' + schemaVar + ' = validate.schema' + schemaPath + ';';
    }

    out += ' var ' + valid + ' = equal(' + data + ', ' + schemaVar + '); if (!' + valid + ') { ';

    const err = generateError(it, 'const', '', 'should be equal to constant', schemaPath, data, { allowedValue: schemaVar });
    if (!it.compositeRule && allErrors) {
        if (it.async) out += ' throw new ValidationError([' + err + ']); ';
        else out += ' validate.errors = [' + err + ']; return false; ';
    } else {
        out += ' var err = ' + err + '; if (vErrors === null) vErrors = [err]; else vErrors.push(err); errors++; ';
    }
    out += ' }';
    if (allErrors) out += ' else { ';
    return out;
};

export const contains = (it: any, keyword: string): string => {
    let out = ' ';
    const level = it.level;
    const dataLevel = it.dataLevel;
    const schema = it.schema[keyword];
    const schemaPath = it.schemaPath + it.util.getProperty(keyword);
    const errSchemaPath = it.errSchemaPath + '/' + keyword;
    const allErrors = !it.opts.allErrors;
    const data = 'data' + (dataLevel || '');
    const valid = 'valid' + level;
    const errs = 'errs__' + level;
    const itCopy = it.util.copy(it);
    itCopy.level++;
    const nextValid = 'valid' + itCopy.level;
    const i = 'i' + level;
    const nextDataLevel = itCopy.dataLevel = it.dataLevel + 1;
    const nextData = 'data' + nextDataLevel;
    const baseId = it.baseId;
    const hasRules = it.opts.strictKeywords ? typeof schema === 'object' && Object.keys(schema).length > 0 || schema === false : it.util.schemaHasRules(schema, it.RULES.all);

    out += ' var ' + errs + ' = errors; var ' + valid + ';';
    if (hasRules) {
        const compositeRule = it.compositeRule;
        it.compositeRule = itCopy.compositeRule = true;
        itCopy.schema = schema;
        itCopy.schemaPath = schemaPath;
        itCopy.errSchemaPath = errSchemaPath;

        out += ' var ' + nextValid + ' = false; for (var ' + i + ' = 0; ' + i + ' < ' + data + '.length; ' + i + '++) { ';
        itCopy.errorPath = it.util.getPathExpr(it.errorPath, i, it.opts.jsonPointers, true);
        const itemData = data + '[' + i + ']';
        itCopy.dataPathArr[nextDataLevel] = i;
        const validate = it.validate(itCopy);
        itCopy.baseId = baseId;
        if (it.util.varOccurences(validate, nextData) < 2) {
            out += ' ' + it.util.varReplace(validate, nextData, itemData) + ' ';
        } else {
            out += ' var ' + nextData + ' = ' + itemData + '; ' + validate + ' ';
        }
        out += ' if (' + nextValid + ') break; } ';
        it.compositeRule = itCopy.compositeRule = compositeRule;
        out += ' if (!' + nextValid + ') {';
    } else {
        out += ' if (' + data + '.length == 0) {';
    }

    const err = generateError(it, 'contains', '', 'should contain a valid item', schemaPath, data);
    if (!it.compositeRule && allErrors) {
        if (it.async) out += ' throw new ValidationError([' + err + ']); ';
        else out += ' validate.errors = [' + err + ']; return false; ';
    } else {
        out += ' var err = ' + err + '; if (vErrors === null) vErrors = [err]; else vErrors.push(err); errors++; ';
    }

    out += ' } else { ';
    if (hasRules) {
        out += ' errors = ' + errs + '; if (vErrors !== null) { if (' + errs + ') vErrors.length = ' + errs + '; else vErrors = null; } ';
    }
    if (it.opts.allErrors) out += ' } ';

    return out;
};

export const dependencies = (it: any, keyword: string): string => {
    let out = ' ';
    const level = it.level;
    const dataLevel = it.dataLevel;
    const schema = it.schema[keyword];
    const schemaPath = it.schemaPath + it.util.getProperty(keyword);
    const errSchemaPath = it.errSchemaPath + '/' + keyword;
    const allErrors = !it.opts.allErrors;
    const data = 'data' + (dataLevel || '');
    const errs = 'errs__' + level;
    const itCopy = it.util.copy(it);
    itCopy.level++;
    const nextValid = 'valid' + itCopy.level;
    const ownProperties = it.opts.ownProperties;

    const propDeps: any = {};
    const schemaDeps: any = {};
    for (const key in schema) {
        if (key === '__proto__') continue;
        const dep = schema[key];
        const target = Array.isArray(dep) ? propDeps : schemaDeps;
        target[key] = dep;
    }

    out += ' var ' + errs + ' = errors;';
    const errorPath = it.errorPath;
    out += ' var missing' + level + ';';

    for (const key in propDeps) {
        const deps = propDeps[key];
        if (deps.length) {
            out += ' if (' + data + it.util.getProperty(key) + ' !== undefined ';
            if (ownProperties) out += ' && Object.prototype.hasOwnProperty.call(' + data + ", '" + it.util.escapeQuotes(key) + "') ";

            if (allErrors) {
                out += ' && ( ';
                deps.forEach((dep: string, i: number) => {
                    if (i) out += ' || ';
                    const depProp = it.util.getProperty(dep);
                    const depData = data + depProp;
                    out += ' ( ( ' + depData + ' === undefined ';
                    if (ownProperties) out += ' || !Object.prototype.hasOwnProperty.call(' + data + ", '" + it.util.escapeQuotes(dep) + "') ";
                    out += ') && (missing' + level + ' = ' + it.util.toQuotedString(it.opts.jsonPointers ? dep : depProp) + ') ) ';
                });
                out += ')) { ';

                const missing = 'missing' + level;
                if (it.opts._errorDataPathProperty) {
                    it.errorPath = it.opts.jsonPointers ? it.util.getPathExpr(errorPath, missing, true) : errorPath + ' + ' + missing;
                }

                const params = {
                    property: key,
                    missingProperty: "' + " + missing + " + '",
                    depsCount: deps.length,
                    deps: it.util.escapeQuotes(deps.length === 1 ? deps[0] : deps.join(', '))
                };
                const msg = 'should have ' + (deps.length === 1 ? 'property ' + it.util.escapeQuotes(deps[0]) : 'properties ' + it.util.escapeQuotes(deps.join(', '))) + ' when property ' + it.util.escapeQuotes(key) + ' is present';
                const err = generateError(it, 'dependencies', '', msg, schemaPath, data, params);

                if (!it.compositeRule && allErrors) {
                    if (it.async) out += ' throw new ValidationError([' + err + ']); ';
                    else out += ' validate.errors = [' + err + ']; return false; ';
                } else {
                    out += ' var err = ' + err + '; if (vErrors === null) vErrors = [err]; else vErrors.push(err); errors++; ';
                }
            } else {
                out += ' ) { ';
                deps.forEach((dep: string) => {
                    const depProp = it.util.getProperty(dep);
                    const depData = data + depProp;
                    if (it.opts._errorDataPathProperty) {
                        it.errorPath = it.util.getPath(errorPath, dep, it.opts.jsonPointers);
                    }
                    out += ' if (' + depData + ' === undefined ';
                    if (ownProperties) out += ' || !Object.prototype.hasOwnProperty.call(' + data + ", '" + it.util.escapeQuotes(dep) + "') ";
                    out += ') { ';
                    const params = {
                        property: key,
                        missingProperty: it.util.escapeQuotes(dep),
                        depsCount: deps.length,
                        deps: it.util.escapeQuotes(deps.length === 1 ? deps[0] : deps.join(', '))
                    };
                    const msg = 'should have ' + (deps.length === 1 ? 'property ' + it.util.escapeQuotes(deps[0]) : 'properties ' + it.util.escapeQuotes(deps.join(', '))) + ' when property ' + it.util.escapeQuotes(key) + ' is present';
                    const err = generateError(it, 'dependencies', '', msg, schemaPath, data, params);
                    out += ' var err = ' + err + '; if (vErrors === null) vErrors = [err]; else vErrors.push(err); errors++; } ';
                });
            }
            out += ' } ';
            // Handle the else part for non-allErrors if needed, but the original logic is a bit convoluted here
        }
    }

    it.errorPath = errorPath;
    const baseId = itCopy.baseId;
    for (const key in schemaDeps) {
        const sch = schemaDeps[key];
        if (it.opts.strictKeywords ? typeof sch === 'object' && Object.keys(sch).length > 0 || sch === false : it.util.schemaHasRules(sch, it.RULES.all)) {
            out += ' ' + nextValid + ' = true; if (' + data + it.util.getProperty(key) + ' !== undefined ';
            if (ownProperties) out += ' && Object.prototype.hasOwnProperty.call(' + data + ", '" + it.util.escapeQuotes(key) + "') ";
            out += ') { ';
            itCopy.schema = sch;
            itCopy.schemaPath = schemaPath + it.util.getProperty(key);
            itCopy.errSchemaPath = errSchemaPath + '/' + it.util.escapeFragment(key);
            out += ' ' + it.validate(itCopy) + ' ';
            itCopy.baseId = baseId;
            out += ' } ';
            if (allErrors) out += ' if (' + nextValid + ') { ';
        }
    }

    if (allErrors) {
        // Closing brackets handled by code generation loop
        out += ' if (' + errs + ' == errors) {';
    }

    return out;
};

export const enumKeyword = (it: any, keyword: string): string => {
    let out = ' ';
    const level = it.level;
    const dataLevel = it.dataLevel;
    const schema = it.schema[keyword];
    const schemaPath = it.schemaPath + it.util.getProperty(keyword);
    const errSchemaPath = it.errSchemaPath + '/' + keyword;
    const allErrors = !it.opts.allErrors;
    const data = 'data' + (dataLevel || '');
    const valid = 'valid' + level;
    const dataVar = it.opts.$data && schema && schema.$data;

    let schemaVar = 'schema' + level;
    if (dataVar) {
        out += ' var ' + schemaVar + ' = ' + it.util.getData(schema.$data, dataLevel, it.dataPathArr) + '; ';
    } else {
        out += ' var ' + schemaVar + ' = validate.schema' + schemaPath + ';';
    }

    out += ' var ' + valid + ';';
    if (dataVar) out += ' if (' + schemaVar + ' === undefined) ' + valid + ' = true; else if (!Array.isArray(' + schemaVar + ')) ' + valid + ' = false; else {';

    const i = 'i' + level;
    out += ' ' + valid + ' = false; for (var ' + i + ' = 0; ' + i + ' < ' + schemaVar + '.length; ' + i + '++) if (equal(' + data + ', ' + schemaVar + '[' + i + '])) { ' + valid + ' = true; break; }';

    if (dataVar) out += ' } ';
    out += ' if (!' + valid + ') { ';

    const err = generateError(it, 'enum', '', 'should be equal to one of the allowed values', schemaPath, data, { allowedValues: schemaVar });
    if (!it.compositeRule && allErrors) {
        if (it.async) out += ' throw new ValidationError([' + err + ']); ';
        else out += ' validate.errors = [' + err + ']; return false; ';
    } else {
        out += ' var err = ' + err + '; if (vErrors === null) vErrors = [err]; else vErrors.push(err); errors++; ';
    }
    out += ' }';
    if (allErrors) out += ' else { ';

    return out;
};

export const format = (it: any, keyword: string, type?: string): string => {
    let out = ' ';
    const level = it.level;
    const dataLevel = it.dataLevel;
    const schema = it.schema[keyword];
    const schemaPath = it.schemaPath + it.util.getProperty(keyword);
    const errSchemaPath = it.errSchemaPath + '/' + keyword;
    const allErrors = !it.opts.allErrors;
    const data = 'data' + (dataLevel || '');

    if (it.opts.format === false) {
        if (allErrors) out += ' if (true) { ';
        return out;
    }

    const dataVar = it.opts.$data && schema && schema.$data;
    let schemaVar;
    if (dataVar) {
        out += ' var schema' + level + ' = ' + it.util.getData(schema.$data, dataLevel, it.dataPathArr) + '; ';
        schemaVar = 'schema' + level;
    } else {
        schemaVar = schema;
    }

    const unknownFormats = it.opts.unknownFormats;
    const isUnknownArray = Array.isArray(unknownFormats);

    if (dataVar) {
        const formatVar = 'format' + level;
        const isObject = 'isObject' + level;
        const formatType = 'formatType' + level;
        out += ' var ' + formatVar + ' = formats[' + schemaVar + ']; var ' + isObject + ' = typeof ' + formatVar + " == 'object' && !(" + formatVar + ' instanceof RegExp) && ' + formatVar + '.validate; var ' + formatType + ' = ' + isObject + ' && ' + formatVar + ".type || 'string'; if (" + isObject + ') { ';
        if (it.async) out += ' var async' + level + ' = ' + formatVar + '.async; ';
        out += ' ' + formatVar + ' = ' + formatVar + '.validate; } if ( ';
        out += ' (' + schemaVar + " !== undefined && typeof " + schemaVar + " != 'string') || ";
        out += ' (';
        if (unknownFormats !== 'ignore') {
            out += ' (' + schemaVar + ' && !' + formatVar;
            if (isUnknownArray) out += ' && self._opts.unknownFormats.indexOf(' + schemaVar + ') == -1 ';
            out += ') || ';
        }
        out += ' (' + formatVar + ' && ' + formatType + " == '" + type + "' && !(typeof " + formatVar + " == 'function' ? ";
        if (it.async) out += ' (async' + level + ' ? await ' + formatVar + '(' + data + ') : ' + formatVar + '(' + data + ')) ';
        else out += ' ' + formatVar + '(' + data + ') ';
        out += ' : ' + formatVar + '.test(' + data + '))))) {';
    } else {
        const formatDef = it.formats[schema];
        if (!formatDef) {
            if (unknownFormats === 'ignore') {
                it.logger.warn('unknown format "' + schema + '" ignored in schema at path "' + it.errSchemaPath + '"');
                if (allErrors) out += ' if (true) { ';
                return out;
            } else if (isUnknownArray && unknownFormats.indexOf(schema) >= 0) {
                if (allErrors) out += ' if (true) { ';
                return out;
            } else {
                throw new Error('unknown format "' + schema + '" is used in schema at path "' + it.errSchemaPath + '"');
            }
        }
        const isObject = typeof formatDef === 'object' && !(formatDef instanceof RegExp) && formatDef.validate;
        const formatType = isObject && formatDef.type || 'string';
        let async = false;
        let validate = formatDef;
        if (isObject) {
            async = formatDef.async === true;
            validate = formatDef.validate;
        }

        if (formatType !== type) {
            if (allErrors) out += ' if (true) { ';
            return out;
        }

        if (async) {
            if (!it.async) throw new Error('async format in sync schema');
            const method = 'formats' + it.util.getProperty(schema) + '.validate';
            out += ' if (!(await ' + method + '(' + data + '))) { ';
        } else {
            out += ' if (! ';
            let method = 'formats' + it.util.getProperty(schema);
            if (isObject) method += '.validate';
            if (typeof validate === 'function') out += ' ' + method + '(' + data + ') ';
            else out += ' ' + method + '.test(' + data + ') ';
            out += ') { ';
        }
    }

    const err = generateError(it, 'format', '', 'should match format "' + (dataVar ? "' + " + schemaVar + " + '" : it.util.escapeQuotes(schema)) + '"', schemaPath, data, { format: dataVar ? schemaVar : it.util.toQuotedString(schema) });
    if (!it.compositeRule && allErrors) {
        if (it.async) out += ' throw new ValidationError([' + err + ']); ';
        else out += ' validate.errors = [' + err + ']; return false; ';
    } else {
        out += ' var err = ' + err + '; if (vErrors === null) vErrors = [err]; else vErrors.push(err); errors++; ';
    }

    out += ' } ';
    if (allErrors) out += ' else { ';

    return out;
};

function generateError(it: any, keyword: string, schemaPath: string, message: string, fullSchemaPath: string, data: string, params: any = {}): string {
    let out = ' { keyword: \'' + keyword + '\' , dataPath: (dataPath || \'\') + ' + it.errorPath + ' , schemaPath: ' + it.util.toQuotedString(it.errSchemaPath + schemaPath) + ' , params: ' + JSON.stringify(params);
    if (it.opts.messages !== false) out += ' , message: \'' + message.replace(/'/g, "\\'") + '\' ';
    if (it.opts.verbose) {
        out += ' , schema: validate.schema' + fullSchemaPath + ' , parentSchema: validate.schema' + it.schemaPath + ' , data: ' + data + ' ';
    }
    out += ' } ';
    return out;
}
