export const not = (it: any, keyword: string): string => {
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

    if (it.opts.strictKeywords ? typeof schema === 'object' && Object.keys(schema).length > 0 || schema === false : it.util.schemaHasRules(schema, it.RULES.all)) {
        itCopy.schema = schema;
        itCopy.schemaPath = schemaPath;
        itCopy.errSchemaPath = errSchemaPath;
        out += ' var ' + errs + ' = errors; ';
        const compositeRule = it.compositeRule;
        it.compositeRule = itCopy.compositeRule = true;
        itCopy.createErrors = false;

        let oldAllErrors;
        if (itCopy.opts.allErrors) {
            oldAllErrors = itCopy.opts.allErrors;
            itCopy.opts.allErrors = false;
        }

        out += ' ' + it.validate(itCopy) + ' ';
        itCopy.createErrors = true;
        if (oldAllErrors) itCopy.opts.allErrors = oldAllErrors;
        it.compositeRule = itCopy.compositeRule = compositeRule;

        out += ' if (' + nextValid + ') { ';
        const err = generateError(it, 'not', '', 'should NOT be valid', schemaPath, data);
        if (!it.compositeRule && allErrors) {
            if (it.async) out += ' throw new ValidationError([' + err + ']); ';
            else out += ' validate.errors = [' + err + ']; return false; ';
        } else {
            out += ' var err = ' + err + '; if (vErrors === null) vErrors = [err]; else vErrors.push(err); errors++; ';
        }
        out += ' } else { errors = ' + errs + '; if (vErrors !== null) { if (' + errs + ') vErrors.length = ' + errs + '; else vErrors = null; } ';
        if (it.opts.allErrors) out += ' } ';
    } else {
        out += ' var err = ';
        const err = generateError(it, 'not', '', 'should NOT be valid', schemaPath, data);
        out += err + '; if (vErrors === null) vErrors = [err]; else vErrors.push(err); errors++; ';
        if (allErrors) out += ' if (false) { ';
    }

    return out;
};

export const oneOf = (it: any, keyword: string): string => {
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
    const prevValid = 'prevValid' + level;
    const passingSchemas = 'passingSchemas' + level;
    let closing = '';

    out += ' var ' + errs + ' = errors, ' + prevValid + ' = false, ' + valid + ' = false, ' + passingSchemas + ' = null; ';
    const compositeRule = it.compositeRule;
    it.compositeRule = itCopy.compositeRule = true;

    schema.forEach((sch: any, i: number) => {
        if (it.opts.strictKeywords ? typeof sch === 'object' && Object.keys(sch).length > 0 || sch === false : it.util.schemaHasRules(sch, it.RULES.all)) {
            itCopy.schema = sch;
            itCopy.schemaPath = schemaPath + '[' + i + ']';
            itCopy.errSchemaPath = errSchemaPath + '/' + i;
            out += ' ' + it.validate(itCopy) + ' ';
        } else {
            out += ' var ' + nextValid + ' = true; ';
        }

        if (i > 0) {
            out += ' if (' + nextValid + ' && ' + prevValid + ') { ' + valid + ' = false; ' + passingSchemas + ' = [' + passingSchemas + ', ' + i + ']; } else { ';
            closing += '}';
        }
        out += ' if (' + nextValid + ') { ' + valid + ' = ' + prevValid + ' = true; ' + passingSchemas + ' = ' + i + '; }';
    });

    it.compositeRule = itCopy.compositeRule = compositeRule;
    out += ' ' + closing + ' if (!' + valid + ') { ';

    const err = generateError(it, 'oneOf', '', 'should match exactly one schema in oneOf', schemaPath, data, { passingSchemas: passingSchemas });
    if (!it.compositeRule && allErrors) {
        if (it.async) out += ' throw new ValidationError(vErrors); ';
        else out += ' validate.errors = vErrors; return false; ';
    } else {
        out += ' var err = ' + err + '; if (vErrors === null) vErrors = [err]; else vErrors.push(err); errors++; ';
    }

    out += ' } else { errors = ' + errs + '; if (vErrors !== null) { if (' + errs + ') vErrors.length = ' + errs + '; else vErrors = null; } ';
    if (it.opts.allErrors) out += ' } ';

    return out;
};

export const pattern = (it: any, keyword: string): string => {
    let out = ' ';
    const level = it.level;
    const dataLevel = it.dataLevel;
    const schema = it.schema[keyword];
    const schemaPath = it.schemaPath + it.util.getProperty(keyword);
    const errSchemaPath = it.errSchemaPath + '/' + keyword;
    const allErrors = !it.opts.allErrors;
    const data = 'data' + (dataLevel || '');
    const dataVar = it.opts.$data && schema && schema.$data;

    let schemaVar;
    if (dataVar) {
        out += ' var schema' + level + ' = ' + it.util.getData(schema.$data, dataLevel, it.dataPathArr) + '; ';
        schemaVar = 'schema' + level;
    } else {
        schemaVar = schema;
    }

    const patternVar = dataVar ? '(new RegExp(' + schemaVar + '))' : it.usePattern(schema);
    out += ' if ( ';
    if (dataVar) out += ' (' + schemaVar + ' !== undefined && typeof ' + schemaVar + " != 'string') || ";
    out += ' !' + patternVar + '.test(' + data + ') ) { ';

    const err = generateError(it, 'pattern', '', 'should match pattern "' + (dataVar ? "' + " + schemaVar + " + '" : it.util.escapeQuotes(schema)) + '"', schemaPath, data, { pattern: dataVar ? schemaVar : it.util.toQuotedString(schema) });
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

export const properties = (it: any, keyword: string): string => {
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
    const keyVar = 'key' + level;
    const idxVar = 'idx' + level;
    const nextDataLevel = itCopy.dataLevel = it.dataLevel + 1;
    const nextData = 'data' + nextDataLevel;
    const propsVar = 'dataProperties' + level;

    const props = Object.keys(schema || {}).filter(k => k !== '__proto__');
    const patProps = it.schema.patternProperties || {};
    const patPropsKeys = Object.keys(patProps).filter(k => k !== '__proto__');
    const addProps = it.schema.additionalProperties;
    const hasProps = props.length || patPropsKeys.length;
    const noAddProps = addProps === false;
    const hasAddSch = typeof addProps === 'object' && Object.keys(addProps).length;
    const remAdd = it.opts.removeAdditional;
    const shouldCheckAdd = noAddProps || hasAddSch || remAdd;
    const ownProperties = it.opts.ownProperties;
    const baseId = it.baseId;
    const required = it.schema.required;
    let requiredHash: any;
    if (required && !(it.opts.$data && required.$data) && required.length < it.opts.loopRequired) {
        requiredHash = it.util.toHash(required);
    }

    out += ' var ' + errs + ' = errors; var ' + nextValid + ' = true; ';
    if (ownProperties) out += ' var ' + propsVar + ' = undefined; ';

    if (shouldCheckAdd) {
        if (ownProperties) {
            out += ' ' + propsVar + ' = ' + propsVar + ' || Object.keys(' + data + '); for (var ' + idxVar + ' = 0; ' + idxVar + ' < ' + propsVar + '.length; ' + idxVar + '++) { var ' + keyVar + ' = ' + propsVar + '[' + idxVar + ']; ';
        } else {
            out += ' for (var ' + keyVar + ' in ' + data + ') { ';
        }

        if (hasProps) {
            out += ' var isAdditional' + level + ' = !(false ';
            if (props.length) {
                if (props.length > 8) {
                    out += ' || validate.schema' + schemaPath + '.hasOwnProperty(' + keyVar + ') ';
                } else {
                    props.forEach(p => {
                        out += ' || ' + keyVar + ' == ' + it.util.toQuotedString(p) + ' ';
                    });
                }
            }
            if (patPropsKeys.length) {
                patPropsKeys.forEach(p => {
                    out += ' || ' + it.usePattern(p) + '.test(' + keyVar + ') ';
                });
            }
            out += ' ); if (isAdditional' + level + ') { ';
        }

        if (remAdd === 'all') {
            out += ' delete ' + data + '[' + keyVar + ']; ';
        } else {
            const oldErrPath = it.errorPath;
            const keyExpr = "' + " + keyVar + " + '";
            if (it.opts._errorDataPathProperty) {
                it.errorPath = it.util.getPathExpr(it.errorPath, keyVar, it.opts.jsonPointers);
            }

            if (noAddProps) {
                if (remAdd) {
                    out += ' delete ' + data + '[' + keyVar + ']; ';
                } else {
                    out += ' ' + nextValid + ' = false; ';
                    const currentW = it.errSchemaPath;
                    const addPropsPath = it.errSchemaPath + '/additionalProperties';
                    const msg = it.opts._errorDataPathProperty ? 'is an invalid additional property' : 'should NOT have additional properties';
                    const err = generateError(it, 'additionalProperties', '/additionalProperties', msg, '.additionalProperties', data, { additionalProperty: keyExpr });
                    if (!it.compositeRule && allErrors) {
                        if (it.async) out += ' throw new ValidationError([' + err + ']); ';
                        else out += ' validate.errors = [' + err + ']; return false; ';
                    } else {
                        out += ' var err = ' + err + '; if (vErrors === null) vErrors = [err]; else vErrors.push(err); errors++; ';
                    }
                    if (allErrors) out += ' break; ';
                }
            } else if (hasAddSch) {
                if (remAdd === 'failing') {
                    out += ' var ' + errs + ' = errors; ';
                    const compositeRule = it.compositeRule;
                    it.compositeRule = itCopy.compositeRule = true;
                    itCopy.schema = addProps;
                    itCopy.schemaPath = it.schemaPath + '.additionalProperties';
                    itCopy.errSchemaPath = it.errSchemaPath + '/additionalProperties';
                    itCopy.errorPath = it.opts._errorDataPathProperty ? it.errorPath : it.util.getPathExpr(it.errorPath, keyVar, it.opts.jsonPointers);
                    const propData = data + '[' + keyVar + ']';
                    itCopy.dataPathArr[nextDataLevel] = keyVar;
                    const validate = it.validate(itCopy);
                    itCopy.baseId = baseId;
                    if (it.util.varOccurences(validate, nextData) < 2) {
                        out += ' ' + it.util.varReplace(validate, nextData, propData) + ' ';
                    } else {
                        out += ' var ' + nextData + ' = ' + propData + '; ' + validate + ' ';
                    }
                    out += ' if (!' + nextValid + ') { errors = ' + errs + '; if (validate.errors !== null) { if (errors) validate.errors.length = errors; else validate.errors = null; } delete ' + data + '[' + keyVar + ']; } ';
                    it.compositeRule = itCopy.compositeRule = compositeRule;
                } else {
                    itCopy.schema = addProps;
                    itCopy.schemaPath = it.schemaPath + '.additionalProperties';
                    itCopy.errSchemaPath = it.errSchemaPath + '/additionalProperties';
                    itCopy.errorPath = it.opts._errorDataPathProperty ? it.errorPath : it.util.getPathExpr(it.errorPath, keyVar, it.opts.jsonPointers);
                    const propData = data + '[' + keyVar + ']';
                    itCopy.dataPathArr[nextDataLevel] = keyVar;
                    const validate = it.validate(itCopy);
                    itCopy.baseId = baseId;
                    if (it.util.varOccurences(validate, nextData) < 2) {
                        out += ' ' + it.util.varReplace(validate, nextData, propData) + ' ';
                    } else {
                        out += ' var ' + nextData + ' = ' + propData + '; ' + validate + ' ';
                    }
                    if (allErrors) out += ' if (!' + nextValid + ') break; ';
                }
            }
            it.errorPath = oldErrPath;
        }

        if (hasProps) out += ' } ';
        out += ' } ';
        if (allErrors) out += ' if (' + nextValid + ') { ';
    }

    const useDefaults = it.opts.useDefaults && !it.compositeRule;
    if (props.length) {
        props.forEach(p => {
            const sch = schema[p];
            if (it.opts.strictKeywords ? typeof sch === 'object' && Object.keys(sch).length > 0 || sch === false : it.util.schemaHasRules(sch, it.RULES.all)) {
                const propPath = it.util.getProperty(p);
                const propData = data + propPath;
                const hasDefault = useDefaults && sch.default !== undefined;
                itCopy.schema = sch;
                itCopy.schemaPath = schemaPath + propPath;
                itCopy.errSchemaPath = errSchemaPath + '/' + it.util.escapeFragment(p);
                itCopy.errorPath = it.util.getPath(it.errorPath, p, it.opts.jsonPointers);
                itCopy.dataPathArr[nextDataLevel] = it.util.toQuotedString(p);
                let validate = it.validate(itCopy);
                itCopy.baseId = baseId;

                let dataVarName;
                if (it.util.varOccurences(validate, nextData) < 2) {
                    validate = it.util.varReplace(validate, nextData, propData);
                    dataVarName = propData;
                } else {
                    dataVarName = nextData;
                    out += ' var ' + nextData + ' = ' + propData + '; ';
                }

                if (hasDefault) {
                    out += ' ' + validate + ' ';
                } else {
                    if (requiredHash && requiredHash[p]) {
                        out += ' if (' + dataVarName + ' === undefined ';
                        if (ownProperties) out += ' || !Object.prototype.hasOwnProperty.call(' + data + ", '" + it.util.escapeQuotes(p) + "') ";
                        out += ') { ' + nextValid + ' = false; ';
                        const oldE = it.errorPath;
                        if (it.opts._errorDataPathProperty) it.errorPath = it.util.getPath(oldE, p, it.opts.jsonPointers);
                        const err = generateError(it, 'required', '/required', "should have required property '" + it.util.escapeQuotes(p) + "'", '.required', data, { missingProperty: it.util.escapeQuotes(p) });
                        if (!it.compositeRule && allErrors) {
                            if (it.async) out += ' throw new ValidationError([' + err + ']); ';
                            else out += ' validate.errors = [' + err + ']; return false; ';
                        } else {
                            out += ' var err = ' + err + '; if (vErrors === null) vErrors = [err]; else vErrors.push(err); errors++; ';
                        }
                        it.errorPath = oldE;
                        out += ' } else { ' + validate + ' } ';
                    } else if (allErrors) {
                        out += ' if (' + dataVarName + ' === undefined ';
                        if (ownProperties) out += ' || !Object.prototype.hasOwnProperty.call(' + data + ", '" + it.util.escapeQuotes(p) + "') ";
                        out += ') { ' + nextValid + ' = true; } else { ' + validate + ' } ';
                    } else {
                        out += ' if (' + dataVarName + ' !== undefined ';
                        if (ownProperties) out += ' && Object.prototype.hasOwnProperty.call(' + data + ", '" + it.util.escapeQuotes(p) + "') ";
                        out += ') { ' + validate + ' } ';
                    }
                }
                if (allErrors) out += ' if (' + nextValid + ') { ';
            }
        });
    }

    if (patPropsKeys.length) {
        patPropsKeys.forEach(p => {
            const sch = patProps[p];
            if (it.opts.strictKeywords ? typeof sch === 'object' && Object.keys(sch).length > 0 || sch === false : it.util.schemaHasRules(sch, it.RULES.all)) {
                itCopy.schema = sch;
                itCopy.schemaPath = it.schemaPath + '.patternProperties' + it.util.getProperty(p);
                itCopy.errSchemaPath = it.errSchemaPath + '/patternProperties/' + it.util.escapeFragment(p);
                if (ownProperties) {
                    out += ' ' + propsVar + ' = ' + propsVar + ' || Object.keys(' + data + '); for (var ' + idxVar + ' = 0; ' + idxVar + ' < ' + propsVar + '.length; ' + idxVar + '++) { var ' + keyVar + ' = ' + propsVar + '[' + idxVar + ']; ';
                } else {
                    out += ' for (var ' + keyVar + ' in ' + data + ') { ';
                }
                out += ' if (' + it.usePattern(p) + '.test(' + keyVar + ')) { ';
                itCopy.errorPath = it.util.getPathExpr(it.errorPath, keyVar, it.opts.jsonPointers);
                const propData = data + '[' + keyVar + ']';
                itCopy.dataPathArr[nextDataLevel] = keyVar;
                const validate = it.validate(itCopy);
                itCopy.baseId = baseId;
                if (it.util.varOccurences(validate, nextData) < 2) {
                    out += ' ' + it.util.varReplace(validate, nextData, propData) + ' ';
                } else {
                    out += ' var ' + nextData + ' = ' + propData + '; ' + validate + ' ';
                }
                if (allErrors) out += ' if (!' + nextValid + ') break; ';
                out += ' } ';
                if (allErrors) out += ' else ' + nextValid + ' = true; ';
                out += ' } ';
                if (allErrors) out += ' if (' + nextValid + ') { ';
            }
        });
    }

    if (allErrors) {
        // Closing brackets handled by code generation loop
        out += ' if (' + errs + ' == errors) {';
    }

    return out;
};

export const propertyNames = (it: any, keyword: string): string => {
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

    out += ' var ' + errs + ' = errors;';
    if (it.opts.strictKeywords ? typeof schema === 'object' && Object.keys(schema).length > 0 || schema === false : it.util.schemaHasRules(schema, it.RULES.all)) {
        itCopy.schema = schema;
        itCopy.schemaPath = schemaPath;
        itCopy.errSchemaPath = errSchemaPath;
        const keyVar = 'key' + level;
        const idxVar = 'idx' + level;
        const i = 'i' + level;
        const keyExpr = "' + " + keyVar + " + '";
        const nextDataLevel = itCopy.dataLevel = it.dataLevel + 1;
        const nextData = 'data' + nextDataLevel;
        const propsVar = 'dataProperties' + level;
        const ownProperties = it.opts.ownProperties;
        const baseId = it.baseId;

        if (ownProperties) out += ' var ' + propsVar + ' = undefined; ';
        if (ownProperties) {
            out += ' ' + propsVar + ' = ' + propsVar + ' || Object.keys(' + data + '); for (var ' + idxVar + ' = 0; ' + idxVar + ' < ' + propsVar + '.length; ' + idxVar + '++) { var ' + keyVar + ' = ' + propsVar + '[' + idxVar + ']; ';
        } else {
            out += ' for (var ' + keyVar + ' in ' + data + ') { ';
        }

        out += ' var startErrs' + level + ' = errors; ';
        const compositeRule = it.compositeRule;
        it.compositeRule = itCopy.compositeRule = true;
        const validate = it.validate(itCopy);
        it.compositeRule = itCopy.compositeRule = compositeRule;
        itCopy.baseId = baseId;

        if (it.util.varOccurences(validate, nextData) < 2) {
            out += ' ' + it.util.varReplace(validate, nextData, keyVar) + ' ';
        } else {
            out += ' var ' + nextData + ' = ' + keyVar + '; ' + validate + ' ';
        }

        out += ' if (!' + nextValid + ') { for (var ' + i + ' = startErrs' + level + '; ' + i + ' < errors; ' + i + '++) { vErrors[' + i + '].propertyName = ' + keyVar + '; } ';
        const err = generateError(it, 'propertyNames', '', "property name '" + keyExpr + "' is invalid", schemaPath, data, { propertyName: keyExpr });
        if (!it.compositeRule && allErrors) {
            if (it.async) out += ' throw new ValidationError(vErrors); ';
            else out += ' validate.errors = vErrors; return false; ';
        } else {
            out += ' var err = ' + err + '; if (vErrors === null) vErrors = [err]; else vErrors.push(err); errors++; ';
        }
        if (allErrors) out += ' break; ';
        out += ' } } ';
    }

    if (allErrors) out += ' if (' + errs + ' == errors) {';

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
