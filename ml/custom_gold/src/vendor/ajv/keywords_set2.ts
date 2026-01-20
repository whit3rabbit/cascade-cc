export const ifKeyword = (it: any, keyword: string): string => {
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

    const thenSch = it.schema.then;
    const elseSch = it.schema.else;
    const hasThen = thenSch !== undefined && (it.opts.strictKeywords ? typeof thenSch === 'object' && Object.keys(thenSch).length > 0 || thenSch === false : it.util.schemaHasRules(thenSch, it.RULES.all));
    const hasElse = elseSch !== undefined && (it.opts.strictKeywords ? typeof elseSch === 'object' && Object.keys(elseSch).length > 0 || elseSch === false : it.util.schemaHasRules(elseSch, it.RULES.all));
    const baseId = itCopy.baseId;

    if (hasThen || hasElse) {
        let failingKeyword;
        itCopy.createErrors = false;
        itCopy.schema = schema;
        itCopy.schemaPath = schemaPath;
        itCopy.errSchemaPath = errSchemaPath;
        out += ' var ' + errs + ' = errors; var ' + valid + ' = true; ';
        const compositeRule = it.compositeRule;
        it.compositeRule = itCopy.compositeRule = true;
        out += ' ' + it.validate(itCopy) + ' ';
        itCopy.baseId = baseId;
        itCopy.createErrors = true;
        out += ' errors = ' + errs + '; if (vErrors !== null) { if (' + errs + ') vErrors.length = ' + errs + '; else vErrors = null; } ';
        it.compositeRule = itCopy.compositeRule = compositeRule;

        if (hasThen) {
            out += ' if (' + nextValid + ') { ';
            itCopy.schema = it.schema.then;
            itCopy.schemaPath = it.schemaPath + '.then';
            itCopy.errSchemaPath = it.errSchemaPath + '/then';
            out += ' ' + it.validate(itCopy) + ' ';
            itCopy.baseId = baseId;
            out += ' ' + valid + ' = ' + nextValid + '; ';
            if (hasThen && hasElse) {
                failingKeyword = 'ifClause' + level;
                out += ' var ' + failingKeyword + " = 'then'; ";
            } else {
                failingKeyword = "'then'";
            }
            out += ' } ';
            if (hasElse) out += ' else { ';
        } else {
            out += ' if (!' + nextValid + ') { ';
        }

        if (hasElse) {
            itCopy.schema = it.schema.else;
            itCopy.schemaPath = it.schemaPath + '.else';
            itCopy.errSchemaPath = it.errSchemaPath + '/else';
            out += ' ' + it.validate(itCopy) + ' ';
            itCopy.baseId = baseId;
            out += ' ' + valid + ' = ' + nextValid + '; ';
            if (hasThen && hasElse) {
                failingKeyword = 'ifClause' + level;
                out += ' var ' + failingKeyword + " = 'else'; ";
            } else {
                failingKeyword = "'else'";
            }
            out += ' } ';
        }

        out += ' if (!' + valid + ') { ';
        const err = generateError(it, 'if', '', 'should match "' + (failingKeyword || 'then') + '" schema', schemaPath, data, { failingKeyword: failingKeyword || 'then' });
        if (!it.compositeRule && allErrors) {
            if (it.async) out += ' throw new ValidationError(vErrors); ';
            else out += ' validate.errors = vErrors; return false; ';
        } else {
            out += ' var err = ' + err + '; if (vErrors === null) vErrors = [err]; else vErrors.push(err); errors++; ';
        }
        out += ' }';
        if (allErrors) out += ' else { ';
    } else if (allErrors) {
        out += ' if (true) { ';
    }

    return out;
};

export const items = (it: any, keyword: string): string => {
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
    let closing = '';

    out += ' var ' + errs + ' = errors; var ' + valid + ';';

    if (Array.isArray(schema)) {
        const additionalItems = it.schema.additionalItems;
        if (additionalItems === false) {
            out += ' ' + valid + ' = ' + data + '.length <= ' + schema.length + '; ';
            const currentErrPath = it.errSchemaPath;
            const additionalItemsPath = it.errSchemaPath + '/additionalItems';
            out += ' if (!' + valid + ') { ';
            const err = generateError(it, 'additionalItems', '', 'should NOT have more than ' + schema.length + ' items', '.additionalItems', data, { limit: schema.length });
            if (!it.compositeRule && allErrors) {
                if (it.async) out += ' throw new ValidationError([' + err + ']); ';
                else out += ' validate.errors = [' + err + ']; return false; ';
            } else {
                out += ' var err = ' + err + '; if (vErrors === null) vErrors = [err]; else vErrors.push(err); errors++; ';
            }
            out += ' } ';
            if (allErrors) {
                closing += '}';
                out += ' else { ';
            }
        }

        schema.forEach((sch: any, idx: number) => {
            if (it.opts.strictKeywords ? typeof sch === 'object' && Object.keys(sch).length > 0 || sch === false : it.util.schemaHasRules(sch, it.RULES.all)) {
                out += ' ' + nextValid + ' = true; if (' + data + '.length > ' + idx + ') { ';
                const itemData = data + '[' + idx + ']';
                itCopy.schema = sch;
                itCopy.schemaPath = schemaPath + '[' + idx + ']';
                itCopy.errSchemaPath = errSchemaPath + '/' + idx;
                itCopy.errorPath = it.util.getPathExpr(it.errorPath, idx, it.opts.jsonPointers, true);
                itCopy.dataPathArr[nextDataLevel] = idx;
                const validate = it.validate(itCopy);
                itCopy.baseId = baseId;
                if (it.util.varOccurences(validate, nextData) < 2) {
                    out += ' ' + it.util.varReplace(validate, nextData, itemData) + ' ';
                } else {
                    out += ' var ' + nextData + ' = ' + itemData + '; ' + validate + ' ';
                }
                out += ' } ';
                if (allErrors) {
                    out += ' if (' + nextValid + ') { ';
                    closing += '}';
                }
            }
        });

        if (typeof additionalItems === 'object' && (it.opts.strictKeywords ? typeof additionalItems === 'object' && Object.keys(additionalItems).length > 0 || additionalItems === false : it.util.schemaHasRules(additionalItems, it.RULES.all))) {
            itCopy.schema = additionalItems;
            itCopy.schemaPath = it.schemaPath + '.additionalItems';
            itCopy.errSchemaPath = it.errSchemaPath + '/additionalItems';
            out += ' ' + nextValid + ' = true; if (' + data + '.length > ' + schema.length + ') { for (var ' + i + ' = ' + schema.length + '; ' + i + ' < ' + data + '.length; ' + i + '++) { ';
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
            if (allErrors) out += ' if (!' + nextValid + ') break; ';
            out += ' } } ';
            if (allErrors) {
                out += ' if (' + nextValid + ') { ';
                closing += '}';
            }
        }
    } else if (it.opts.strictKeywords ? typeof schema === 'object' && Object.keys(schema).length > 0 || schema === false : it.util.schemaHasRules(schema, it.RULES.all)) {
        itCopy.schema = schema;
        itCopy.schemaPath = schemaPath;
        itCopy.errSchemaPath = errSchemaPath;
        out += ' for (var ' + i + ' = 0; ' + i + ' < ' + data + '.length; ' + i + '++) { ';
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
        if (allErrors) out += ' if (!' + nextValid + ') break; ';
        out += ' }';
    }

    if (allErrors) {
        out += ' ' + closing + ' if (' + errs + ' == errors) {';
    }

    return out;
};

export const limit = (it: any, keyword: string): string => {
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

    const isMax = keyword === 'maximum';
    const exclusiveKeyword = isMax ? 'exclusiveMaximum' : 'exclusiveMinimum';
    const exclusiveSchema = it.schema[exclusiveKeyword];
    const exclusiveDataVar = it.opts.$data && exclusiveSchema && exclusiveSchema.$data;
    const op = isMax ? '<' : '>';
    const invOp = isMax ? '>' : '<';

    let isExcl = false;
    let limitVar = schemaVar;

    if (exclusiveDataVar) {
        const exclData = it.util.getData(exclusiveSchema.$data, dataLevel, it.dataPathArr);
        const exclVar = 'exclusive' + level;
        const exclType = 'exclType' + level;
        out += ' var schemaExcl' + level + ' = ' + exclData + '; ';
        const schemaExclVar = 'schemaExcl' + level;
        out += ' var ' + exclVar + '; var ' + exclType + ' = typeof ' + schemaExclVar + '; if (' + exclType + " != 'boolean' && " + exclType + " != 'undefined' && " + exclType + " != 'number') { ";
        const err = generateError(it, exclusiveKeyword, '', exclusiveKeyword + ' should be boolean', '.' + exclusiveKeyword, data);
        if (!it.compositeRule && allErrors) {
            if (it.async) out += ' throw new ValidationError([' + err + ']); ';
            else out += ' validate.errors = [' + err + ']; return false; ';
        } else {
            out += ' var err = ' + err + '; if (vErrors === null) vErrors = [err]; else vErrors.push(err); errors++; ';
        }
        out += ' } else if ( ';
        if (dataVar) out += ' (' + schemaVar + ' !== undefined && typeof ' + schemaVar + " != 'number') || ";
        out += ' ' + exclType + " == 'number' ? ( (" + exclVar + ' = ' + schemaVar + ' === undefined || ' + schemaExclVar + ' ' + op + '= ' + schemaVar + ') ? ' + data + ' ' + invOp + '= ' + schemaExclVar + ' : ' + data + ' ' + invOp + ' ' + schemaVar + ' ) : ( (' + exclVar + ' = ' + schemaExclVar + ' === true) ? ' + data + ' ' + invOp + '= ' + schemaVar + ' : ' + data + ' ' + invOp + ' ' + schemaVar + ' ) || ' + data + ' !== ' + data + ') { var op' + level + ' = ' + exclVar + " ? '" + op + "' : '" + op + "='; ";
    } else {
        const isExclNumber = typeof exclusiveSchema === 'number';
        let opStr = op;
        limitVar = schemaVar;
        isExcl = false;
        let currentErrSchemaPath = errSchemaPath;
        let currentKeyword = keyword;

        if (isExclNumber && dataVar) {
            out += ' if ( ';
            if (dataVar) out += ' (' + schemaVar + ' !== undefined && typeof ' + schemaVar + " != 'number') || ";
            out += ' ( ' + schemaVar + ' === undefined || ' + exclusiveSchema + ' ' + op + '= ' + schemaVar + ' ? ' + data + ' ' + invOp + '= ' + exclusiveSchema + ' : ' + data + ' ' + invOp + ' ' + schemaVar + ' ) || ' + data + ' !== ' + data + ') { ';
        } else {
            if (isExclNumber && schema === undefined) {
                isExcl = true;
                currentKeyword = exclusiveKeyword;
                currentErrSchemaPath = it.errSchemaPath + '/' + exclusiveKeyword;
                limitVar = exclusiveSchema;
            } else {
                if (isExclNumber) limitVar = Math[isMax ? 'min' : 'max'](exclusiveSchema, schema);
                if (exclusiveSchema === (isExclNumber ? limitVar : true)) {
                    isExcl = true;
                    currentKeyword = exclusiveKeyword;
                    currentErrSchemaPath = it.errSchemaPath + '/' + exclusiveKeyword;
                } else {
                    opStr += '=';
                }
            }
            out += ' if ( ';
            if (dataVar) out += ' (' + schemaVar + ' !== undefined && typeof ' + schemaVar + " != 'number') || ";
            out += ' ' + data + ' ' + invOp + (isExcl ? '' : '=') + ' ' + limitVar + ' || ' + data + ' !== ' + data + ') { ';
        }
    }

    const err = generateError(it, keyword, '', 'should be ' + op + (dataVar ? "' + " + limitVar : limitVar + "'"), '.' + keyword, data, { comparison: (dataVar ? 'op' + level : "'" + op + (isExcl ? '' : '=') + "'"), limit: limitVar, exclusive: isExcl });
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

export const limitItems = (it: any, keyword: string): string => {
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

    const op = keyword === 'maxItems' ? '>' : '<';
    out += ' if ( ';
    if (dataVar) out += ' (' + schemaVar + ' !== undefined && typeof ' + schemaVar + " != 'number') || ";
    out += ' ' + data + '.length ' + op + ' ' + schemaVar + ') { ';

    const msg = 'should NOT have ' + (keyword === 'maxItems' ? 'more' : 'fewer') + ' than ' + (dataVar ? "' + " + schemaVar + " + '" : schema) + ' items';
    const err = generateError(it, keyword, '', msg, '.' + keyword, data, { limit: schemaVar });

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

export const limitLength = (it: any, keyword: string): string => {
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

    const op = keyword === 'maxLength' ? '>' : '<';
    out += ' if ( ';
    if (dataVar) out += ' (' + schemaVar + ' !== undefined && typeof ' + schemaVar + " != 'number') || ";
    if (it.opts.unicode === false) out += ' ' + data + '.length ';
    else out += ' ucs2length(' + data + ') ';
    out += ' ' + op + ' ' + schemaVar + ') { ';

    const msg = 'should NOT be ' + (keyword === 'maxLength' ? 'longer' : 'shorter') + ' than ' + (dataVar ? "' + " + schemaVar + " + '" : schema) + ' characters';
    const err = generateError(it, keyword, '', msg, '.' + keyword, data, { limit: schemaVar });

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

export const limitProperties = (it: any, keyword: string): string => {
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

    const op = keyword === 'maxProperties' ? '>' : '<';
    out += ' if ( ';
    if (dataVar) out += ' (' + schemaVar + ' !== undefined && typeof ' + schemaVar + " != 'number') || ";
    out += ' Object.keys(' + data + ').length ' + op + ' ' + schemaVar + ') { ';

    const msg = 'should NOT have ' + (keyword === 'maxProperties' ? 'more' : 'fewer') + ' than ' + (dataVar ? "' + " + schemaVar + " + '" : schema) + ' properties';
    const err = generateError(it, keyword, '', msg, '.' + keyword, data, { limit: schemaVar });

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

export const multipleOf = (it: any, keyword: string): string => {
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

    out += ' var division' + level + '; if ( ';
    if (dataVar) out += ' ' + schemaVar + ' !== undefined && ( typeof ' + schemaVar + " != 'number' || ";
    out += ' (division' + level + ' = ' + data + ' / ' + schemaVar + ', ';
    if (it.opts.multipleOfPrecision) {
        out += ' Math.abs(Math.round(division' + level + ') - division' + level + ') > 1e-' + it.opts.multipleOfPrecision + ' ';
    } else {
        out += ' division' + level + ' !== parseInt(division' + level + ') ';
    }
    out += ' ) ';
    if (dataVar) out += ' ) ';
    out += ' ) { ';

    const err = generateError(it, 'multipleOf', '', 'should be multiple of ' + (dataVar ? "' + " + schemaVar : schema + "'"), '.' + keyword, data, { multipleOf: schemaVar });
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
