export default function template(it: any, keyword?: string, assign?: string): string {
    let out = '';
    const async = it.schema.$async === true;
    const hasRules = it.util.schemaHasRulesExcept(it.schema, it.RULES.all, '$ref');
    const id = it.self._getId(it.schema);

    if (it.opts.strictKeywords) {
        const unknownRule = it.util.schemaUnknownRules(it.schema, it.RULES.keywords);
        if (unknownRule) {
            const msg = 'unknown keyword: ' + unknownRule;
            if (it.opts.strictKeywords === 'log') it.logger.warn(msg);
            else throw new Error(msg);
        }
    }

    if (it.isTop) {
        out += ' var validate = ';
        if (async) {
            it.async = true;
            out += 'async ';
        }
        out += "function(data, dataPath, parentData, parentDataProperty, rootData) { 'use strict'; ";
        if (id && (it.opts.sourceCode || it.opts.processCode)) {
            out += ' /*# sourceURL=' + id + ' */ ';
        }
    }

    if (typeof it.schema === 'boolean' || !(hasRules || it.schema.$ref)) {
        const level = it.level;
        const dataLevel = it.dataLevel;
        const allErrors = !it.opts.allErrors;
        const data = 'data' + (dataLevel || '');
        const valid = 'valid' + level;

        if (it.schema === false) {
            if (it.isTop) it.opts.allErrors = false; // Force stop on first error
            else out += ' var ' + valid + ' = false; ';

            const error = generateError(it, 'false schema', '/false schema', 'boolean schema is false');
            if (!it.compositeRule && allErrors) {
                if (it.async) out += ' throw new ValidationError([' + error + ']); ';
                else out += ' validate.errors = [' + error + ']; return false; ';
            } else {
                out += ' var err = ' + error + '; if (vErrors === null) vErrors = [err]; else vErrors.push(err); errors++; ';
            }
        } else {
            if (it.isTop) {
                if (async) out += ' return data; ';
                else out += ' validate.errors = null; return true; ';
            } else {
                out += ' var ' + valid + ' = true; ';
            }
        }

        if (it.isTop) out += ' }; return validate; ';
        return out;
    }

    if (it.isTop) {
        it.level = 0;
        it.dataLevel = 0;
        it.rootId = it.resolve.fullPath(it.self._getId(it.root.schema));
        it.baseId = it.baseId || it.rootId;
        delete it.isTop;
        it.dataPathArr = [''];

        if (it.schema.default !== undefined && it.opts.useDefaults && it.opts.strictDefaults) {
            const msg = 'default is ignored in the schema root';
            if (it.opts.strictDefaults === 'log') it.logger.warn(msg);
            else throw new Error(msg);
        }

        out += ' var vErrors = null; ';
        out += ' var errors = 0; ';
        out += ' if (rootData === undefined) rootData = data; ';
    } else {
        const level = it.level;
        const dataLevel = it.dataLevel;
        if (id) it.baseId = it.resolve.url(it.baseId, id);
        if (async && !it.async) throw new Error('async schema in sync schema');
        out += ' var errs_' + level + ' = errors;';
    }

    const level = it.level;
    const dataLevel = it.dataLevel;
    const data = 'data' + (dataLevel || '');
    const valid = 'valid' + level;
    const allErrors = !it.opts.allErrors;

    let type = it.schema.type;
    let isTypeArray = Array.isArray(type);
    let coerced: any;

    if (type && it.opts.nullable && it.schema.nullable === true) {
        if (isTypeArray) {
            if (type.indexOf('null') === -1) type = type.concat('null');
        } else if (type !== 'null') {
            type = [type, 'null'];
            isTypeArray = true;
        }
    }

    if (isTypeArray && type.length === 1) {
        type = type[0];
        isTypeArray = false;
    }

    if (it.schema.$ref && hasRules) {
        if (it.opts.extendRefs === 'fail') throw new Error('$ref: validation keywords used in schema at path "' + it.errSchemaPath + '" (see option extendRefs)');
        else if (it.opts.extendRefs !== true) {
            it.logger.warn('$ref: keywords ignored in schema at path "' + it.errSchemaPath + '"');
        }
    }

    if (it.schema.$comment && it.opts.$comment) {
        out += ' ' + it.RULES.all.$comment.code(it, '$comment');
    }

    if (type) {

        if (it.opts.coerceTypes) coerced = it.util.coerceToTypes(it.opts.coerceTypes, type);
        const rules = it.RULES.types[type];
        if (coerced || isTypeArray || rules === true || (rules && !hasRules(rules))) {
            const schemaPath = it.schemaPath + '.type';
            const errSchemaPath = it.errSchemaPath + '/type';
            const method = isTypeArray ? 'checkDataTypes' : 'checkDataType';

            out += ' if (' + it.util[method](type, data, it.opts.strictNumbers, true) + ') { ';
            if (coerced) {
                const dataType = 'dataType' + level;
                const coercedData = 'coerced' + level;
                out += ' var ' + dataType + ' = typeof ' + data + '; var ' + coercedData + ' = undefined; ';
                if (it.opts.coerceTypes === 'array') {
                    out += ' if (' + dataType + " == 'object' && Array.isArray(" + data + ') && ' + data + '.length == 1) { ' + data + ' = ' + data + '[0]; ' + dataType + ' = typeof ' + data + '; if (' + it.util.checkDataType(it.schema.type, data, it.opts.strictNumbers) + ') ' + coercedData + ' = ' + data + '; } ';
                }
                out += ' if (' + coercedData + ' !== undefined) ; ';

                for (const t of coerced) {
                    if (t === 'string') {
                        out += ' else if (' + dataType + " == 'number' || " + dataType + " == 'boolean') " + coercedData + " = '' + " + data + '; else if (' + data + " === null) " + coercedData + " = ''; ";
                    } else if (t === 'number' || t === 'integer') {
                        out += ' else if (' + dataType + ' == "boolean" || ' + data + ' === null || (' + dataType + ' == "string" && ' + data + ' && ' + data + ' == +' + data;
                        if (t === 'integer') out += ' && !(' + data + ' % 1)';
                        out += ')) ' + coercedData + ' = +' + data + '; ';
                    } else if (t === 'boolean') {
                        out += ' else if (' + data + " === 'false' || " + data + ' === 0 || ' + data + " === null) " + coercedData + ' = false; else if (' + data + " === 'true' || " + data + ' === 1) ' + coercedData + ' = true; ';
                    } else if (t === 'null') {
                        out += ' else if (' + data + " === '' || " + data + ' === 0 || ' + data + ' === false) ' + coercedData + ' = null; ';
                    } else if (it.opts.coerceTypes === 'array' && t === 'array') {
                        out += ' else if (' + dataType + " == 'string' || " + dataType + " == 'number' || " + dataType + " == 'boolean' || " + data + ' == null) ' + coercedData + ' = [' + data + ']; ';
                    }
                }

                out += ' else { ';
                const err = generateError(it, 'type', '/type', 'should be ' + (isTypeArray ? type.join(',') : type), { type: isTypeArray ? type.join(',') : type });
                if (!it.compositeRule && allErrors) {
                    if (it.async) out += ' throw new ValidationError([' + err + ']); ';
                    else out += ' validate.errors = [' + err + ']; return false; ';
                } else {
                    out += ' var err = ' + err + '; if (vErrors === null) vErrors = [err]; else vErrors.push(err); errors++; ';
                }
                out += ' } if (' + coercedData + ' !== undefined) { ';
                const parentData = dataLevel ? 'data' + (dataLevel - 1 || '') : 'parentData';
                const parentProp = dataLevel ? it.dataPathArr[dataLevel] : 'parentDataProperty';
                out += ' ' + data + ' = ' + coercedData + '; ';
                if (!dataLevel) out += ' if (' + parentData + ' !== undefined)';
                out += ' ' + parentData + '[' + parentProp + '] = ' + coercedData + '; } ';
            } else {
                const err = generateError(it, 'type', '/type', 'should be ' + (isTypeArray ? type.join(',') : type), { type: isTypeArray ? type.join(',') : type });
                if (!it.compositeRule && allErrors) {
                    if (it.async) out += ' throw new ValidationError([' + err + ']); ';
                    else out += ' validate.errors = [' + err + ']; return false; ';
                } else {
                    out += ' var err = ' + err + '; if (vErrors === null) vErrors = [err]; else vErrors.push(err); errors++; ';
                }
            }
            out += ' } ';
        }
    }

    if (it.schema.$ref && !hasRules) {
        out += ' ' + it.RULES.all.$ref.code(it, '$ref') + ' ';
        if (allErrors) {
            out += ' } if (errors === ' + (it.isTop ? '0' : 'errs_' + level) + ') { ';
        }
    } else {
        for (const group of it.RULES) {
            if (shouldUseGroup(it, group)) {
                if (group.type) out += ' if (' + it.util.checkDataType(group.type, data, it.opts.strictNumbers) + ') { ';

                if (it.opts.useDefaults) {
                    if (group.type === 'object' && it.schema.properties) {
                        for (const prop in it.schema.properties) {
                            const sch = it.schema.properties[prop];
                            if (sch.default !== undefined) {
                                const propPath = data + it.util.getProperty(prop);
                                if (it.compositeRule) {
                                    if (it.opts.strictDefaults) {
                                        const msg = 'default is ignored for: ' + propPath;
                                        if (it.opts.strictDefaults === 'log') it.logger.warn(msg);
                                        else throw new Error(msg);
                                    }
                                } else {
                                    out += ' if (' + propPath + ' === undefined ';
                                    if (it.opts.useDefaults === 'empty') out += ' || ' + propPath + " === null || " + propPath + " === '' ";
                                    out += ' ) ' + propPath + ' = ';
                                    if (it.opts.useDefaults === 'shared') out += ' ' + it.useDefault(sch.default) + ' ';
                                    else out += ' ' + JSON.stringify(sch.default) + ' ';
                                    out += '; ';
                                }
                            }
                        }
                    } else if (group.type === 'array' && Array.isArray(it.schema.items)) {
                        it.schema.items.forEach((sch: any, i: number) => {
                            if (sch.default !== undefined) {
                                const itemPath = data + '[' + i + ']';
                                if (it.compositeRule) {
                                    if (it.opts.strictDefaults) {
                                        const msg = 'default is ignored for: ' + itemPath;
                                        if (it.opts.strictDefaults === 'log') it.logger.warn(msg);
                                        else throw new Error(msg);
                                    }
                                } else {
                                    out += ' if (' + itemPath + ' === undefined ';
                                    if (it.opts.useDefaults === 'empty') out += ' || ' + itemPath + " === null || " + itemPath + " === '' ";
                                    out += ' ) ' + itemPath + ' = ';
                                    if (it.opts.useDefaults === 'shared') out += ' ' + it.useDefault(sch.default) + ' ';
                                    else out += ' ' + JSON.stringify(sch.default) + ' ';
                                    out += '; ';
                                }
                            }
                        });
                    }
                }

                let closingBrackets = '';
                for (const rule of group.rules) {
                    if (it.schema[rule.keyword] !== undefined || (rule.implements && rule.implements.some((k: string) => it.schema[k] !== undefined))) {
                        const code = rule.code(it, rule.keyword, group.type);
                        if (code) {
                            out += ' ' + code + ' ';
                            if (allErrors) closingBrackets += '}';
                        }
                    }
                }

                if (allErrors) {
                    out += ' ' + closingBrackets + ' ';
                    out += ' if (errors === ' + (it.isTop ? '0' : 'errs_' + level) + ') { ';
                }

                if (group.type) {
                    out += ' } ';
                    if (type && type === group.type && !coerced) {
                        out += ' else { ';
                        const err = generateError(it, 'type', '/type', 'should be ' + type, { type: type });
                        if (!it.compositeRule && allErrors) {
                            if (it.async) out += ' throw new ValidationError([' + err + ']); ';
                            else out += ' validate.errors = [' + err + ']; return false; ';
                        } else {
                            out += ' var err = ' + err + '; if (vErrors === null) vErrors = [err]; else vErrors.push(err); errors++; ';
                        }
                        out += ' } ';
                    }
                }
            }
        }
    }

    if (allErrors) {
        // This is handled by the closing brace of the rules
    }

    if (it.isTop) {
        if (async) out += ' if (errors === 0) return data; else throw new ValidationError(vErrors); ';
        else out += ' validate.errors = vErrors; return errors === 0; ';
        out += ' }; return validate; ';
    } else {
        out += ' var ' + valid + ' = errors === errs_' + level + ';';
    }

    return out;
}

function shouldUseGroup(it: any, group: any): boolean {
    return group.rules.some((rule: any) => {
        return it.schema[rule.keyword] !== undefined || (rule.implements && rule.implements.some((k: string) => it.schema[k] !== undefined));
    });
}

function generateError(it: any, keyword: string, schemaPath: string, message: string, params: any = {}): string {
    let out = ' { keyword: \'' + keyword + '\' , dataPath: (dataPath || \'\') + ' + it.errorPath + ' , schemaPath: ' + it.util.toQuotedString(it.errSchemaPath + schemaPath) + ' , params: ' + JSON.stringify(params);
    if (it.opts.messages !== false) out += ' , message: \'' + message.replace(/'/g, "\\'") + '\' ';
    if (it.opts.verbose) {
        out += ' , schema: validate.schema' + it.schemaPath + it.util.getProperty(keyword) + ' , parentSchema: validate.schema' + it.schemaPath + ' , data: data' + (it.dataLevel || '');
    }
    out += ' } ';
    return out;
}
