export default function ref(it: any, keyword: string): string {
    let out = ' ';
    const level = it.level;
    const dataLevel = it.dataLevel;
    const schema = it.schema[keyword];
    const errSchemaPath = it.errSchemaPath + '/' + keyword;
    const allErrors = !it.opts.allErrors;
    const data = 'data' + (dataLevel || '');
    const valid = 'valid' + level;

    let async, code;
    if (schema === '#' || schema === '#/') {
        if (it.isRoot) {
            async = it.async;
            code = 'validate';
        } else {
            async = it.root.schema.$async === true;
            code = 'root.refVal[0]';
        }
    } else {
        const res = it.resolveRef(it.baseId, schema, it.isRoot);
        if (res === undefined) {
            const msg = it.MissingRefError.message(it.baseId, schema);
            if (it.opts.missingRefs === 'fail') {
                it.logger.error(msg);
                const err = generateError(it, '$ref', '/$ref', "can't resolve reference " + it.util.escapeQuotes(schema), { ref: it.util.escapeQuotes(schema) });
                if (!it.compositeRule && allErrors) {
                    if (it.async) out += ' throw new ValidationError([' + err + ']); ';
                    else out += ' validate.errors = [' + err + ']; return false; ';
                } else {
                    out += ' var err = ' + err + '; if (vErrors === null) vErrors = [err]; else vErrors.push(err); errors++; ';
                }
                if (allErrors) out += ' if (false) { ';
            } else if (it.opts.missingRefs === 'ignore') {
                it.logger.warn(msg);
                if (allErrors) out += ' if (true) { ';
            } else {
                throw new it.MissingRefError(it.baseId, schema, msg);
            }
        } else if (res.inline) {
            const itCopy = it.util.copy(it);
            itCopy.level++;
            const nextValid = 'valid' + itCopy.level;
            itCopy.schema = res.schema;
            itCopy.schemaPath = '';
            itCopy.errSchemaPath = schema;
            const refCode = it.validate(itCopy).replace(/validate\.schema/g, res.code);
            out += ' ' + refCode + ' ';
            if (allErrors) out += ' if (' + nextValid + ') { ';
        } else {
            async = res.$async === true || (it.async && res.$async !== false);
            code = res.code;
        }
    }

    if (code) {
        const call = it.opts.passContext ? code + '.call(this, ' : code + '( ';
        const args = data + ', (dataPath || \'\')' + (it.errorPath !== '""' ? ' + ' + it.errorPath : '');
        const parentData = dataLevel ? 'data' + (dataLevel - 1 || '') : 'parentData';
        const parentProp = dataLevel ? it.dataPathArr[dataLevel] : 'parentDataProperty';
        const callStr = call + args + ' , ' + parentData + ' , ' + parentProp + ', rootData) ';

        if (async) {
            if (!it.async) throw new Error('async schema referenced by sync schema');
            if (allErrors) out += ' var ' + valid + '; ';
            out += ' try { await ' + callStr + '; ';
            if (allErrors) out += ' ' + valid + ' = true; ';
            out += ' } catch (e) { if (!(e instanceof ValidationError)) throw e; if (vErrors === null) vErrors = e.errors; else vErrors = vErrors.concat(e.errors); errors = vErrors.length; ';
            if (allErrors) out += ' ' + valid + ' = false; ';
            out += ' } ';
            if (allErrors) out += ' if (' + valid + ') { ';
        } else {
            out += ' if (!' + callStr + ') { if (vErrors === null) vErrors = ' + code + '.errors; else vErrors = vErrors.concat(' + code + '.errors); errors = vErrors.length; } ';
            if (allErrors) out += ' else { ';
        }
    }

    return out;
}

function generateError(it: any, keyword: string, schemaPath: string, message: string, params: any = {}): string {
    let out = ' { keyword: \'' + keyword + '\' , dataPath: (dataPath || \'\') + ' + it.errorPath + ' , schemaPath: ' + it.util.toQuotedString(it.errSchemaPath + schemaPath) + ' , params: ' + JSON.stringify(params);
    if (it.opts.messages !== false) out += ' , message: \'' + message.replace(/'/g, "\\'") + '\' ';
    if (it.opts.verbose) {
        out += ' , schema: ' + it.util.toQuotedString(it.schema['$ref']) + ' , parentSchema: validate.schema' + it.schemaPath + ' , data: data' + (it.dataLevel || '');
    }
    out += ' } ';
    return out;
}
