import resolve from './resolver.js';
import * as util from './util.js';
import { ValidationError, MissingRefError } from './errors.js';
import stringify from './stringify.js';
import template from './template.js';
import SchemaObject from './schema_obj.js';

export default function compile(this: any, schema: any, root?: any, localRefs?: any, baseId?: string): any {
    const self = this;
    const opts = this._opts;
    const refVal = [undefined];
    const refs: any = {};
    const patterns: string[] = [];
    const defaults: any[] = [];
    const customRules: any[] = [];

    root = root || { schema, refVal, refs };

    const comp: any = getCompilation.call(this, schema, root, baseId);
    if (comp.compiling) return comp.validate = function proxy() {
        const v = comp.validate;
        const res = v.apply(this, arguments);
        (proxy as any).errors = v.errors;
        return res;
    };

    const formats = this._formats;
    const RULES = this.RULES;

    try {
        const validate = _compile(schema, root, localRefs, baseId);
        comp.validate = validate;
        return validate;
    } finally {
        endCompilation.call(this, schema, root, baseId);
    }

    function _compile(sch: any, rootObj: any, localR: any, bId?: string): any {
        const isRoot = !rootObj || rootObj.schema === sch;
        if (rootObj.schema !== root.schema) return compile.call(self, sch, rootObj, localR, bId);

        const async = sch.$async === true;
        const source = template({
            isTop: true,
            schema: sch,
            isRoot: isRoot,
            baseId: bId,
            root: rootObj,
            schemaPath: '',
            errSchemaPath: '#',
            errorPath: '""',
            MissingRefError: MissingRefError,
            RULES: RULES,
            validate: template,
            util: util,
            resolve: resolve,
            resolveRef: resolveRef,
            usePattern: usePattern,
            useDefault: useDefault,
            useCustomRule: useCustomRule,
            opts: opts,
            formats: formats,
            logger: self.logger,
            self: self
        });

        const prepended = mI1(refVal, Hn5) + mI1(patterns, Kn5) + mI1(defaults, Vn5) + mI1(customRules, Dn5);
        const code = prepended + source;

        let validate: any;
        try {
            const func = new Function('self', 'RULES', 'formats', 'root', 'refVal', 'defaults', 'customRules', 'equal', 'ucs2length', 'ValidationError', code);
            validate = func(self, RULES, formats, root, refVal, defaults, customRules, util.equal, util.ucs2length, ValidationError);
            refVal[0] = validate;
        } catch (e) {
            self.logger.error('Error compiling schema, function code:', code);
            throw e;
        }

        validate.schema = sch;
        validate.errors = null;
        validate.refs = refs;
        validate.refVal = refVal;
        validate.root = isRoot ? validate : rootObj;
        if (async) validate.$async = true;
        if (opts.sourceCode === true) validate.source = { code, patterns, defaults };

        return validate;
    }

    function resolveRef(base: string, ref: string, isRoot: boolean) {
        ref = resolve.url(base, ref);
        let index = refs[ref];
        if (index !== undefined) return { code: 'refVal' + index, schema: refVal[index], inline: true };

        if (!isRoot && root.refs) {
            index = root.refs[ref];
            if (index !== undefined) return { code: 'refVal[' + index + ']', schema: root.refVal[index], inline: true };
        }

        const code = 'refVal' + refVal.length;
        let validate = resolve.call(self, _compile, root, ref);
        if (validate === undefined) {
            const inline = localRefs && localRefs[ref];
            if (inline) validate = resolve.inlineRef(inline, opts.inlineRefs) ? inline : compile.call(self, inline, root, localRefs, base);
        }

        if (validate === undefined) throw new MissingRefError(base, ref);

        index = refVal.length;
        refVal[index] = validate;
        refs[ref] = index;

        return {
            code: code,
            schema: validate,
            inline: true
        };
    }

    function usePattern(p: string) {
        let index = patterns.indexOf(p);
        if (index === -1) {
            index = patterns.length;
            patterns[index] = p;
        }
        return 'pattern' + index;
    }

    function useDefault(d: any) {
        switch (typeof d) {
            case 'boolean':
            case 'number':
                return '' + d;
            case 'string':
                return util.toQuotedString(d);
            case 'object':
                if (d === null) return 'null';
                const str = stringify(d);
                const index = defaults.indexOf(d); // Should use a better check
                if (index === -1) {
                    const i = defaults.length;
                    defaults[i] = d;
                    return 'default' + i;
                }
                return 'default' + index;
        }
    }

    function useCustomRule(rule: any, sch: any, parent: any, it: any) {
        if (self._opts.validateSchema !== false) {
            const deps = rule.definition.dependencies;
            if (deps && !deps.every((d: string) => Object.prototype.hasOwnProperty.call(parent, d))) {
                throw new Error('parent schema must have all required keywords: ' + deps.join(','));
            }
            const validateSchema = rule.definition.validateSchema;
            if (validateSchema) {
                if (!validateSchema(sch)) {
                    const msg = 'keyword schema is invalid: ' + self.errorsText(validateSchema.errors);
                    if (self._opts.validateSchema === 'log') self.logger.error(msg);
                    else throw new Error(msg);
                }
            }
        }

        const definition = rule.definition;
        let compiled;
        if (definition.compile) compiled = definition.compile.call(self, sch, parent, it);
        else if (definition.macro) {
            compiled = definition.macro.call(self, sch, parent, it);
            if (opts.validateSchema !== false) self.validateSchema(compiled, true);
        } else if (definition.inline) compiled = definition.inline.call(self, it, rule.keyword, sch, parent);
        else if (definition.validate) compiled = definition.validate;

        if (compiled === undefined) throw new Error('custom keyword "' + rule.keyword + '" failed to compile');

        const index = customRules.length;
        customRules[index] = compiled;
        return {
            code: 'customRule' + index,
            validate: compiled
        };
    }
}

function getCompilation(this: any, schema: any, root: any, baseId?: string) {
    const index = findCompilation.call(this, schema, root, baseId);
    if (index >= 0) return { index, compiling: true };
    const i = this._compilations.length;
    this._compilations[i] = { schema, root, baseId };
    return { index: i, compiling: false };
}

function endCompilation(this: any, schema: any, root: any, baseId?: string) {
    const index = findCompilation.call(this, schema, root, baseId);
    if (index >= 0) this._compilations.splice(index, 1);
}

function findCompilation(this: any, schema: any, root: any, baseId?: string) {
    for (let i = 0; i < this._compilations.length; i++) {
        const c = this._compilations[i];
        if (c.schema === schema && c.root === root && c.baseId === baseId) return i;
    }
    return -1;
}

function Kn5(i: number, p: string[]) {
    return 'var pattern' + i + ' = new RegExp(' + util.toQuotedString(p[i]) + ');';
}

function Vn5(i: number) {
    return 'var default' + i + ' = defaults[' + i + '];';
}

function Hn5(i: number, r: any[]) {
    return r[i] === undefined ? '' : 'var refVal' + i + ' = refVal[' + i + '];';
}

function Dn5(i: number) {
    return 'var customRule' + i + ' = customRules[' + i + '];';
}

function mI1(arr: any[], fn: any) {
    if (!arr.length) return '';
    let res = '';
    for (let i = 0; i < arr.length; i++) res += fn(i, arr);
    return res;
}
