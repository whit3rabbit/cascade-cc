import * as uri from './uri.js';
import equal from 'fast-deep-equal';
import * as util from './util.js';
import SchemaObject from './schema_obj.js';
import traverse from './traverse.js';

export default function resolve(this: any, compile: any, root: any, ref: any): any {
    let refVal = this._refs[ref];
    if (typeof refVal === 'string') {
        if (this._refs[refVal]) refVal = this._refs[refVal];
        else return resolve.call(this, compile, root, refVal);
    }

    refVal = refVal || this._schemas[ref];
    if (refVal instanceof SchemaObject) {
        return inlineRef(refVal.schema, this._opts.inlineRefs)
            ? refVal.schema
            : (refVal.validate || this._compile(refVal));
    }

    const res = resolveSchema.call(this, root, ref);
    let schema: any, validate: any;
    if (res) {
        schema = res.schema;
        root = res.root;
        const baseId = res.baseId;
        if (schema instanceof SchemaObject) {
            validate = schema.validate || compile.call(this, schema.schema, root, undefined, baseId);
        } else if (schema !== undefined) {
            validate = inlineRef(schema, this._opts.inlineRefs)
                ? schema
                : compile.call(this, schema, root, undefined, baseId);
        }
    }
    return validate;
}

resolve.normalizeId = normalizeId;
resolve.fullPath = fullPath;
resolve.url = url;
resolve.ids = getIds;
resolve.inlineRef = inlineRef;
resolve.schema = resolveSchema;

function resolveSchema(this: any, root: any, ref: any): any {
    const p = uri.parse(ref);
    const schemaId = getFullPath(p);
    const baseId = fullPath(this._getId(root.schema));

    if (Object.keys(root.schema).length === 0 || schemaId !== baseId) {
        const id = normalizeId(schemaId);
        let refVal = this._refs[id];
        if (typeof refVal === 'string') {
            return resolveMissingSchema.call(this, root, refVal, p);
        } else if (refVal instanceof SchemaObject) {
            if (!refVal.validate) this._compile(refVal);
            root = refVal;
        } else {
            refVal = this._schemas[id];
            if (refVal instanceof SchemaObject) {
                if (!refVal.validate) this._compile(refVal);
                if (id === normalizeId(ref)) {
                    return { schema: refVal, root: root, baseId: baseId };
                }
                root = refVal;
            } else {
                return;
            }
        }
        if (!root.schema) return;
        const newBaseId = fullPath(this._getId(root.schema));
        return getRes.call(this, p, newBaseId, root.schema, root);
    }
    return getRes.call(this, p, baseId, root.schema, root);
}

function resolveMissingSchema(this: any, root: any, refVal: any, p: any): any {
    const res = resolveSchema.call(this, root, refVal);
    if (res) {
        let { schema, baseId } = res;
        root = res.root;
        const id = this._getId(schema);
        if (id) baseId = url(baseId, id);
        return getRes.call(this, p, baseId, schema, root);
    }
}

const RULES_WITH_SCHEMAS = util.toHash(['properties', 'patternProperties', 'enum', 'dependencies', 'definitions']);

function getRes(this: any, p: any, baseId: any, schema: any, root: any) {
    p.fragment = p.fragment || '';
    if (p.fragment.slice(0, 1) !== '/') return;
    const parts = p.fragment.split('/');
    for (let i = 1; i < parts.length; i++) {
        const part = parts[i];
        if (part) {
            const partUnescaped = util.unescapeFragment(part);
            schema = schema[partUnescaped];
            if (schema === undefined) break;
            let id;
            if (!RULES_WITH_SCHEMAS[partUnescaped]) {
                id = this._getId(schema);
                if (id) baseId = url(baseId, id);
                if (schema.$ref) {
                    const ref = url(baseId, schema.$ref);
                    const res = resolveSchema.call(this, root, ref);
                    if (res) {
                        schema = res.schema;
                        root = res.root;
                        baseId = res.baseId;
                    }
                }
            }
        }
    }
    if (schema !== undefined && schema !== root.schema) {
        return { schema, root, baseId };
    }
}

const SCHEMA_KEYWORDS = util.toHash(['type', 'format', 'pattern', 'maxLength', 'minLength', 'maxProperties', 'minProperties', 'maxItems', 'minItems', 'maximum', 'minimum', 'uniqueItems', 'multipleOf', 'required', 'enum']);

function inlineRef(schema: any, limit: any) {
    if (limit === false) return false;
    if (limit === undefined || limit === true) return isSimple(schema);
    return countKeywords(schema) <= limit;
}

function isSimple(schema: any) {
    if (Array.isArray(schema)) {
        for (let i = 0; i < schema.length; i++) {
            const sch = schema[i];
            if (typeof sch === 'object' && !isSimple(sch)) return false;
        }
    } else {
        for (const key in schema) {
            if (key === '$ref') return false;
            const sch = schema[key];
            if (typeof sch === 'object' && !isSimple(sch)) return false;
        }
    }
    return true;
}

function countKeywords(schema: any) {
    let count = 0;
    if (Array.isArray(schema)) {
        for (let i = 0; i < schema.length; i++) {
            const sch = schema[i];
            if (typeof sch === 'object') count += countKeywords(sch);
            if (count === Infinity) return Infinity;
        }
    } else {
        for (const key in schema) {
            if (key === '$ref') return Infinity;
            if (SCHEMA_KEYWORDS[key]) count++;
            else {
                const sch = schema[key];
                if (typeof sch === 'object') count += countKeywords(sch) + 1;
                if (count === Infinity) return Infinity;
            }
        }
    }
    return count;
}

function fullPath(id: string, normalize?: boolean) {
    if (normalize !== false) id = normalizeId(id);
    const p = uri.parse(id);
    return getFullPath(p);
}

function getFullPath(p: any) {
    return uri.serialize(p).split('#')[0] + '#';
}

const TRAILING_HASH = /#\/?$/;

function normalizeId(id: string) {
    return id ? id.replace(TRAILING_HASH, '') : '';
}

function url(base: string, id: string) {
    id = normalizeId(id);
    return uri.resolve(base, id);
}

function getIds(this: any, schema: any) {
    const schemaId = normalizeId(this._getId(schema));
    const baseIds: { [key: string]: string } = { '': schemaId };
    const fullPaths: { [key: string]: string } = { '': fullPath(schemaId, false) };
    const ids: { [id: string]: any } = {};
    const self = this;

    traverse(schema, { allKeys: true }, (sch: any, jsonPtr: string, rootSchema: any, parentPtr?: string, parentKeyword?: string, parentSchema?: any, keyIndex?: string | number) => {
        if (jsonPtr === '') return;
        const id = self._getId(sch);
        let baseId = baseIds[parentPtr || ''];
        let fullP = fullPaths[parentPtr || ''] + '/' + parentKeyword;
        if (keyIndex !== undefined) {
            fullP += '/' + (typeof keyIndex === 'number' ? keyIndex : util.escapeFragment(String(keyIndex)));
        }

        if (typeof id === 'string') {
            baseId = normalizeId(baseId ? uri.resolve(baseId, id) : id);
            let ref = self._refs[baseId];
            if (typeof ref === 'string') ref = self._refs[ref];
            if (ref && ref.schema) {
                if (!equal(sch, ref.schema)) throw new Error('id "' + baseId + '" resolves to more than one schema');
            } else if (baseId !== normalizeId(fullP)) {
                if (baseId[0] === '#') {
                    if (ids[baseId] && !equal(sch, ids[baseId])) throw new Error('id "' + baseId + '" resolves to more than one schema');
                    ids[baseId] = sch;
                } else {
                    self._refs[baseId] = fullP;
                }
            }
        }
        baseIds[jsonPtr] = baseId;
        fullPaths[jsonPtr] = fullP;
    });

    return ids;
}
