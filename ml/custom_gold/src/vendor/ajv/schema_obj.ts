import * as util from './util.js';

export default class SchemaObject {
    [key: string]: any;
    constructor(obj: any) {
        util.copy(obj, this);
    }
}
