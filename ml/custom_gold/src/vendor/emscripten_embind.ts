
/**
 * Emscripten Embind Logic (Duplicate of wasm_bindings_base.ts due to mapping)
 */

export function createWasmBindings(Module: any) {
    // Setup error types
    function BindingError(this: any, message: string) {
        this.name = "BindingError";
        this.message = message;
        this.stack = (new Error(message)).stack;
    }
    BindingError.prototype = Object.create(Error.prototype);
    BindingError.prototype.constructor = BindingError;
    Module.BindingError = BindingError;

    function InternalError(this: any, message: string) {
        this.name = "InternalError";
        this.message = message;
        this.stack = (new Error(message)).stack;
    }
    InternalError.prototype = Object.create(Error.prototype);
    InternalError.prototype.constructor = InternalError;
    Module.InternalError = InternalError;

    function UnboundTypeError(this: any, message: string) {
        this.name = "UnboundTypeError";
        this.message = message;
        this.stack = (new Error(message)).stack;
    }
    UnboundTypeError.prototype = Object.create(Error.prototype);
    UnboundTypeError.prototype.constructor = UnboundTypeError;
    Module.UnboundTypeError = UnboundTypeError;

    // Embind Type Registry
    const registeredTypes: Record<number, any> = {};
    const typeNames: Record<number, string> = {};

    Module.count_emval_handles = function () {
        return 0; // Stub
    };

    return Module;
}
