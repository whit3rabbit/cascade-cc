
/**
 * Emscripten WASM Loader (Deobfuscated from chunk_702.ts)
 */

import { createWasmBindings } from './wasm_bindings_base.js';

export async function loadWasmModule(moduleConfig: any = {}) {
    const Module: any = moduleConfig;

    // Check if we need to fetch the binary (if strictly matching chunk_702, it has fetch logic)
    // Here we assume node environment or standard Module.wasmBinary / locateFile usage

    // Create base Module if not exists
    if (!Module.print) Module.print = console.log;
    if (!Module.printErr) Module.printErr = console.error;

    // Memory setup
    if (!Module.wasmMemory) {
        Module.wasmMemory = new WebAssembly.Memory({ initial: 256, maximum: 2147483648 / 65536 });
    }
    Module.buffer = Module.wasmMemory.buffer;

    // Initialize heap views
    Module.HEAPU8 = new Uint8Array(Module.buffer);
    Module.HEAP8 = new Int8Array(Module.buffer);
    Module.HEAP16 = new Int16Array(Module.buffer);
    Module.HEAPU16 = new Uint16Array(Module.buffer);
    Module.HEAP32 = new Int32Array(Module.buffer);
    Module.HEAPU32 = new Uint32Array(Module.buffer);
    Module.HEAPF32 = new Float32Array(Module.buffer);
    Module.HEAPF64 = new Float64Array(Module.buffer);

    // Get Embind imports
    const wasmImports = createWasmBindings(Module); // This corresponds to 'h4' in chunk_702

    // Prepare import object
    const imports = {
        env: wasmImports,
        wasi_snapshot_preview1: wasmImports // Often needed for WASI
    };

    // Instantiate
    // We expect Module.wasmBinary or Module.instantiateWasm to be handled.
    // If not, we do standard instantiation.
    let instance: any;
    if (Module.wasmBinary) {
        instance = await WebAssembly.instantiate(Module.wasmBinary, imports);
    } else {
        // Fallback or error?
        // chunk_702 SA() function fetches.
        // We leave this as generic loader logic for now.
        throw new Error("Wasm binary not provided in module config");
    }

    // Wiring
    if (instance.instance) instance = instance.instance;
    Module.asm = instance.exports;

    // Call onRuntimeInitialized
    if (Module.onRuntimeInitialized) Module.onRuntimeInitialized();

    return Module;
}
