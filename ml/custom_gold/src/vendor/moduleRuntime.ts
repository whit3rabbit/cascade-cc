/**
 * Bundler runtime helpers for deobfuscated code.
 */

// __commonJS
export const U = (A: any, Q?: any) => () => (Q || A((Q = {
    exports: {}
}).exports, Q), Q.exports);

// __esm
export const O = (A: any, Q?: any) => () => (A && (Q = A(A = 0)), Q);

// __name
export const __name = (target: any, value: string) => Object.defineProperty(target, "name", { value, configurable: true });

// __export
export const M5 = (A: any, Q: any) => {
    for (var B in Q) Object.defineProperty(A, B, {
        get: Q[B],
        enumerable: !0,
        configurable: !0,
        set: (G) => Q[B] = () => G
    })
};
