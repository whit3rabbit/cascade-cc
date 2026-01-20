import * as React from "react";
import { InternalStdinContext } from "./inkContexts.js";

/**
 * Hook to listen for terminal input and keys.
 * Deobfuscated from xe8 in chunk_205.ts.
 */
export function useInput(
    onInput: (input: string, key: any, event: any) => void,
    options: { isActive?: boolean } = {}
) {
    const { setRawMode, internal_eventEmitter, internal_exitOnCtrlC } = React.useContext(InternalStdinContext);

    React.useEffect(() => {
        if (options.isActive === false) return;
        setRawMode(true);
        return () => setRawMode(false);
    }, [options.isActive, setRawMode]);

    React.useEffect(() => {
        if (options.isActive === false) return;

        const handler = (event: any) => {
            const { input, key } = event;
            // Do not trigger onInput if it's Ctrl-C and exitOnCtrlC is enabled
            if (!(input === "c" && key.ctrl) || !internal_exitOnCtrlC) {
                onInput(input, key, event);
            }
        };

        internal_eventEmitter.on("input", handler);
        return () => {
            internal_eventEmitter.off("input", handler);
        };
    }, [options.isActive, internal_exitOnCtrlC, internal_eventEmitter, onInput]);
}
