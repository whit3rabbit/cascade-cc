import * as React from "react";
import { Box } from "./inkBox.js";
import { InternalAppContext } from "./inkContexts.js";

interface StaticProps<T> {
    items: T[];
    children: (item: T, index: number) => React.ReactNode;
}

/**
 * Renders children in a static (one-time) block.
 * Deobfuscated from Si in chunk_205.ts.
 */
export function Static<T>({ items, children }: StaticProps<T>) {
    // Static logic usually involves tracking which items have already been "flushed"
    // In chunk_205 it uses a state to slice items
    const [lastIndex, setLastIndex] = React.useState(0);
    const newItems = items.slice(lastIndex);

    React.useLayoutEffect(() => {
        setLastIndex(items.length);
    }, [items.length]);

    return (
        <Box internal_static={true as any} style={{ position: "absolute", flexDirection: "column" } as any}>
            {newItems.map((item, i) => children(item, lastIndex + i))}
        </Box>
    );
}
