import * as React from "react";

/**
 * High-level Box component for Ink.
 * Deobfuscated from lU in chunk_201.ts.
 */
export const Box = React.forwardRef<any, any>(({
    children,
    flexWrap = "nowrap",
    flexDirection = "row",
    flexGrow = 0,
    flexShrink = 1,
    ...props
}, ref) => {
    return React.createElement("ink-box", {
        ref,
        style: {
            flexWrap,
            flexDirection,
            flexGrow,
            flexShrink,
            ...props,
            overflowX: props.overflowX ?? props.overflow ?? "visible",
            overflowY: props.overflowY ?? props.overflow ?? "visible"
        }
    }, children);
});

Box.displayName = "Box";
