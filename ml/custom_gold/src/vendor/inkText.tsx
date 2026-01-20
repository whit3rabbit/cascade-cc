import * as React from "react";
import { useTheme } from "../services/terminal/themeManager.js";
import { getColorValue } from "../utils/shared/theme.js";

export interface TextProps {
    color?: string;
    backgroundColor?: string;
    dimColor?: boolean;
    bold?: boolean;
    italic?: boolean;
    underline?: boolean;
    strikethrough?: boolean;
    inverse?: boolean;
    wrap?: "wrap" | "truncate" | "truncate-end" | "truncate-start" | "truncate-middle" | "anywhere";
    children: React.ReactNode;
}

/**
 * Low-level wrapper for ink-text.
 * Deobfuscated from iU in chunk_202.ts.
 */
export const RawText: React.FC<any> = ({ children, ...props }) => {
    if (children === undefined || children === null) return null;
    return React.createElement("ink-text", props, children);
};

/**
 * Styled text that automatically resolves theme colors.
 * Deobfuscated from C in chunk_202.ts.
 */
export const Text: React.FC<TextProps> = ({
    color,
    backgroundColor,
    dimColor = false,
    bold = false,
    italic = false,
    underline = false,
    strikethrough = false,
    inverse = false,
    wrap = "wrap",
    children
}) => {
    const [theme] = useTheme();

    const resolvedColor = dimColor ? theme?.inactive : getColorValue(color, theme!);
    const resolvedBgColor = getColorValue(backgroundColor, theme!);

    return (
        <RawText
            style={{
                flexGrow: 0,
                flexShrink: 1,
                flexDirection: "row",
                textWrap: wrap
            }}
            textStyles={{
                ...(resolvedColor && { color: resolvedColor }),
                ...(resolvedBgColor && { backgroundColor: resolvedBgColor }),
                dim: dimColor,
                bold,
                italic,
                underline,
                strikethrough,
                inverse
            }}
        >
            {children}
        </RawText>
    );
};
