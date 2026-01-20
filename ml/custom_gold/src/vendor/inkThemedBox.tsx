import * as React from "react";
import { Box } from "./inkBox.js";
import { useTheme } from "../services/terminal/themeManager.js";
import { getColorValue } from "../utils/shared/theme.js";

interface ThemedBoxProps {
    borderColor?: string;
    borderTopColor?: string;
    borderBottomColor?: string;
    borderLeftColor?: string;
    borderRightColor?: string;
    children?: React.ReactNode;
    [key: string]: any;
}

/**
 * Box component that resolves border colors via theme.
 * Deobfuscated from nWB in chunk_204.ts.
 */
export const ThemedBox = React.forwardRef<any, ThemedBoxProps>(({
    borderColor,
    borderTopColor,
    borderBottomColor,
    borderLeftColor,
    borderRightColor,
    children,
    ...props
}, ref) => {
    const [theme] = useTheme();

    const resolvedBorderColor = getColorValue(borderColor, theme!);
    const resolvedTopColor = getColorValue(borderTopColor, theme!);
    const resolvedBottomColor = getColorValue(borderBottomColor, theme!);
    const resolvedLeftColor = getColorValue(borderLeftColor, theme!);
    const resolvedRightColor = getColorValue(borderRightColor, theme!);

    return (
        <Box
            ref={ref}
            borderColor={resolvedBorderColor}
            borderTopColor={resolvedTopColor}
            borderBottomColor={resolvedBottomColor}
            borderLeftColor={resolvedLeftColor}
            borderRightColor={resolvedRightColor}
            {...props}
        >
            {children}
        </Box>
    );
});

ThemedBox.displayName = "ThemedBox";
