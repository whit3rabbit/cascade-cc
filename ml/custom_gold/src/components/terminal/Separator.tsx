
import React from "react";
import { Box, Text } from "ink";

export interface SeparatorProps {
    orientation?: "horizontal" | "vertical";
    width?: number | string;
    dividerChar?: string;
    dividerColor?: string;
    dividerDimColor?: boolean;
    title?: string;
    padding?: number;
    titlePadding?: number;
    titleColor?: string;
    titleDimColor?: boolean;
}

export const Separator: React.FC<SeparatorProps> = ({
    orientation = "horizontal",
    width = "100%",
    dividerChar,
    dividerColor,
    dividerDimColor = true,
    title,
    padding = 0,
    titlePadding = 1,
    titleColor = "text",
    titleDimColor = true
}) => {
    const isVertical = orientation === "vertical";
    const char = dividerChar || (isVertical ? "│" : "─");

    if (isVertical) {
        return (
            <Box
                height="100%"
                borderStyle={{
                    topLeft: "",
                    top: "",
                    topRight: "",
                    right: char,
                    bottomRight: "",
                    bottom: "",
                    bottomLeft: "",
                    left: ""
                }}
                borderColor={dividerColor}
                borderDimColor={dividerDimColor}
                borderBottom={false}
                borderTop={false}
                borderLeft={false}
                borderRight={true}
            />
        );
    }

    const divider = (
        <Box
            width={title ? "auto" : width}
            borderStyle={{
                topLeft: "",
                top: "",
                topRight: "",
                right: "",
                bottomRight: "",
                bottom: char,
                bottomLeft: "",
                left: ""
            }}
            borderColor={dividerColor}
            borderDimColor={dividerDimColor}
            flexGrow={1}
            borderBottom={true}
            borderTop={false}
            borderLeft={false}
            borderRight={false}
        />
    );

    if (!title) {
        return (
            <Box paddingX={padding} width={width}>
                {divider}
            </Box>
        );
    }

    return (
        <Box
            width={width}
            paddingX={padding}
            flexDirection="row"
            alignItems="center"
            gap={titlePadding}
        >
            {divider}
            <Box>
                <Text color={titleColor} dimColor={titleDimColor}>
                    {title}
                </Text>
            </Box>
            {divider}
        </Box>
    );
};
