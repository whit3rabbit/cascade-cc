import React from "react";
import { Box } from "ink";
import { Text } from "../../vendor/inkText.js";
import { formatBytes } from "../../services/terminal/settings.js";

interface FileReadStatusProps {
    type: "image" | "notebook" | "pdf" | "text";
    file: {
        originalSize?: number;
        numLines?: number; // for text
        cells?: any[]; // for notebook
    };
}

export const FileReadStatus: React.FC<FileReadStatusProps> = ({ type, file }) => {
    switch (type) {
        case "image": {
            const size = file.originalSize ? formatBytes(file.originalSize) : "unknown";
            return (
                <Box height={1}>
                    <Text>Read image ({size})</Text>
                </Box>
            );
        }
        case "notebook": {
            const cells = file.cells || [];
            if (cells.length < 1) {
                return <Text color="red">No cells found in notebook</Text>;
            }
            return (
                <Box height={1}>
                    <Text>Read <Text bold>{cells.length}</Text> cells</Text>
                </Box>
            );
        }
        case "pdf": {
            const size = file.originalSize ? formatBytes(file.originalSize) : "unknown";
            return (
                <Box height={1}>
                    <Text>Read PDF ({size})</Text>
                </Box>
            );
        }
        case "text": {
            const lines = file.numLines || 0;
            return (
                <Box height={1}>
                    <Text>Read <Text bold>{lines}</Text> {lines === 1 ? "line" : "lines"}</Text>
                </Box>
            );
        }
        default:
            return null;
    }
};
