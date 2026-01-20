import React, { useMemo } from "react";
import { Text } from "../../vendor/inkText.js";
import { useTerminalSize } from "../../vendor/inkContexts.js";
import { foldContent, formatJsonInBlocks, stripUnderline } from "../../services/terminal/terminalUtils.js";
import { Indent } from "./Indent.js";

/**
 * Renders a block of output, potentially folded and with ANSI codes stripped.
 */
export const OutputBlock: React.FC<{
    content: string;
    verbose?: boolean;
    isError?: boolean;
    isWarning?: boolean;
}> = ({ content, verbose, isError, isWarning }) => {
    const { columns } = useTerminalSize();

    const formattedContent = useMemo(() => {
        const text = formatJsonInBlocks(content);
        if (verbose) return stripUnderline(text);
        return stripUnderline(foldContent(text, columns));
    }, [content, verbose, columns]);

    return (
        <Indent>
            <Text color={isError ? "error" : isWarning ? "warning" : undefined}>
                {formattedContent}
            </Text>
        </Indent>
    );
};
