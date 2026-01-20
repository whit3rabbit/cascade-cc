import React, { createContext, useContext } from "react";
import { Box } from "ink";
import { Text } from "../../vendor/inkText.js";

const IndentContext = createContext<boolean>(false);

interface IndentProps {
    children: React.ReactNode;
    height?: number;
}

/**
 * Component to indent content with a visual "⎿" prefix.
 */
export const Indent: React.FC<IndentProps> = ({ children, height }) => {
    const isNested = useContext(IndentContext);

    if (isNested) {
        return <>{children}</>;
    }

    return (
        <IndentContext.Provider value={true}>
            <Box flexDirection="row" height={height} overflowY="hidden">
                <Text>  ⎿  </Text>
                {children}
            </Box>
        </IndentContext.Provider>
    );
};

export const IndentProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    return (
        <IndentContext.Provider value={true}>
            {children}
        </IndentContext.Provider>
    );
};
