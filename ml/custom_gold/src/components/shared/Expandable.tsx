import React, { createContext, useContext } from "react";
import { Text } from "../../vendor/inkText.js";
import { Shortcut } from "./Shortcut.js";

const ExpandableContext = createContext<boolean>(false);

/**
 * Provides a context to mark content as expandable.
 */
export const ExpandableProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => (
    <ExpandableContext.Provider value={true}>
        {children}
    </ExpandableContext.Provider>
);

/**
 * Renders a hint to expand content if it's within an expandable context.
 */
export const ExpandHint: React.FC = () => {
    const isExpandable = useContext(ExpandableContext);
    if (isExpandable) return null;

    return (
        <Text dimColor>
            <Shortcut shortcut="ctrl+o" action="expand" parens />
        </Text>
    );
};
