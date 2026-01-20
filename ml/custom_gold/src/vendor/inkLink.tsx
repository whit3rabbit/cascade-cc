import * as React from "react";
import { RawText } from "./inkText.js";
import { hasLinkSupport } from "../utils/shared/terminalCapabilities.js";

interface LinkProps {
    url: string;
    fallback?: React.ReactNode;
    children?: React.ReactNode;
}

/**
 * Terminal hyperlink component with fallback support.
 * Deobfuscated from X9 in chunk_204.ts.
 */
export const Link: React.FC<LinkProps> = ({ children, url, fallback }) => {
    const label = children ?? url;

    if (hasLinkSupport()) {
        return (
            <RawText>
                {React.createElement("ink-link", { href: url }, label)}
            </RawText>
        );
    }

    return (
        <RawText>
            {fallback ?? label}
        </RawText>
    );
};
