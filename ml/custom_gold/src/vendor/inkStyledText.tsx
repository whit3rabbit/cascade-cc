import * as React from "react";
import { Text } from "./inkText.js";
import { Link } from "./inkLink.js";
import { ansiToSegments } from "./ansiToSegments.js";

interface StyledTextProps {
    children: React.ReactNode;
}

/**
 * Component that converts ANSI strings into nested Ink components.
 * Deobfuscated from _3 in chunk_205.ts.
 */
export const StyledText = React.memo<StyledTextProps>(({ children }) => {
    if (typeof children !== "string") {
        return <Text>{String(children)}</Text>;
    }

    if (children === "") return null;

    const segments = ansiToSegments(children);
    if (segments.length === 0) return null;

    if (segments.length === 1 && Object.keys(segments[0].props).length === 0) {
        return <Text>{segments[0].text}</Text>;
    }

    return (
        <Text>
            {segments.map((seg, i) => {
                const { hyperlink, ...styles } = seg.props;
                const hasStyles = Object.keys(styles).length > 0;

                let content: React.ReactNode = seg.text;
                if (hasStyles) {
                    content = <Text {...styles}>{seg.text}</Text>;
                }

                if (hyperlink) {
                    return <Link key={i} url={hyperlink}>{content}</Link>;
                }

                return <React.Fragment key={i}>{content}</React.Fragment>;
            })}
        </Text>
    );
});

StyledText.displayName = "StyledText";
