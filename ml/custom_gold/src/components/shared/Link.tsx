
import React from "react";
import { Text } from "ink";

export function Link({ url, children }: { url: string; children: React.ReactNode }) {
    return (
        <Text color="blue" underline>
            {children || url}
        </Text>
    );
}
