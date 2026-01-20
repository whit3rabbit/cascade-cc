
import React, { useState, useEffect, useMemo } from 'react';
import { Box, Text, useInput } from 'ink';

// Mocking dC0 and other utils
const validate = (val: string, schema: any) => ({ isValid: true, value: val });

export function McpInputForm({
    serverName,
    request,
    onResponse,
    signal
}: any) {
    const { message, requestedSchema } = request;
    const [status, setStatus] = useState<"accept" | "decline" | null>(null);
    const [answers, setAnswers] = useState<Record<string, any>>(() => {
        const initial: any = {};
        if (requestedSchema.properties) {
            for (const [key, schema] of Object.entries(requestedSchema.properties) as any) {
                if (schema.default !== undefined) initial[key] = schema.default;
            }
        }
        return initial;
    });

    const fields = useMemo(() => {
        const required = requestedSchema.required || [];
        return Object.entries(requestedSchema.properties || {}).map(([name, schema]: any) => ({
            name,
            schema,
            isRequired: required.includes(name)
        }));
    }, [requestedSchema]);

    const [activeIndex, setActiveIndex] = useState(0);

    // Ink input handling would go here...
    // ... logic from ed2 ...

    return (
        <Box flexDirection="column" borderStyle="round" borderColor="cyan" padding={1}>
            <Text bold>MCP Server "{serverName}" requests your input</Text>
            <Box padding={1}>
                <Text>{message}</Text>
            </Box>

            {fields.map((field, index) => (
                <Box key={field.name} flexDirection="column">
                    <Text color={index === activeIndex ? "green" : undefined}>
                        {index === activeIndex ? "> " : "  "}
                        {field.name}{field.isRequired ? "*" : ""}: {String(answers[field.name] ?? "<unset>")}
                    </Text>
                    {field.schema.description && (
                        <Box marginLeft={2}>
                            <Text dimColor>{field.schema.description}</Text>
                        </Box>
                    )}
                </Box>
            ))}

            <Box marginTop={1}>
                <Text bold color={status === "accept" ? "green" : undefined}>Accept</Text>
                <Text bold color={status === "decline" ? "red" : undefined}>Decline</Text>
            </Box>
        </Box>
    );
}

export default McpInputForm;
