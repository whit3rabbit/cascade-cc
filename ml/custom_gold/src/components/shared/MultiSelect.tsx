import React from "react";
import { Box, Text } from "ink";
import { figures } from "../../vendor/terminalFigures.js";
import { useMultiSelect, MultiSelectOption } from "../../hooks/useMultiSelect.js";
import TextInput from "ink-text-input";

interface InputOptionProps {
    option: MultiSelectOption;
    isFocused: boolean;
    isSelected: boolean;
    shouldShowDownArrow: boolean;
    shouldShowUpArrow: boolean;
    maxIndexWidth: number;
    index: number;
    inputValue: string;
    onInputChange: (value: string) => void;
    onSubmit: () => void;
    onExit: () => void;
    layout: "compact" | "normal";
    children: React.ReactNode;
}

function InputOption({
    option,
    isFocused,
    shouldShowDownArrow,
    shouldShowUpArrow,
    maxIndexWidth,
    index,
    inputValue,
    onInputChange,
    onSubmit,
    onExit,
    children
}: InputOptionProps) {
    return (
        <Box gap={1}>
            <Box>
                <Text color="dimColor">
                    {shouldShowUpArrow ? figures.arrowUp : shouldShowDownArrow ? figures.arrowDown : " "}
                </Text>
            </Box>
            <Box>
                <Text color="dimColor">{`${index}.`.padEnd(maxIndexWidth + 1)}</Text>
            </Box>
            {children}
            <Box>
                <Text color={isFocused ? "blue" : undefined}>{option.label}: </Text>
                {isFocused ? (
                    <TextInput
                        value={inputValue}
                        onChange={onInputChange}
                        onSubmit={onSubmit}
                    />
                ) : (
                    <Text>{inputValue}</Text>
                )}
            </Box>
        </Box>
    );
}

interface SelectOptionProps {
    isFocused: boolean;
    isSelected: boolean;
    shouldShowDownArrow: boolean;
    shouldShowUpArrow: boolean;
    description?: string;
    children: React.ReactNode;
}

function SelectOption({
    isFocused,
    shouldShowDownArrow,
    shouldShowUpArrow,
    description,
    children
}: SelectOptionProps) {
    return (
        <Box flexDirection="column">
            <Box gap={1}>
                <Box>
                    <Text color="dimColor">
                        {shouldShowUpArrow ? figures.arrowUp : shouldShowDownArrow ? figures.arrowDown : " "}
                    </Text>
                </Box>
                {children}
            </Box>
            {description && isFocused && (
                <Box marginLeft={4}>
                    <Text color="dimColor">{description}</Text>
                </Box>
            )}
        </Box>
    );
}

export interface MultiSelectProps {
    isDisabled?: boolean;
    visibleOptionCount?: number;
    options: MultiSelectOption[];
    defaultValue?: string[];
    onCancel: () => void;
    onChange?: (values: string[]) => void;
    onFocus?: (value: string) => void;
    focusValue?: string;
    submitButtonText?: string;
    onSubmit?: () => void;
}

export function MultiSelect({
    isDisabled = false,
    visibleOptionCount = 5,
    options,
    defaultValue = [],
    onCancel,
    onChange,
    onFocus,
    focusValue,
    submitButtonText,
    onSubmit
}: MultiSelectProps) {
    const {
        visibleOptions,
        focusedValue,
        selectedValues,
        inputValues,
        isSubmitFocused,
        updateInputValue,
        visibleFromIndex,
        visibleToIndex
    } = useMultiSelect({
        visibleOptionCount,
        options,
        defaultValue,
        onChange,
        onCancel,
        onFocus,
        focusValue,
        onSubmit,
        submitButtonText
    });

    const maxIndexWidth = String(options.length).length;

    return (
        <Box flexDirection="column">
            <Box flexDirection="column">
                {visibleOptions.map((option, index) => {
                    // Cast focusedValue to match option.value type if needed
                    const isFocused = focusedValue === option.value && !isSubmitFocused;
                    const isSelected = selectedValues.includes(option.value);
                    const isFirstVisible = option.index === visibleFromIndex; // Note: option includes index property added by hook
                    const isLastVisible = option.index === visibleToIndex - 1;
                    const canScrollDown = visibleToIndex < options.length;
                    const canScrollUp = visibleFromIndex > 0;
                    const displayIndex = visibleFromIndex + index + 1;

                    if (option.type === "input") {
                        const inputValue = inputValues.get(option.value) || "";
                        return (
                            <Box key={String(option.value)} gap={1}>
                                <InputOption
                                    option={option}
                                    isFocused={isFocused}
                                    isSelected={false}
                                    shouldShowDownArrow={canScrollDown && isLastVisible}
                                    shouldShowUpArrow={canScrollUp && isFirstVisible}
                                    maxIndexWidth={maxIndexWidth}
                                    index={displayIndex}
                                    inputValue={inputValue}
                                    onInputChange={(val) => updateInputValue(option.value, val)}
                                    onSubmit={() => { }}
                                    onExit={onCancel}
                                    layout="compact"
                                >
                                    <Text color={isSelected ? "green" : undefined}>
                                        [{isSelected ? figures.tick : " "}]
                                    </Text>
                                </InputOption>
                            </Box>
                        );
                    }

                    return (
                        <Box key={String(option.value)} gap={1}>
                            <SelectOption
                                isFocused={isFocused}
                                isSelected={false}
                                shouldShowDownArrow={canScrollDown && isLastVisible}
                                shouldShowUpArrow={canScrollUp && isFirstVisible}
                                description={option.description}
                            >
                                <Text color="dimColor">{`${displayIndex}.`.padEnd(maxIndexWidth + 1)}</Text>
                                <Text color={isSelected ? "green" : undefined}>
                                    [{isSelected ? figures.tick : " "}]
                                </Text>
                                <Text color={isFocused ? "blue" : undefined}> {option.label}</Text>
                            </SelectOption>
                        </Box>
                    );
                })}
            </Box>
            {submitButtonText && onSubmit && (
                <Box marginTop={0} gap={1}>
                    {isSubmitFocused ? (
                        <Text color="blue">{figures.pointer}</Text>
                    ) : (
                        <Text> </Text>
                    )}
                    <Box marginLeft={3}>
                        <Text color={isSubmitFocused ? "blue" : undefined} bold>
                            {submitButtonText}
                        </Text>
                    </Box>
                </Box>
            )}
        </Box>
    );
}
