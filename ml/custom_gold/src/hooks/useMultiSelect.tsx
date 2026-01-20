
import { isDeepStrictEqual } from "node:util";
import React, { useReducer, useState, useCallback, useMemo, useEffect } from "react";
import { useInput, Box, Text } from "ink";
import { figures } from "../vendor/terminalFigures.js";

// --- Multiselect Hook Logic ---

export interface SelectionNode<K, V> {
    value: K;
    next?: SelectionNode<K, V>;
    previous?: SelectionNode<K, V>;
    index: number;
    original: V;
}

export class DoublyLinkedListMap<K, V> extends Map<K, SelectionNode<K, V>> {
    first?: SelectionNode<K, V>;
    last?: SelectionNode<K, V>;

    constructor(items: Array<{ value: K } & V>) {
        const entries: Array<[K, SelectionNode<K, V>]> = [];
        let first: SelectionNode<K, V> | undefined;
        let last: SelectionNode<K, V> | undefined;
        let previous: SelectionNode<K, V> | undefined;
        let index = 0;

        for (const item of items) {
            const entry: SelectionNode<K, V> = {
                value: item.value,
                original: item,
                previous,
                next: undefined,
                index
            };
            if (previous) {
                previous.next = entry;
            }
            if (!first) first = entry;
            last = entry;
            entries.push([item.value, entry]);
            index++;
            previous = entry;
        }
        super(entries);
        this.first = first;
        this.last = last;
    }
}

export interface MultiSelectOption {
    label: string | React.ReactNode;
    value: any;
    type?: "input" | "option"; // Supported types
    description?: string;
}

interface MultiSelectState {
    optionMap: DoublyLinkedListMap<any, any>;
    visibleOptionCount: number;
    focusedValue: any;
    visibleFromIndex: number;
    visibleToIndex: number;
    previousValue: any[];
    value: any[];
    // Adding input state support if we want to follow chunk_389 more closely
    // inputValues: Map<any, string>; 
    // isSubmitFocused: boolean;
}

type MultiSelectAction =
    | { type: "focus-next-option" }
    | { type: "focus-previous-option" }
    | { type: "toggle-focused-option" }
    | { type: "reset"; state: MultiSelectState };

const multiSelectReducer = (state: MultiSelectState, action: MultiSelectAction): MultiSelectState => {
    switch (action.type) {
        case "focus-next-option": {
            if (!state.focusedValue) return state;
            const current = state.optionMap.get(state.focusedValue);
            if (!current) return state;

            const next = current.next || state.optionMap.first;
            if (!next) return state;

            if (!current.next && next === state.optionMap.first) {
                return {
                    ...state,
                    focusedValue: next.value,
                    visibleFromIndex: 0,
                    visibleToIndex: state.visibleOptionCount
                };
            }

            if (next.index >= state.visibleToIndex) {
                const nextTo = Math.min(state.optionMap.size, state.visibleToIndex + 1);
                const nextFrom = nextTo - state.visibleOptionCount;
                return {
                    ...state,
                    focusedValue: next.value,
                    visibleFromIndex: nextFrom,
                    visibleToIndex: nextTo
                };
            }

            return {
                ...state,
                focusedValue: next.value
            };
        }
        case "focus-previous-option": {
            if (!state.focusedValue) return state;
            const current = state.optionMap.get(state.focusedValue);
            if (!current) return state;

            const prev = current.previous || state.optionMap.last;
            if (!prev) return state;

            if (!current.previous && prev === state.optionMap.last) {
                const size = state.optionMap.size;
                const nextFrom = Math.max(0, size - state.visibleOptionCount);
                return {
                    ...state,
                    focusedValue: prev.value,
                    visibleFromIndex: nextFrom,
                    visibleToIndex: size
                };
            }

            if (prev.index < state.visibleFromIndex) {
                const nextFrom = Math.max(0, state.visibleFromIndex - 1);
                const nextTo = nextFrom + state.visibleOptionCount;
                return {
                    ...state,
                    focusedValue: prev.value,
                    visibleFromIndex: nextFrom,
                    visibleToIndex: nextTo
                };
            }

            return {
                ...state,
                focusedValue: prev.value
            };
        }
        case "toggle-focused-option": {
            if (!state.focusedValue) return state;
            const newValue = [...state.value];
            const index = newValue.indexOf(state.focusedValue);
            if (index !== -1) {
                newValue.splice(index, 1);
            } else {
                newValue.push(state.focusedValue);
            }
            return {
                ...state,
                previousValue: state.value,
                value: newValue
            };
        }
        case "reset":
            return action.state;
        default:
            return state;
    }
};

const initMultiSelectState = ({ visibleOptionCount, defaultValue, options }: { visibleOptionCount: number; defaultValue: any[]; options: any[] }): MultiSelectState => {
    const count = typeof visibleOptionCount === "number" ? Math.min(visibleOptionCount, options.length) : options.length;
    const map = new DoublyLinkedListMap(options);
    const initialValue = defaultValue ?? [];
    return {
        optionMap: map,
        visibleOptionCount: count,
        focusedValue: map.first?.value,
        visibleFromIndex: 0,
        visibleToIndex: count,
        previousValue: initialValue,
        value: initialValue
    };
};

export function useMultiSelect({
    visibleOptionCount = 5,
    options,
    defaultValue = [],
    onChange,
    onSubmit,
    onCancel,
    onFocus,
    focusValue,
    submitButtonText // Not used yet in reducer but part of interface
}: {
    visibleOptionCount?: number;
    options: MultiSelectOption[];
    defaultValue?: any[];
    onChange?: (value: any[]) => void;
    onSubmit?: (value: any[]) => void;
    onCancel?: () => void;
    onFocus?: (value: any) => void;
    focusValue?: any;
    submitButtonText?: string;
}) {
    const [state, dispatch] = useReducer(multiSelectReducer, { visibleOptionCount, defaultValue, options }, initMultiSelectState);
    const [lastOptions, setLastOptions] = useState(options);

    // Additional state for inputs and submit button
    const [inputValues, setInputValues] = useState<Map<any, string>>(new Map());
    const [isSubmitFocused, setIsSubmitFocused] = useState(false);

    if (options !== lastOptions && !isDeepStrictEqual(options, lastOptions)) {
        dispatch({
            type: "reset",
            state: initMultiSelectState({ visibleOptionCount, defaultValue, options })
        });
        setLastOptions(options);
    }

    const focusNextOption = useCallback(() => {
        if (!isSubmitFocused) {
            // Check if we are at the last option and submit button is enabled
            // Should add logic to move to submit button
        }
        dispatch({ type: "focus-next-option" });
    }, [isSubmitFocused]);

    const focusPreviousOption = useCallback(() => dispatch({ type: "focus-previous-option" }), []);
    const toggleFocusedOption = useCallback(() => dispatch({ type: "toggle-focused-option" }), []);
    const submit = useCallback(() => onSubmit?.(state.value), [state.value, onSubmit]);

    const updateInputValue = useCallback((key: any, val: string) => {
        setInputValues(prev => new Map(prev).set(key, val));
    }, []);

    const visibleOptions = useMemo(() => {
        return options
            .map((opt, idx) => ({ ...opt, index: idx }))
            .slice(state.visibleFromIndex, state.visibleToIndex);
    }, [options, state.visibleFromIndex, state.visibleToIndex]);

    useEffect(() => {
        if (!isDeepStrictEqual(state.previousValue, state.value)) {
            onChange?.(state.value);
        }
    }, [state.previousValue, state.value, onChange]);

    return {
        focusedValue: state.focusedValue,
        visibleFromIndex: state.visibleFromIndex,
        visibleToIndex: state.visibleToIndex,
        value: state.value,
        selectedValues: state.value,
        visibleOptions,
        inputValues,
        isSubmitFocused,
        updateInputValue,

        focusNextOption,
        focusPreviousOption,
        toggleFocusedOption,
        submit
    };
}

export function useMultiSelectInput({
    isDisabled = false,
    state
}: {
    isDisabled?: boolean;
    state: ReturnType<typeof useMultiSelect>;
}) {
    useInput((input, key) => {
        if (isDisabled) return;

        if (key.downArrow || (key.ctrl && input === "n") || (!key.ctrl && !key.shift && input === "j")) {
            state.focusNextOption();
        }
        if (key.upArrow || (key.ctrl && input === "p") || (!key.ctrl && !key.shift && input === "k")) {
            state.focusPreviousOption();
        }
        if (input === " ") {
            state.toggleFocusedOption();
        }
        if (key.return) {
            state.submit();
        }
    }, { isActive: !isDisabled });
}

export function MultiSelectItem({
    isFocused,
    isSelected,
    children
}: {
    isFocused: boolean;
    isSelected: boolean;
    children: React.ReactNode
}) {
    return (
        <Box gap= { 1} paddingLeft = { isFocused? 0: 2 } >
            { isFocused && (
                <Text color="blue" > { figures.pointer } </Text>
            )
}
<Text color={ isSelected ? "green" : (isFocused ? "blue" : undefined) }>
    { children }
    </Text>
{
    isSelected && (
        <Text color="green" > { figures.tick } </Text>
            )
}
</Box>
    );
}
