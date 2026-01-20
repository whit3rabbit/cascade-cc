import * as React from "react";
import { MenuLinkedList, MenuItem } from "../../utils/shared/linkedList.js";

interface SelectState<T> {
    visibleOptionCount: number;
    optionMap: MenuLinkedList<T>;
    focusedValue?: T;
    visibleFromIndex: number;
    visibleToIndex: number;
}

type SelectAction<T> =
    | { type: "focus-next-option" }
    | { type: "focus-previous-option" }
    | { type: "focus-next-page" }
    | { type: "focus-previous-page" }
    | { type: "reset"; state: SelectState<T> }
    | { type: "set-focus"; value: T };

function selectReducer<T>(state: SelectState<T>, action: SelectAction<T>): SelectState<T> {
    const { optionMap, focusedValue, visibleOptionCount, visibleFromIndex, visibleToIndex } = state;

    switch (action.type) {
        case "focus-next-option": {
            if (focusedValue === undefined) return state;
            const current = optionMap.get(focusedValue);
            if (!current) return state;

            const next = current.next || optionMap.first;
            if (!next) return state;

            // Wrap around
            if (!current.next && next === optionMap.first) {
                return {
                    ...state,
                    focusedValue: next.value,
                    visibleFromIndex: 0,
                    visibleToIndex: visibleOptionCount
                };
            }

            if (next.index < visibleToIndex) {
                return { ...state, focusedValue: next.value };
            }

            const newToIndex = Math.min(optionMap.size, visibleToIndex + 1);
            const newFromIndex = newToIndex - visibleOptionCount;
            return {
                ...state,
                focusedValue: next.value,
                visibleFromIndex: newFromIndex,
                visibleToIndex: newToIndex
            };
        }

        case "focus-previous-option": {
            if (focusedValue === undefined) return state;
            const current = optionMap.get(focusedValue);
            if (!current) return state;

            const prev = current.previous || optionMap.last;
            if (!prev) return state;

            // Wrap around
            if (!current.previous && prev === optionMap.last) {
                const size = optionMap.size;
                return {
                    ...state,
                    focusedValue: prev.value,
                    visibleFromIndex: Math.max(0, size - visibleOptionCount),
                    visibleToIndex: size
                };
            }

            if (prev.index >= visibleFromIndex) {
                return { ...state, focusedValue: prev.value };
            }

            const newFromIndex = Math.max(0, visibleFromIndex - 1);
            const newToIndex = newFromIndex + visibleOptionCount;
            return {
                ...state,
                focusedValue: prev.value,
                visibleFromIndex: newFromIndex,
                visibleToIndex: newToIndex
            };
        }

        case "focus-next-page": {
            if (focusedValue === undefined) return state;
            const current = optionMap.get(focusedValue);
            if (!current) return state;

            const nextIndex = Math.min(optionMap.size - 1, current.index + visibleOptionCount);
            let next = optionMap.first;
            while (next && next.index < nextIndex) {
                if (next.next) next = next.next; else break;
            }
            if (!next) return state;

            const newToIndex = Math.min(optionMap.size, next.index + 1);
            const newFromIndex = Math.max(0, newToIndex - visibleOptionCount);
            return {
                ...state,
                focusedValue: next.value,
                visibleFromIndex: newFromIndex,
                visibleToIndex: newToIndex
            };
        }

        case "focus-previous-page": {
            if (focusedValue === undefined) return state;
            const current = optionMap.get(focusedValue);
            if (!current) return state;

            const prevIndex = Math.max(0, current.index - visibleOptionCount);
            let next = optionMap.first;
            while (next && next.index < prevIndex) {
                if (next.next) next = next.next; else break;
            }
            if (!next) return state;

            const newFromIndex = Math.max(0, next.index);
            const newToIndex = Math.min(optionMap.size, newFromIndex + visibleOptionCount);
            return {
                ...state,
                focusedValue: next.value,
                visibleFromIndex: newFromIndex,
                visibleToIndex: newToIndex
            };
        }

        case "set-focus": {
            if (focusedValue === action.value) return state;
            const target = optionMap.get(action.value);
            if (!target) return state;

            if (target.index >= visibleFromIndex && target.index < visibleToIndex) {
                return { ...state, focusedValue: action.value };
            }

            let from, to;
            if (target.index < visibleFromIndex) {
                from = target.index;
                to = Math.min(optionMap.size, from + visibleOptionCount);
            } else {
                to = Math.min(optionMap.size, target.index + 1);
                from = Math.max(0, to - visibleOptionCount);
            }

            return {
                ...state,
                focusedValue: action.value,
                visibleFromIndex: from,
                visibleToIndex: to
            };
        }

        case "reset":
            return action.state;
    }
}

function initState<T>({
    visibleOptionCount,
    options,
    initialFocusValue,
    currentViewport
}: {
    visibleOptionCount: number;
    options: Array<{ label: string; value: T; description?: string; type?: string }>;
    initialFocusValue?: T;
    currentViewport?: { visibleFromIndex: number; visibleToIndex: number };
}): SelectState<T> {
    const count = typeof visibleOptionCount === "number" ? Math.min(visibleOptionCount, options.length) : options.length;
    const map = new MenuLinkedList<T>(options);
    const targetNode = initialFocusValue !== undefined ? map.get(initialFocusValue) : undefined;
    const effectiveFocus = targetNode ? initialFocusValue : map.first?.value;

    let from = 0;
    let to = count;

    if (targetNode) {
        const idx = targetNode.index;
        if (currentViewport) {
            if (idx >= currentViewport.visibleFromIndex && idx < currentViewport.visibleToIndex) {
                from = currentViewport.visibleFromIndex;
                to = Math.min(map.size, currentViewport.visibleToIndex);
            } else if (idx < currentViewport.visibleFromIndex) {
                from = idx;
                to = Math.min(map.size, from + count);
            } else {
                to = Math.min(map.size, idx + 1);
                from = Math.max(0, to - count);
            }
        } else if (idx >= count) {
            to = Math.min(map.size, idx + 1);
            from = Math.max(0, to - count);
        }
        from = Math.max(0, Math.min(from, map.size - 1));
        to = Math.min(map.size, Math.max(count, to));
    }

    return {
        optionMap: map,
        visibleOptionCount: count,
        focusedValue: effectiveFocus,
        visibleFromIndex: from,
        visibleToIndex: to
    };
}

/**
 * Hook for managing selectable menu state and viewport.
 * Deobfuscated from weA in chunk_206.ts.
 */
export function useSelectBase<T>({
    visibleOptionCount = 5,
    options,
    initialFocusValue,
    onFocus,
    focusValue
}: {
    visibleOptionCount?: number;
    options: Array<{ label: string; value: T; description?: string; type?: string }>;
    initialFocusValue?: T;
    onFocus?: (value: T) => void;
    focusValue?: T;
}) {
    const [state, dispatch] = React.useReducer(
        selectReducer as any,
        { visibleOptionCount, options, initialFocusValue: focusValue || initialFocusValue },
        initState as any
    ) as [SelectState<T>, React.Dispatch<SelectAction<T>>];

    const [prevOptions, setPrevOptions] = React.useState(options);

    if (options !== prevOptions) {
        dispatch({
            type: "reset",
            state: initState({
                visibleOptionCount,
                options,
                initialFocusValue: focusValue ?? state.focusedValue ?? initialFocusValue,
                currentViewport: {
                    visibleFromIndex: state.visibleFromIndex,
                    visibleToIndex: state.visibleToIndex
                }
            })
        });
        setPrevOptions(options);
    }

    const focusNextOption = React.useCallback(() => dispatch({ type: "focus-next-option" }), []);
    const focusPreviousOption = React.useCallback(() => dispatch({ type: "focus-previous-option" }), []);
    const focusNextPage = React.useCallback(() => dispatch({ type: "focus-next-page" }), []);
    const focusPreviousPage = React.useCallback(() => dispatch({ type: "focus-previous-page" }), []);
    const focusOption = React.useCallback((value: T) => dispatch({ type: "set-focus", value }), []);

    const visibleOptions = React.useMemo(() => {
        return options
            .map((opt, i) => ({ ...opt, index: i }))
            .slice(state.visibleFromIndex, state.visibleToIndex);
    }, [options, state.visibleFromIndex, state.visibleToIndex]);

    const effectiveFocus = React.useMemo(() => {
        if (state.focusedValue === undefined) return undefined;
        if (options.some(o => o.value === state.focusedValue)) return state.focusedValue;
        return options[0]?.value;
    }, [state.focusedValue, options]);

    const isInInput = React.useMemo(() => {
        return options.find(o => o.value === effectiveFocus)?.type === "input";
    }, [effectiveFocus, options]);

    React.useEffect(() => {
        if (effectiveFocus !== undefined) onFocus?.(effectiveFocus);
    }, [effectiveFocus, onFocus]);

    React.useEffect(() => {
        if (focusValue !== undefined) dispatch({ type: "set-focus", value: focusValue });
    }, [focusValue]);

    return {
        focusedValue: effectiveFocus,
        visibleFromIndex: state.visibleFromIndex,
        visibleToIndex: state.visibleToIndex,
        visibleOptions,
        isInInput: isInInput ?? false,
        focusNextOption,
        focusPreviousOption,
        focusNextPage,
        focusPreviousPage,
        focusOption,
        options
    };
}

/**
 * High-level selection hook with default value support.
 * Deobfuscated from uKB in chunk_206.ts.
 */
export function useSelect<T>({
    visibleOptionCount = 5,
    options,
    defaultValue,
    onChange,
    onCancel,
    onFocus,
    focusValue
}: {
    visibleOptionCount?: number;
    options: Array<{ label: string; value: T; description?: string; type?: string }>;
    defaultValue?: T;
    onChange?: (value: T) => void;
    onCancel?: () => void;
    onFocus?: (value: T) => void;
    focusValue?: T;
}) {
    const [value, setValue] = React.useState(defaultValue);
    const base = useSelectBase({
        visibleOptionCount,
        options,
        initialFocusValue: undefined,
        onFocus,
        focusValue
    });

    const selectFocusedOption = React.useCallback(() => {
        setValue(base.focusedValue);
    }, [base.focusedValue]);

    return {
        ...base,
        value,
        selectFocusedOption,
        onChange,
        onCancel
    };
}
