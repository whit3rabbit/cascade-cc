import { useInput } from "../../vendor/useInput.js";

interface SelectionInputProps {
    isDisabled?: boolean;
    disableSelection?: boolean | "numeric";
    state: any;
    options: any[];
    isMultiSelect?: boolean;
    onUpFromFirstItem?: () => void;
    onInputModeToggle?: (value: any) => void;
}

/**
 * Hook to handle list-specific keybindings (Arrows, PageUp/Down, Return).
 * Deobfuscated from dKB in chunk_206.ts.
 */
export function useSelectionInput({
    isDisabled = false,
    disableSelection = false,
    state,
    options,
    isMultiSelect = false,
    onUpFromFirstItem,
    onInputModeToggle
}: SelectionInputProps) {
    useInput((input, key) => {
        const focusedOption = options.find(o => o.value === state.focusedValue);
        const isInInput = focusedOption?.type === "input";

        if (key.tab && onInputModeToggle && state.focusedValue !== undefined) {
            onInputModeToggle(state.focusedValue);
            return;
        }

        if (isInInput) {
            // If we are in an input field, only intercept navigation keys
            if (!(key.upArrow || key.downArrow || key.escape || (key.ctrl && (input === "n" || input === "p")))) {
                return;
            }
        }

        // Navigation
        if (key.downArrow || (key.ctrl && input === "n") || (!key.ctrl && !key.shift && input === "j")) {
            state.focusNextOption();
        }

        if (key.upArrow || (key.ctrl && input === "p") || (!key.ctrl && !key.shift && input === "k")) {
            if (onUpFromFirstItem && state.visibleFromIndex === 0) {
                const first = options[0];
                if (first && state.focusedValue === first.value) {
                    onUpFromFirstItem();
                    return;
                }
            }
            state.focusPreviousOption();
        }

        if (key.pageDown) state.focusNextPage();
        if (key.pageUp) state.focusPreviousPage();

        // Selection
        if (disableSelection !== true) {
            const isInteraction = isMultiSelect ? (key.return || input === " ") : key.return;
            if (isInteraction && state.focusedValue !== undefined) {
                if (focusedOption?.disabled !== true) {
                    state.selectFocusedOption?.();
                    state.onChange?.(state.focusedValue);
                }
            }

            // Numeric shortcuts
            if (disableSelection !== "numeric" && /^[0-9]+$/.test(input)) {
                const idx = parseInt(input) - 1;
                if (idx >= 0 && idx < state.options.length) {
                    const opt = state.options[idx];
                    if (opt.disabled === true) return;
                    if (opt.type === "input") {
                        state.focusOption(opt.value);
                        return;
                    }
                    state.onChange?.(opt.value);
                    return;
                }
            }
        }

        if (key.escape) state.onCancel?.();
    }, {
        isActive: !isDisabled
    });
}
