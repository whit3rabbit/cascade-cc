import * as React from "react";
import { Theme, themes } from "../../utils/shared/theme.js";

export interface ThemeContextValue {
    theme: Theme | null;
    setTheme: (theme: Theme) => void;
    setPreviewTheme: (theme: Theme | null) => void;
    savePreview: () => void;
    cancelPreview: () => void;
    currentTheme: Theme | null;
}

export const ThemeContext = React.createContext<ThemeContextValue>({
    theme: null,
    setTheme: () => { },
    setPreviewTheme: () => { },
    savePreview: () => { },
    cancelPreview: () => { },
    currentTheme: null
});

/**
 * Provider for theme management, allowing global and preview theme states.
 * Deobfuscated from Nl1 in chunk_202.ts.
 */
export const ThemeProvider: React.FC<{
    children: React.ReactNode;
    initialState: Theme;
}> = ({ children, initialState }) => {
    const [theme, setThemeState] = React.useState<Theme>(initialState);
    const [previewTheme, setPreviewTheme] = React.useState<Theme | null>(null);

    const value = React.useMemo(() => ({
        theme,
        setTheme: (newTheme: Theme) => {
            // trackFeatureUsage("theme"); // Logic from v9
            setThemeState(newTheme);
            setPreviewTheme(null);
        },
        setPreviewTheme: (preview: Theme | null) => {
            setPreviewTheme(preview);
        },
        savePreview: () => {
            if (previewTheme) {
                setThemeState(previewTheme);
                setPreviewTheme(null);
            }
        },
        cancelPreview: () => {
            setPreviewTheme(null);
        },
        currentTheme: previewTheme ?? theme
    }), [theme, previewTheme]);

    return (
        <ThemeContext.Provider value={value}>
            {children}
        </ThemeContext.Provider>
    );
};

export const useTheme = (): [Theme | null, (theme: Theme) => void] => {
    const { currentTheme, setTheme } = React.useContext(ThemeContext);
    return [currentTheme, setTheme];
};


export const useThemePreview = () => {
    const { setPreviewTheme, savePreview, cancelPreview } = React.useContext(ThemeContext);
    return { setPreviewTheme, savePreview, cancelPreview };
};

export const getThemeStyle = (theme: any, key: string) => {
    // Stub implementation to safely access theme properties
    return theme ? theme[key] : undefined;
};
