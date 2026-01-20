
// Logic from chunk_563.ts (Wizard Framework & Agent Creation)

import React, { useState, useEffect, useContext, createContext, useCallback, useMemo } from "react";
import { Box, Text } from "ink";
import { PermissionSelect } from "../permissions/PermissionComponents.js";
import { Shortcut } from "../shared/Shortcut.js";

// --- Wizard Context (EO0) ---
type WizardContextType = {
    currentStepIndex: number;
    totalSteps: number;
    wizardData: any;
    setWizardData: (data: any) => void;
    updateWizardData: (data: any) => void;
    goNext: () => void;
    goBack: () => void;
    goToStep: (index: number) => void;
    cancel: () => void;
    title?: string;
    showStepCounter?: boolean;
};

const WizardContext = createContext<WizardContextType | null>(null);

export function useWizard() {
    const context = useContext(WizardContext);
    if (!context) throw new Error("useWizard must be used within a WizardProvider");
    return context;
}

// --- Wizard Component (zO0) ---
export function Wizard({
    steps,
    initialData = {},
    onComplete,
    onCancel,
    children,
    title,
    showStepCounter = true
}: any) {
    const [currentStepIndex, setCurrentStepIndex] = useState(0);
    const [wizardData, setWizardData] = useState(initialData);
    const [isComplete, setIsComplete] = useState(false);

    useEffect(() => {
        if (isComplete) onComplete(wizardData);
    }, [isComplete, wizardData, onComplete]);

    const goNext = useCallback(() => {
        if (currentStepIndex < steps.length - 1) {
            setCurrentStepIndex(i => i + 1);
        } else {
            setIsComplete(true);
        }
    }, [currentStepIndex, steps.length]);

    const goBack = useCallback(() => {
        if (currentStepIndex > 0) {
            setCurrentStepIndex(i => i - 1);
        } else if (onCancel) {
            onCancel();
        }
    }, [currentStepIndex, onCancel]);

    const goToStep = useCallback((index: number) => {
        if (index >= 0 && index < steps.length) {
            setCurrentStepIndex(index);
        }
    }, [steps.length]);

    const updateWizardData = useCallback((data: any) => {
        setWizardData((prev: any) => ({ ...prev, ...data }));
    }, []);

    const value = useMemo(() => ({
        currentStepIndex,
        totalSteps: steps.length,
        wizardData,
        setWizardData,
        updateWizardData,
        goNext,
        goBack,
        goToStep,
        cancel: onCancel,
        title,
        showStepCounter
    }), [currentStepIndex, steps.length, wizardData, updateWizardData, goNext, goBack, goToStep, onCancel, title, showStepCounter]);

    const StepComponent = steps[currentStepIndex];
    if (!StepComponent || isComplete) return null;

    return (
        <WizardContext.Provider value={value}>
            {children || <StepComponent />}
        </WizardContext.Provider>
    );
}

// --- Wizard Layout Container (bI) ---
export function WizardLayout({
    title,
    titleColor = "text",
    borderColor = "suggestion",
    children,
    subtitle,
    footerText
}: any) {
    const { currentStepIndex, totalSteps, title: wizardTitle, showStepCounter } = useWizard();

    return (
        <Box flexDirection="column">
            <Box borderStyle="round" borderColor={borderColor} flexDirection="column" paddingX={1}>
                <Box flexDirection="column" marginBottom={1}>
                    <Text bold color={titleColor}>
                        {title || wizardTitle || "Wizard"}
                        {showStepCounter !== false && ` (${currentStepIndex + 1}/${totalSteps})`}
                    </Text>
                    {subtitle && <Text dimColor>{subtitle}</Text>}
                </Box>
                <Box flexDirection="column">
                    {children}
                </Box>
            </Box>
            <Box marginLeft={3}>
                <Text dimColor>{footerText || "Enter to select · Esc to back"}</Text>
            </Box>
        </Box>
    );
}

// --- Agent Creation Steps ---

export function AgentLocationStep() {
    const { goNext, updateWizardData, cancel } = useWizard();
    return (
        <WizardLayout subtitle="Choose location">
            <Box marginTop={1}>
                <PermissionSelect
                    options={[
                        { label: "Project (.claude/agents/)", value: "projectSettings" },
                        { label: "Personal (~/.claude/agents/)", value: "userSettings" }
                    ]}
                    onChange={(val) => {
                        updateWizardData({ location: val });
                        goNext();
                    }}
                    onCancel={cancel}
                />
            </Box>
        </WizardLayout>
    );
}

export function AgentMethodStep() {
    const { goNext, updateWizardData, goBack, goToStep } = useWizard();
    return (
        <WizardLayout subtitle="Creation method">
            <Box marginTop={1}>
                <PermissionSelect
                    options={[
                        { label: "Generate with Claude (recommended)", value: "generate" },
                        { label: "Manual configuration", value: "manual" }
                    ]}
                    onChange={(val) => {
                        const isGenerate = val === "generate";
                        updateWizardData({ method: val, wasGenerated: isGenerate });
                        if (isGenerate) goNext();
                        else goToStep(3); // Skip generation steps
                    }}
                    onCancel={goBack}
                />
            </Box>
        </WizardLayout>
    );
}
