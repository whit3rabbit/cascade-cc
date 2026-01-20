// Logic from chunk_489.ts (AskUserQuestion Tool, Wizard)

import React, { useCallback, useEffect, useMemo, useReducer } from "react";
import { Box, Text, useInput, useStdout } from "ink";
import InkTextInput from "ink-text-input";
import { z } from "zod";
import stringWidth from "string-width";
import { figures } from "../../vendor/terminalFigures.js";
import { PermissionRuleSummary, PermissionSelect } from "./PermissionComponents.js";

const TextInput = InkTextInput as unknown as React.FC<{
    value: string;
    onChange: (value: string) => void;
    onSubmit?: (value: string) => void;
    placeholder?: string;
    showCursor?: boolean;
}>;

const HEADER_MAX_CHARS = 12;

const ASK_USER_QUESTION_DESCRIPTION = "Asks the user multiple choice questions to gather information, clarify ambiguity, understand preferences, make decisions or offer them choices.";
const ASK_USER_QUESTION_PROMPT = `Use this tool when you need to ask the user questions during execution. This allows you to:\n1. Gather user preferences or requirements\n2. Clarify ambiguous instructions\n3. Get decisions on implementation choices as you work\n4. Offer choices to the user about what direction to take.\n\nUsage notes:\n- Users will always be able to select \"Other\" to provide custom text input\n- Use multiSelect: true to allow multiple answers to be selected for a question\n- If you recommend a specific option, make that the first option in the list and add \"(Recommended)\" at the end of the label\n`;

const OptionSchema = z.object({
    label: z.string().describe("The display text for this option that the user will see and select. Should be concise (1-5 words) and clearly describe the choice."),
    description: z.string().describe("Explanation of what this option means or what will happen if chosen. Useful for providing context about trade-offs or implications.")
});

const QuestionSchema = z.object({
    question: z.string().describe("The complete question to ask the user. Should be clear, specific, and end with a question mark. Example: \"Which library should we use for date formatting?\" If multiSelect is true, phrase it accordingly, e.g. \"Which features do you want to enable?\""),
    header: z.string().describe(`Very short label displayed as a chip/tag (max ${HEADER_MAX_CHARS} chars). Examples: "Auth method", "Library", "Approach".`),
    options: z.array(OptionSchema).min(2).max(4).describe("The available choices for this question. Must have 2-4 options. There should be no 'Other' option, that will be provided automatically."),
    multiSelect: z.boolean().describe("Set to true to allow the user to select multiple options instead of just one. Use when choices are not mutually exclusive.")
});

const AskUserQuestionInputSchema = z.strictObject({
    questions: z.array(QuestionSchema).min(1).max(4).describe("Questions to ask the user (1-4 questions)"),
    answers: z.record(z.string(), z.string()).optional().describe("User answers collected by the permission component")
}).refine((input) => {
    const questions = input.questions.map((question) => question.question);
    if (questions.length !== new Set(questions).size) return false;
    for (const question of input.questions) {
        const labels = question.options.map((option) => option.label);
        if (labels.length !== new Set(labels).size) return false;
    }
    return true;
}, {
    message: "Question texts must be unique, option labels must be unique within each question"
});

export const AskUserQuestionTool = {
    name: "AskUserQuestion",
    async description() {
        return ASK_USER_QUESTION_DESCRIPTION;
    },
    async prompt() {
        return ASK_USER_QUESTION_PROMPT;
    },
    inputSchema: AskUserQuestionInputSchema,
    userFacingName() {
        return "";
    },
    isEnabled() {
        return true;
    },
    isConcurrencySafe() {
        return true;
    },
    isReadOnly() {
        return true;
    },
    requiresUserInteraction() {
        return true;
    },
    async checkPermissions(input: any) {
        return {
            behavior: "ask",
            message: "Answer questions?",
            updatedInput: input
        };
    },
    renderToolUseMessage() {
        return null;
    },
    renderToolUseProgressMessage() {
        return null;
    },
    renderToolResultMessage({ answers }: any) {
        return <AskUserQuestionResult answers={answers} />;
    },
    renderToolUseRejectedMessage() {
        return (
            <Box flexDirection="row" marginTop={1}>
                <Text color="text">{figures.warning} </Text>
                <Text>User declined to answer questions</Text>
            </Box>
        );
    },
    renderToolUseErrorMessage() {
        return null;
    },
    async call({ questions, answers = {} }: any) {
        return {
            data: {
                questions,
                answers
            }
        };
    },
    mapToolResultToToolResultBlockParam({ answers }: any, toolUseId: string) {
        return {
            type: "tool_result",
            content: `User has answered your questions: ${Object.entries(answers)
                .map(([question, answer]) => `"${question}"="${answer}"`)
                .join(", ")}. You can now continue with the user's answers in mind.`,
            tool_use_id: toolUseId
        };
    }
};

function AskUserQuestionResult({ answers }: { answers: Record<string, string> }) {
    return (
        <Box flexDirection="column" marginTop={1}>
            <Box flexDirection="row">
                <Text color="text">{figures.info} </Text>
                <Text>User answered Claude's questions:</Text>
            </Box>
            <Box flexDirection="column">
                {Object.entries(answers).map(([question, answer]) => (
                    <Text key={question} color="inactive">· {question} → {answer}</Text>
                ))}
            </Box>
        </Box>
    );
}

type QuestionState = {
    selectedValue?: string | string[];
    textInputValue?: string;
};

type WizardState = {
    currentQuestionIndex: number;
    answers: Record<string, string>;
    questionStates: Record<string, QuestionState>;
    isInTextInput: boolean;
};

type WizardAction =
    | { type: "next-question" }
    | { type: "prev-question" }
    | { type: "update-question-state"; questionText: string; updates: Partial<QuestionState>; isMultiSelect: boolean }
    | { type: "set-answer"; questionText: string; answer: string; shouldAdvance: boolean }
    | { type: "set-text-input-mode"; isInInput: boolean };

const initialWizardState: WizardState = {
    currentQuestionIndex: 0,
    answers: {},
    questionStates: {},
    isInTextInput: false
};

function questionReducer(state: WizardState, action: WizardAction): WizardState {
    switch (action.type) {
        case "next-question":
            return { ...state, currentQuestionIndex: state.currentQuestionIndex + 1, isInTextInput: false };
        case "prev-question":
            return { ...state, currentQuestionIndex: Math.max(0, state.currentQuestionIndex - 1), isInTextInput: false };
        case "update-question-state": {
            const prevState = state.questionStates[action.questionText];
            const nextState: QuestionState = {
                selectedValue: action.updates.selectedValue ?? prevState?.selectedValue ?? (action.isMultiSelect ? [] : undefined),
                textInputValue: action.updates.textInputValue ?? prevState?.textInputValue ?? ""
            };
            return {
                ...state,
                questionStates: {
                    ...state.questionStates,
                    [action.questionText]: nextState
                }
            };
        }
        case "set-answer": {
            const nextState = {
                ...state,
                answers: {
                    ...state.answers,
                    [action.questionText]: action.answer
                }
            };
            if (action.shouldAdvance) {
                return {
                    ...nextState,
                    currentQuestionIndex: nextState.currentQuestionIndex + 1,
                    isInTextInput: false
                };
            }
            return nextState;
        }
        case "set-text-input-mode":
            return { ...state, isInTextInput: action.isInInput };
        default:
            return state;
    }
}

function useQuestionWizard() {
    const [state, dispatch] = useReducer(questionReducer, initialWizardState);

    const nextQuestion = useCallback(() => dispatch({ type: "next-question" }), []);
    const prevQuestion = useCallback(() => dispatch({ type: "prev-question" }), []);
    const updateQuestionState = useCallback((questionText: string, updates: Partial<QuestionState>, isMultiSelect: boolean) => {
        dispatch({ type: "update-question-state", questionText, updates, isMultiSelect });
    }, []);
    const setAnswer = useCallback((questionText: string, answer: string, shouldAdvance = true) => {
        dispatch({ type: "set-answer", questionText, answer, shouldAdvance });
    }, []);
    const setTextInputMode = useCallback((isInInput: boolean) => {
        dispatch({ type: "set-text-input-mode", isInInput });
    }, []);

    return {
        ...state,
        nextQuestion,
        prevQuestion,
        updateQuestionState,
        setAnswer,
        setTextInputMode
    };
}

function Divider() {
    const columns = process.stdout.columns || 80;
    const line = "-".repeat(Math.max(10, Math.min(columns, 60)));
    return <Text dimColor>{line}</Text>;
}

function WizardTabs({
    questions,
    currentQuestionIndex,
    answers,
    hideSubmitTab = false
}: {
    questions: any[];
    currentQuestionIndex: number;
    answers: Record<string, string>;
    hideSubmitTab?: boolean;
}) {
    const columns = process.stdout.columns || 80;

    const labels = useMemo(() => {
        const submitLabel = hideSubmitTab ? "" : ` ${figures.tick} Submit `;
        const minLeftPadding = 2;
        const minRightPadding = 2;
        const headerWidth = stringWidth(`${figures.arrowLeft} `) + stringWidth(` ${figures.arrowRight}`) + stringWidth(submitLabel);
        const available = columns - headerWidth;
        const headers = questions.map((question, index) => question?.header || `Q${index + 1}`);

        if (available <= 0) {
            return headers.map((label, index) => (index == currentQuestionIndex ? label.slice(0, 3) : ""));
        }

        const totalWidth = headers.map((label) => minLeftPadding + minRightPadding + stringWidth(label)).reduce((sum, value) => sum + value, 0);
        if (totalWidth <= available) return headers;

        const currentLabel = headers[currentQuestionIndex] || "";
        const currentWidth = minLeftPadding + minRightPadding + stringWidth(currentLabel);
        const minOtherWidth = 6;
        const maxCurrent = Math.min(currentWidth, available / 2);
        const remaining = available - maxCurrent;
        const otherCount = Math.max(headers.length - 1, 1);
        const perOther = Math.max(minOtherWidth, Math.floor(remaining / otherCount));

        const truncate = (label: string, width: number, fallback?: string) => {
            if (stringWidth(label) <= width) return label;
            let trimmed = label;
            while (trimmed.length > 1 && stringWidth(`${trimmed}…`) > width) {
                trimmed = trimmed.slice(0, -1);
            }
            if (trimmed.length === 0 && fallback) return fallback;
            return trimmed.length > 0 ? `${trimmed}…` : fallback ?? label;
        };

        return headers.map((label, index) => {
            if (index === currentQuestionIndex) {
                const width = maxCurrent - minLeftPadding - minRightPadding;
                return truncate(label, width);
            }
            const width = perOther - minLeftPadding - minRightPadding;
            return truncate(label, width, label.length > 0 ? `${label[0]}…` : label);
        });
    }, [columns, currentQuestionIndex, hideSubmitTab, questions]);

    const singleQuestion = questions.length === 1 && hideSubmitTab;

    return (
        <Box flexDirection="row" marginBottom={1}>
            {!singleQuestion && (
                <Text color={currentQuestionIndex === 0 ? "inactive" : undefined}>{figures.arrowLeft} </Text>
            )}
            {questions.map((question, index) => {
                const isCurrent = index === currentQuestionIndex;
                const answered = question?.question && answers[question.question];
                const icon = answered ? figures.checkboxOn : figures.checkboxOff;
                const label = labels[index] || question?.header || `Q${index + 1}`;
                return (
                    <Box key={question?.question || `question-${index}`}>
                        {isCurrent ? (
                            <Text backgroundColor="permission" color="inverseText"> {icon} {label} </Text>
                        ) : (
                            <Text> {icon} {label} </Text>
                        )}
                    </Box>
                );
            })}
            {!hideSubmitTab && (
                <Box key="submit">
                    {currentQuestionIndex === questions.length ? (
                        <Text backgroundColor="permission" color="inverseText"> {figures.tick} Submit </Text>
                    ) : (
                        <Text> {figures.tick} Submit </Text>
                    )}
                </Box>
            )}
            {!singleQuestion && (
                <Text color={currentQuestionIndex === questions.length ? "inactive" : undefined}> {figures.arrowRight}</Text>
            )}
        </Box>
    );
}

function SectionTitle({ title }: { title: string }) {
    return (
        <Box marginBottom={1}>
            <Text>{title}</Text>
        </Box>
    );
}

function MultiSelect({
    options,
    defaultValue,
    onChange,
    onFocus,
    onCancel,
    submitButtonText,
    onSubmit
}: {
    options: any[];
    defaultValue?: string[];
    onChange: (values: string[]) => void;
    onFocus?: (value: string) => void;
    onCancel?: () => void;
    submitButtonText?: string;
    onSubmit?: () => void;
}) {
    const [focusedIndex, setFocusedIndex] = React.useState(0);
    const [selectedValues, setSelectedValues] = React.useState<string[]>(defaultValue ?? []);
    const [inputValues, setInputValues] = React.useState<Record<string, string>>({});

    const submitIndex = submitButtonText ? options.length : -1;

    useInput((input, key) => {
        const focusedOption = options[focusedIndex];
        const isInputOption = focusedOption?.type === "input";

        if (isInputOption && !key.upArrow && !key.downArrow && !key.escape && !key.return && input !== " " && input !== "k" && input !== "j") {
            return;
        }
        if (key.upArrow || input === "k") {
            setFocusedIndex((prev) => Math.max(0, prev - 1));
        }
        if (key.downArrow || input === "j") {
            setFocusedIndex((prev) => Math.min(submitIndex >= 0 ? options.length : options.length - 1, prev + 1));
        }
        if (key.escape) onCancel?.();

        if (key.return || input === " ") {
            if (submitIndex === focusedIndex) {
                onSubmit?.();
                return;
            }
            const option = options[focusedIndex];
            if (!option) return;
            const alreadySelected = selectedValues.includes(option.value);
            const next = alreadySelected
                ? selectedValues.filter((value) => value !== option.value)
                : [...selectedValues, option.value];
            setSelectedValues(next);
            onChange(next);
        }
    });

    useEffect(() => {
        const focused = options[focusedIndex];
        if (focused && onFocus) onFocus(focused.value);
    }, [focusedIndex, onFocus, options]);

    return (
        <Box flexDirection="column" gap={1}>
            {options.map((option, index) => {
                const isFocused = index === focusedIndex;
                const isSelected = selectedValues.includes(option.value);
                return (
                    <Box key={option.value} flexDirection="column">
                        <Text color={isFocused ? "permission" : undefined}>
                            {isFocused ? figures.pointer : " "} {isSelected ? figures.checkboxOn : figures.checkboxOff} {option.label}
                        </Text>
                        {option.description && isFocused && (
                            <Text dimColor>{option.description}</Text>
                        )}
                        {isFocused && option.type === "input" && (
                            <Box paddingLeft={2}>
                                <TextInput
                                    value={inputValues[option.value] ?? option.initialValue ?? ""}
                                    onChange={(value) => {
                                        setInputValues((prev) => ({ ...prev, [option.value]: value }));
                                        option.onChange?.(value);
                                    }}
                                    onSubmit={(value) => {
                                        option.onChange?.(value);
                                    }}
                                    placeholder={option.placeholder}
                                    showCursor={true}
                                />
                            </Box>
                        )}
                    </Box>
                );
            })}
            {submitButtonText && (
                <Text color={focusedIndex === submitIndex ? "permission" : undefined}>
                    {focusedIndex === submitIndex ? figures.pointer : " "} {submitButtonText}
                </Text>
            )}
        </Box>
    );
}

function QuestionScreen({
    question,
    questions,
    currentQuestionIndex,
    answers,
    questionStates,
    hideSubmitTab,
    onUpdateQuestionState,
    onAnswer,
    onTextInputFocus,
    onCancel,
    onSubmit
}: any) {
    const setFocus = useCallback((value: string) => {
        onTextInputFocus(value === "__other__");
    }, [onTextInputFocus]);

    const baseOptions = question.options.map((option: any) => ({
        type: "text",
        value: option.label,
        label: option.label,
        description: option.description
    }));

    const textState = questionStates[question.question];

    const otherOption = {
        type: "input",
        value: "__other__",
        label: "Other",
        placeholder: question.multiSelect ? "Type something" : "Type something.",
        initialValue: textState?.textInputValue ?? "",
        onChange: (value: string) => onUpdateQuestionState(question.question, { textInputValue: value }, question.multiSelect ?? false)
    };

    const options = [...baseOptions, otherOption];

    return (
        <Box flexDirection="column" marginTop={1}>
            <Divider />
            <Box flexDirection="column" paddingTop={0}>
                <WizardTabs
                    questions={questions}
                    currentQuestionIndex={currentQuestionIndex}
                    answers={answers}
                    hideSubmitTab={hideSubmitTab}
                />
                <SectionTitle title={question.question} />
                <Box marginTop={1}>
                    {question.multiSelect ? (
                        <MultiSelect
                            options={options}
                            defaultValue={Array.isArray(questionStates[question.question]?.selectedValue)
                                ? questionStates[question.question]?.selectedValue
                                : undefined}
                            onChange={(values: string[]) => {
                                onUpdateQuestionState(question.question, { selectedValue: values }, true);
                                const otherInput = values.includes("__other__") ? textState?.textInputValue : undefined;
                                const selectedValues = values.filter((value) => value !== "__other__").concat(otherInput ? [otherInput] : []);
                                onAnswer(question.question, selectedValues, undefined, false);
                            }}
                            onFocus={setFocus}
                            onCancel={onCancel}
                            submitButtonText={currentQuestionIndex === questions.length - 1 ? "Submit" : "Next"}
                            onSubmit={onSubmit}
                        />
                    ) : (
                        <PermissionSelect
                            options={options}
                            defaultValue={typeof questionStates[question.question]?.selectedValue === "string"
                                ? questionStates[question.question]?.selectedValue
                                : undefined}
                            onChange={(value: string, inputValue?: string) => {
                                onUpdateQuestionState(question.question, { selectedValue: value }, false);
                                const otherInput = value === "__other__" ? textState?.textInputValue : undefined;
                                onAnswer(question.question, value, otherInput);
                            }}
                            onFocus={setFocus}
                            onCancel={onCancel}
                            layout="compact-vertical"
                        />
                    )}
                </Box>
                <Box marginTop={1}>
                    <Text color="inactive" dimColor>
                        Enter to select · Tab/Arrow keys to navigate · Esc to cancel
                    </Text>
                </Box>
            </Box>
        </Box>
    );
}

function ReviewScreen({
    questions,
    currentQuestionIndex,
    answers,
    allQuestionsAnswered,
    permissionResult,
    onFinalResponse
}: any) {
    return (
        <Box flexDirection="column" marginTop={1}>
            <Divider />
            <Box flexDirection="column" borderTop borderColor="inactive" paddingTop={0}>
                <WizardTabs
                    questions={questions}
                    currentQuestionIndex={currentQuestionIndex}
                    answers={answers}
                />
                <SectionTitle title="Review your answers" />
                <Box flexDirection="column" marginTop={1}>
                    {!allQuestionsAnswered && (
                        <Box marginBottom={1}>
                            <Text color="warning">{figures.warning} You have not answered all questions</Text>
                        </Box>
                    )}
                    {Object.keys(answers).length > 0 && (
                        <Box flexDirection="column" marginBottom={1}>
                            {questions
                                .filter((question: any) => question?.question && answers[question.question])
                                .map((question: any) => (
                                    <Box key={question.question} flexDirection="column" marginLeft={1}>
                                        <Text>• {question.question}</Text>
                                        <Box marginLeft={2}>
                                            <Text color="success">{figures.arrowRight} {answers[question.question]}</Text>
                                        </Box>
                                    </Box>
                                ))}
                        </Box>
                    )}
                </Box>
                <PermissionRuleSummary permissionResult={permissionResult} toolType="tool" />
                <Text color="inactive">Ready to submit your answers?</Text>
                <Box marginTop={1}>
                    <PermissionSelect
                        options={[
                            { type: "text", label: "Submit answers", value: "submit" },
                            { type: "text", label: "Cancel", value: "cancel" }
                        ]}
                        onChange={(value) => onFinalResponse(value)}
                        onCancel={() => onFinalResponse("cancel")}
                    />
                </Box>
            </Box>
        </Box>
    );
}

export function AskUserQuestionWizard({ toolUseConfirm, onDone, onReject }: any) {
    const parsed = AskUserQuestionInputSchema.safeParse(toolUseConfirm.input);
    const questions = parsed.success ? parsed.data.questions || [] : [];

    const {
        currentQuestionIndex,
        answers,
        questionStates,
        isInTextInput,
        nextQuestion,
        prevQuestion,
        updateQuestionState,
        setAnswer,
        setTextInputMode
    } = useQuestionWizard();

    const activeQuestion = currentQuestionIndex < questions.length ? questions[currentQuestionIndex] : null;
    const isReview = currentQuestionIndex === questions.length;
    const allAnswered = questions.every((question: any) => question?.question && answers[question.question]);
    const hideSubmitTab = questions.length === 1 && !questions[0]?.multiSelect;

    const handleCancel = useCallback(() => {
        onDone();
        onReject();
        toolUseConfirm.onReject();
    }, [onDone, onReject, toolUseConfirm]);

    const handleSubmit = useCallback((finalAnswers: Record<string, string>) => {
        const payload = { ...toolUseConfirm.input, answers: finalAnswers };
        onDone();
        toolUseConfirm.onAllow(payload, []);
    }, [onDone, toolUseConfirm]);

    const handleAnswer = useCallback((questionText: string, selected: string | string[], otherInput?: string, shouldAdvance = true) => {
        const isMulti = Array.isArray(selected);
        const answerText = isMulti ? selected.join(", ") : (otherInput || selected);

        const onlyQuestion = questions.length === 1;
        if (!isMulti && onlyQuestion && shouldAdvance) {
            handleSubmit({ ...answers, [questionText]: answerText });
            return;
        }

        setAnswer(questionText, answerText, shouldAdvance);
    }, [answers, handleSubmit, questions.length, setAnswer]);

    useInput((input, key) => {
        if (isInTextInput && !isReview) return;
        if (key.return) return;
        if ((key.leftArrow || (key.shift && key.tab)) && currentQuestionIndex > 0) prevQuestion();
        const maxIndex = hideSubmitTab ? questions.length - 1 : questions.length;
        if ((key.rightArrow || (key.tab && !key.shift)) && currentQuestionIndex < maxIndex) nextQuestion();
    });

    if (activeQuestion) {
        return (
            <QuestionScreen
                question={activeQuestion}
                questions={questions}
                currentQuestionIndex={currentQuestionIndex}
                answers={answers}
                questionStates={questionStates}
                hideSubmitTab={hideSubmitTab}
                onUpdateQuestionState={updateQuestionState}
                onAnswer={handleAnswer}
                onTextInputFocus={setTextInputMode}
                onCancel={handleCancel}
                onSubmit={nextQuestion}
            />
        );
    }

    if (isReview) {
        return (
            <ReviewScreen
                questions={questions}
                currentQuestionIndex={currentQuestionIndex}
                answers={answers}
                allQuestionsAnswered={allAnswered}
                permissionResult={toolUseConfirm.permissionResult}
                onFinalResponse={(value: string) => {
                    if (value === "cancel") {
                        handleCancel();
                        return;
                    }
                    if (value === "submit") handleSubmit(answers);
                }}
            />
        );
    }

    return null;
}
