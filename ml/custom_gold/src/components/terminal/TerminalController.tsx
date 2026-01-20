
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Box, Text } from "ink";

// Local components 
import { TerminalInput } from "./TerminalInput.js";
import { Shortcut } from "../shared/Shortcut.js";
import {
  VimTerminalInput,
  PromptModeIndicator,
  TerminalStatusLine,
  PreInputOverlays,
  TopPromptAddon,
  PromptStashIndicator,
  PastedContentIndicator,
  ModelPicker,
  ThinkingModePicker,
  BashDialog,
  Separator
} from "./TerminalUIComponents.js";

// Hooks & Contexts
import { useAppState } from "../../contexts/AppStateContext.js";
import { useNotifications } from "../../services/terminal/NotificationService.js";
import { useAutocomplete as useCommandSuggestions } from "../../hooks/useAutocomplete.js";
import {
  getShellSnapshot,
  useHistorySearch,
  getNextAttachmentId,
  usePromptSuggestion,
  useInputUndoBuffer,
  usePasteListener,
  usePromptSubmissionStatus,
  normalizeHistorySearchEntry,
  findThinkingKeywords,
  findWarningKeywords,
  getThinkingKeywordLevel,
  isRainbowKeyword,
  thinkingKeywordColors,
  thinkingKeywordShimmerColors,
  getSessionHints,
  updateSessionHints,
  STASH_HINT_TIMEOUT_MS,
  resetInputHints,
  detectInputModeFromPrefix,
  useHistoryNavigation,
  isQueuedCommandMode,
  trackEvent,
  trackSuggestionTiming,
  registerAttachment,
  logAttachmentPaste,
  normalizePastedText,
  countLines,
  MAX_PASTE_SIZE,
  formatPasteAsAttachment,
  createMessageSelectorHandler,
  popQueuedInput,
  useMcpAtMentionListener,
  pathUtils,
  getWorkspaceRoot,
  useInputKeypress,
  getPlatformId,
  OPTION_META_HINTS,
  trackShortcut,
  openExternalEditor,
  modeCycleHotkey,
  getNextPromptMode,
  setPlanModeExit,
  shouldPromptModeSwitch,
  setAutoAcceptMode,
  setEnableEditsToast,
  applyModeChange,
  modelPickerHotkey,
  thinkingPickerHotkey,
  useTerminalSize,
  dimText,
  INLINE_SUGGESTION_HINT,
  isVimInputEnabled,
  getBorderColorOverride,
  KNOWN_BORDER_COLORS,
  borderColorLookup,
  getActiveBanner
} from "../../hooks/terminal/useTerminalControllerHooks.js";

export type TerminalInteractionControllerProps = {
  debug: boolean;
  ideSelection: any;
  toolPermissionContext: any;
  setToolPermissionContext: (next: any) => void;
  apiKeyStatus: any;
  commands: any[];
  agents: any[];
  isLoading: boolean;
  verbose: boolean;
  messages: any[];
  onAutoUpdaterResult: (result: any) => void;
  autoUpdaterResult: any;
  input: string;
  onInputChange: (value: string) => void;
  mode: string;
  onModeChange: (mode: string) => void;
  stashedPrompt: { text: string; cursorOffset: number } | undefined;
  setStashedPrompt: (value: { text: string; cursorOffset: number } | undefined) => void;
  submitCount: number;
  onShowMessageSelector: () => void;
  mcpClients: any;
  pastedContents: Record<string, any>;
  setPastedContents: React.Dispatch<React.SetStateAction<Record<string, any>>>;
  vimMode: any;
  setVimMode: (mode: any) => void;
  showBashesDialog: boolean;
  setShowBashesDialog: (value: boolean) => void;
  showDiffDialog: boolean;
  setShowDiffDialog: (value: boolean) => void;
  tasksSelected: boolean;
  setTasksSelected: (value: boolean) => void;
  diffSelected: boolean;
  setDiffSelected: (value: boolean) => void;
  onExit: () => void;
  getToolUseContext: (...args: any[]) => any;
  onSubmit: (value: string, ctx: any) => Promise<void>;
  isSearchingHistory: boolean;
  setIsSearchingHistory: (value: boolean) => void;
};

// --- Interaction Controller (o17) ---
export function TerminalInteractionController({
  debug,
  ideSelection,
  toolPermissionContext,
  setToolPermissionContext,
  apiKeyStatus,
  commands,
  agents,
  isLoading,
  verbose,
  messages,
  onAutoUpdaterResult,
  autoUpdaterResult,
  input,
  onInputChange,
  mode,
  onModeChange,
  stashedPrompt,
  setStashedPrompt,
  submitCount,
  onShowMessageSelector,
  mcpClients,
  pastedContents,
  setPastedContents,
  vimMode,
  setVimMode,
  showBashesDialog,
  setShowBashesDialog,
  showDiffDialog,
  setShowDiffDialog,
  tasksSelected,
  setTasksSelected,
  diffSelected,
  setDiffSelected,
  onExit,
  getToolUseContext,
  onSubmit,
  isSearchingHistory,
  setIsSearchingHistory
}: TerminalInteractionControllerProps) {
  const shellSnapshot = getShellSnapshot();
  const [isAutoUpdating, setIsAutoUpdating] = useState(false);
  const [exitMessage, setExitMessage] = useState<{ show: boolean; key?: string }>({ show: false });
  const [cursorOffset, setCursorOffset] = useState(input.length);
  const [appState, setAppState] = useAppState();
  const {
    historyQuery,
    setHistoryQuery,
    historyMatch,
    historyFailedMatch
  } = useHistorySearch(
    (entry: any) => {
      const display = typeof entry === "string" ? entry : entry.display;
      submitInput(display);
    },
    input,
    onInputChange,
    setCursorOffset,
    cursorOffset,
    onModeChange,
    mode,
    isSearchingHistory,
    setIsSearchingHistory
  );
  const nextAttachmentIdRef = useRef(getNextAttachmentId(messages));
  const [helpOpen, setHelpOpen] = useState(false);
  const [teamsSelected, setTeamsSelected] = useState(false);
  const [diffDialogOpen, setDiffDialogOpen] = useState(false);
  const [isPasting, setIsPasting] = useState(false);
  const [isExternalEditorOpen, setIsExternalEditorOpen] = useState(false);
  const [showModelPicker, setShowModelPicker] = useState(false);
  const [showThinkingPicker, setShowThinkingPicker] = useState(false);
  const [pastedContentSelected, setPastedContentSelected] = useState(false);
  const [selectedPastedIndex, setSelectedPastedIndex] = useState(0);

  const isCursorBeforeFirstNewline = useMemo(() => {
    const firstNewline = input.indexOf("\n");
    if (firstNewline === -1) return true;
    return cursorOffset <= firstNewline;
  }, [input, cursorOffset]);

  const imagePastes = useMemo(
    () => Object.values(pastedContents).filter((item: any) => item.type === "image"),
    [pastedContents]
  );

  const teamList = useMemo(() => {
    return [] as any[];
  }, [appState.teamContext]);

  const {
    suggestion: inlineSuggestion,
    markAccepted: markSuggestionAccepted,
    logOutcomeAtSubmission: logSuggestionOutcome,
    markShown: markSuggestionShown
  } = usePromptSuggestion({
    inputValue: input,
    isAssistantResponding: isLoading
  });

  const inputForSearch = useMemo(
    () => (isSearchingHistory && historyMatch ? normalizeHistorySearchEntry(typeof historyMatch === "string" ? historyMatch : historyMatch.display) : input),
    [isSearchingHistory, historyMatch, input]
  );

  const thinkingKeywordMatches = useMemo(() => findThinkingKeywords(inputForSearch), [inputForSearch]);
  const warningKeywordMatches = useMemo(() => findWarningKeywords(inputForSearch), [inputForSearch]);

  const inputHighlights = useMemo(() => {
    const highlights: any[] = [];

    if (isSearchingHistory && historyMatch && !historyFailedMatch) {
      highlights.push({
        start: cursorOffset,
        end: cursorOffset + historyQuery.length,
        style: { type: "solid", color: "warning" },
        priority: 20
      });
    }

    if (thinkingKeywordMatches.length > 0) {
      const thinkingLevel = getThinkingKeywordLevel(inputForSearch);
      if (thinkingLevel.level !== "none") {
        const baseColor = thinkingKeywordColors[thinkingLevel.level];
        const shimmerColor = thinkingKeywordShimmerColors[thinkingLevel.level];
        for (const match of thinkingKeywordMatches) {
          highlights.push({
            start: match.start,
            end: match.end,
            style: isRainbowKeyword(match.word)
              ? { type: "rainbow", useShimmer: true }
              : { type: "shimmer", baseColor, shimmerColor },
            priority: 10
          });
        }
      }
    }

    for (const match of warningKeywordMatches) {
      highlights.push({
        start: match.start,
        end: match.end,
        style: { type: "solid", color: "warning" },
        priority: 15
      });
    }

    return highlights;
  }, [
    isSearchingHistory,
    historyQuery,
    historyMatch,
    historyFailedMatch,
    cursorOffset,
    thinkingKeywordMatches,
    warningKeywordMatches,
    inputForSearch
  ]);

  const { addNotification, removeNotification } = useNotifications();

  useEffect(() => {
    if (!thinkingKeywordMatches.length) return;
    if (thinkingKeywordMatches.length && !appState.thinkingEnabled) {
      addNotification({
        key: "thinking-toggled-via-keyword",
        jsx: <Text color="suggestion">Thinking on</Text>,
        priority: "immediate",
        timeoutMs: 3000
      });
    }
  }, [addNotification, appState.thinkingEnabled, setAppState, thinkingKeywordMatches.length]);

  const previousInputLength = useRef(input.length);
  const lastHintInputLength = useRef(input.length);
  const dismissStashHint = useCallback(() => {
    removeNotification("stash-hint");
  }, [removeNotification]);

  useEffect(() => {
    const previousLength = previousInputLength.current;
    const lastHintLength = lastHintInputLength.current;
    const currentLength = input.length;

    previousInputLength.current = currentLength;

    if (currentLength > lastHintLength) {
      lastHintInputLength.current = currentLength;
      return;
    }

    if (currentLength === 0) {
      lastHintInputLength.current = 0;
      return;
    }

    const droppedBelowThreshold = lastHintLength >= 20 && currentLength <= 5;
    const wasPreviouslyAboveThreshold = previousLength >= 20 && currentLength <= 5;

    if (droppedBelowThreshold && !wasPreviouslyAboveThreshold) {
      if (!getSessionHints().hasUsedStash) {
        addNotification({
          key: "stash-hint",
          jsx: (
            <Text dimColor>
              Tip: <Shortcut shortcut="ctrl+s" action="stash" />
            </Text>
          ),
          priority: "immediate",
          timeoutMs: STASH_HINT_TIMEOUT_MS
        });
      }
      lastHintInputLength.current = currentLength;
    }
  }, [input.length, addNotification]);

  const {
    pushToBuffer: pushUndoBuffer,
    undo: undoInput,
    canUndo,
    clearBuffer: clearUndoBuffer
  } = useInputUndoBuffer({
    maxBufferSize: 50,
    debounceMs: 1000
  });

  usePasteListener({
    input,
    pastedContents,
    onInputChange,
    setCursorOffset,
    setPastedContents
  });

  const promptStatusText = usePromptSubmissionStatus({
    input,
    submitCount
  });

  const handleInputChange = useCallback(
    (nextInput: string) => {
      if (nextInput === "?") {
        trackEvent("tengu_help_toggled", {});
        setHelpOpen((prev) => !prev);
        return;
      }

      setHelpOpen(false);
      dismissStashHint();
      resetInputHints();

      const wasSingleCharAdd = nextInput.length === input.length + 1;
      const isAtStart = cursorOffset === 0;
      const detectedMode = detectInputModeFromPrefix(nextInput);

      if (wasSingleCharAdd && isAtStart && detectedMode !== "prompt") {
        onModeChange(detectedMode);
        return;
      }

      const normalizedInput = nextInput.replaceAll("\t", "    ");
      if (input !== normalizedInput) {
        pushUndoBuffer(input, cursorOffset, pastedContents);
      }

      setTasksSelected(false);
      setDiffSelected(false);
      setTeamsSelected(false);
      onInputChange(normalizedInput);
    },
    [
      onInputChange,
      onModeChange,
      input,
      cursorOffset,
      pushUndoBuffer,
      pastedContents,
      setTasksSelected,
      setDiffSelected,
      setTeamsSelected,
      dismissStashHint
    ]
  );

  const {
    resetHistory,
    onHistoryUp,
    onHistoryDown,
    dismissSearchHint,
    historyIndex
  } = useHistoryNavigation((nextInput: string, nextMode: string, nextPastes: Record<string, any>) => {
    handleInputChange(nextInput);
    onModeChange(nextMode);
    setPastedContents(nextPastes);
  }, input, pastedContents, setCursorOffset);

  useEffect(() => {
    if (isSearchingHistory) dismissSearchHint();
  }, [isSearchingHistory, dismissSearchHint]);

  function setSelectionMode(mode: "tasks" | "diff" | "none") {
    setTasksSelected(mode === "tasks");
    setDiffSelected(mode === "diff");
  }

  function handleHistoryUp() {
    if (suggestions.length > 1) return;
    if (appState.queuedCommands.some((entry: any) => isQueuedCommandMode(entry.mode))) {
      popQueuedCommand();
      return;
    }
    if (diffSelected) {
      if (Object.values(appState.tasks).filter((entry: any) => entry.status === "running").length > 0) setSelectionMode("tasks");
      else if (teamList.length > 0) {
        setTeamsSelected(true);
        setSelectionMode("none");
      } else setSelectionMode("none");
      return;
    }
    if (teamsSelected) {
      const runningTaskCount = Object.values(appState.tasks).filter((entry: any) => entry.status === "running").length;
      setTeamsSelected(false);
      setSelectionMode(runningTaskCount > 0 ? "tasks" : "none");
      return;
    }
    if (tasksSelected) {
      setSelectionMode("none");
      return;
    }
    if (isCursorBeforeFirstNewline && imagePastes.length > 0 && !pastedContentSelected) {
      setPastedContentSelected(true);
      setSelectedPastedIndex(imagePastes.length - 1);
      return;
    }
    onHistoryUp();
  }

  function handleHistoryDown() {
    if (suggestions.length > 1) return;
    if (pastedContentSelected) return;
    const runningTaskCount = Object.values(appState.tasks).filter((entry: any) => entry.status === "running").length;

    if (tasksSelected) {
      if (teamList.length > 0) {
        setTeamsSelected(true);
        setSelectionMode("none");
      }
      return;
    }
    if (teamsSelected) return;
    if (diffSelected) return;

    const didAdvance = onHistoryDown();
    const hasTeams = teamList.length > 0;
    if (didAdvance) {
      if (runningTaskCount > 0) {
        setSelectionMode("tasks");
        setTeamsSelected(false);
        if (!getSessionHints().hasSeenTasksHint) {
          updateSessionHints((previous: any) => {
            if (previous.hasSeenTasksHint === true) return previous;
            return { ...previous, hasSeenTasksHint: true };
          });
        }
      } else if (hasTeams) {
        setTeamsSelected(true);
        setSelectionMode("none");
      }
    }
  }

  const [suggestionsState, setSuggestionsState] = useState({
    suggestions: [],
    selectedSuggestion: -1,
    commandArgumentHint: undefined as any
  });

  const submitInput = useCallback(
    async (value: string, allowAutocomplete = false) => {
      if (tasksSelected || teamsSelected || diffSelected) return;
      if (value.trim() === "" && appState.promptSuggestion.text && appState.promptSuggestion.shownAt > 0) {
        markSuggestionAccepted();
        value = appState.promptSuggestion.text;
      }
      const hasImagePaste = Object.values(pastedContents).some((entry: any) => entry.type === "image");
      if (value.trim() === "" && !hasImagePaste) return;

      const onlyDirectorySuggestions =
        suggestionsState.suggestions.length > 0 &&
        suggestionsState.suggestions.every((entry: any) => entry.description === "directory");

      if (suggestionsState.suggestions.length > 0 && !allowAutocomplete && !onlyDirectorySuggestions) return;

      if (appState.promptSuggestion.text && appState.promptSuggestion.shownAt > 0) {
        logSuggestionOutcome(value);
      }

      removeNotification("stash-hint");
      await onSubmit(value, {
        setCursorOffset,
        clearBuffer: clearUndoBuffer,
        resetHistory
      });
    },
    [
      appState.promptSuggestion,
      tasksSelected,
      teamsSelected,
      diffSelected,
      suggestionsState.suggestions,
      onSubmit,
      clearUndoBuffer,
      resetHistory,
      logSuggestionOutcome,
      markSuggestionAccepted,
      pastedContents,
      removeNotification
    ]
  );

  const {
    suggestions,
    selectedSuggestion,
    commandArgumentHint
  } = useCommandSuggestions({
    commands,
    onInputChange,
    onSubmit: submitInput,
    setCursorOffset,
    input,
    cursorOffset,
    mode,
    agents,
    setSuggestionsState,
    suggestionsState,
    suppressSuggestions: isSearchingHistory || historyIndex > 0,
    markAccepted: markSuggestionAccepted
  });

  const shouldShowInlineSuggestion = mode === "prompt" && suggestions.length === 0 && inlineSuggestion;
  if (shouldShowInlineSuggestion) markSuggestionShown();

  if (appState.promptSuggestion.text && !inlineSuggestion && appState.promptSuggestion.shownAt === 0) {
    trackSuggestionTiming("timing", appState.promptSuggestion.text);
    setAppState((current: any) => ({
      ...current,
      promptSuggestion: {
        text: null,
        promptId: null,
        shownAt: 0,
        acceptedAt: 0,
        generationRequestId: null
      }
    }));
  }

  function handleImagePaste(content: string, mediaType?: string, dimensions?: any) {
    trackEvent("tengu_paste_image", {});
    onModeChange("prompt");

    const nextId = nextAttachmentIdRef.current++;
    const attachment = {
      id: nextId,
      type: "image",
      content,
      mediaType: mediaType || "image/png",
      dimensions
    };

    registerAttachment(attachment);
    setTimeout(() => logAttachmentPaste(attachment), 0);

    setPastedContents((prev: any) => ({
      ...prev,
      [nextId]: attachment
    }));
  }

  function handleTextPaste(text: string) {
    const normalized = normalizePastedText(text).replace(/\r/g, "\n").replaceAll("\t", "    ");
    const lineCount = countLines(normalized);
    const maxLines = Math.min(rows - 10, 2);

    if (normalized.length > MAX_PASTE_SIZE || lineCount > maxLines) {
      const nextId = nextAttachmentIdRef.current++;
      const attachment = { id: nextId, type: "text", content: normalized };

      setPastedContents((prev: any) => ({
        ...prev,
        [nextId]: attachment
      }));

      insertAtCursor(formatPasteAsAttachment(nextId, lineCount));
    } else {
      insertAtCursor(normalized);
    }
  }

  function insertAtCursor(value: string) {
    pushUndoBuffer(input, cursorOffset, pastedContents);
    const nextValue = input.slice(0, cursorOffset) + value + input.slice(cursorOffset);
    onInputChange(nextValue);
    setCursorOffset(cursorOffset + value.length);
  }

  const showMessageSelector = createMessageSelectorHandler(() => { }, () => onShowMessageSelector());

  const popQueuedCommand = useCallback(async () => {
    const result = await popQueuedInput(
      input,
      cursorOffset,
      async () =>
        new Promise((resolve) =>
          setAppState((current: any) => {
            resolve(current);
            return current;
          })
        ),
      setAppState
    );

    if (!result) return false;
    onInputChange(result.text);
    onModeChange("prompt");
    setCursorOffset(result.cursorOffset);
    return true;
  }, [setAppState, onInputChange, onModeChange, input, cursorOffset]);

  useMcpAtMentionListener(mcpClients, function (atMention: any) {
    trackEvent("tengu_ext_at_mentioned", {});
    let insertion;
    const relativePath = pathUtils.relative(getWorkspaceRoot(), atMention.filePath);

    if (atMention.lineStart && atMention.lineEnd) {
      insertion =
        atMention.lineStart === atMention.lineEnd
          ? `@${relativePath}#L${atMention.lineStart} `
          : `@${relativePath}#L${atMention.lineStart}-${atMention.lineEnd} `;
    } else {
      insertion = `@${relativePath} `;
    }

    const charBeforeCursor = input[cursorOffset - 1] ?? " ";
    if (!/\s/.test(charBeforeCursor)) insertion = ` ${insertion}`;
    insertAtCursor(insertion);
  });

  useInputKeypress((key: any, keyData: any) => {
    if (showDiffDialog) return;

    if (getPlatformId() === "macos" && key in OPTION_META_HINTS) {
      const hint = OPTION_META_HINTS[key];
      addNotification({
        key: "option-meta-hint",
        jsx: (
          <Text dimColor>
            To enable {hint}, run /terminal-setup
          </Text>
        ),
        priority: "immediate",
        timeoutMs: 5000
      });
    }

    if (pastedContentSelected) {
      if (keyData.leftArrow) {
        setSelectedPastedIndex((prev) => (prev > 0 ? prev - 1 : imagePastes.length - 1));
        return;
      }
      if (keyData.rightArrow) {
        setSelectedPastedIndex((prev) => (prev < imagePastes.length - 1 ? prev + 1 : 0));
        return;
      }
      if (keyData.backspace || keyData.delete) {
        const selected = imagePastes[selectedPastedIndex];
        if (selected) {
          setPastedContents((prev: any) => {
            const next = { ...prev } as any;
            delete next[selected.id];
            return next;
          });
        }
        setPastedContentSelected(false);
        setSelectedPastedIndex(0);
        return;
      }
      if (keyData.downArrow || keyData.escape) {
        setPastedContentSelected(false);
        return;
      }
      return;
    }

    if (keyData.ctrl && key === "_") {
      if (canUndo) {
        trackShortcut("ctrl-underscore");
        const undoState = undoInput();
        if (undoState) {
          onInputChange(undoState.text);
          setCursorOffset(undoState.cursorOffset);
          setPastedContents(undoState.pastedContents);
        }
      }
      return;
    }

    if (keyData.ctrl && key.toLowerCase() === "g") {
      trackEvent("tengu_external_editor_used", {});
      trackShortcut("external-editor");
      setIsExternalEditorOpen(true);
      const edited = openExternalEditor(input);
      setIsExternalEditorOpen(false);
      if (edited !== null && edited !== input) {
        pushUndoBuffer(input, cursorOffset, pastedContents);
        onInputChange(edited);
        setCursorOffset(edited.length);
      }
      return;
    }

    if (keyData.ctrl && key.toLowerCase() === "s") {
      if (input.trim() === "" && stashedPrompt !== undefined) {
        onInputChange(stashedPrompt.text);
        setCursorOffset(stashedPrompt.cursorOffset);
        setStashedPrompt(undefined);
      } else if (input.trim() !== "") {
        setStashedPrompt({ text: input, cursorOffset });
        onInputChange("");
        setCursorOffset(0);
        trackShortcut("prompt-stash");
        updateSessionHints((current: any) => {
          if (current.hasUsedStash) return current;
          return { ...current, hasUsedStash: true };
        });
      }
      return;
    }

    if (keyData.return && tasksSelected) {
      setShowBashesDialog(true);
      setSelectionMode("none");
      return;
    }

    if (keyData.return && teamsSelected) {
      setDiffDialogOpen(true);
      setTeamsSelected(false);
      return;
    }

    if (tasksSelected && keyData.rightArrow) {
      if (teamList.length > 0) {
        setTasksSelected(false);
        setTeamsSelected(true);
        return;
      }
    }

    if (teamsSelected && keyData.leftArrow) {
      if (Object.values(appState.tasks).filter((entry: any) => entry.status === "running").length > 0) {
        setTeamsSelected(false);
        setTasksSelected(true);
        return;
      }
    }

    if (cursorOffset === 0 && (keyData.escape || keyData.backspace || keyData.delete)) {
      onModeChange("prompt");
      setHelpOpen(false);
    }

    if (helpOpen && input === "" && (keyData.backspace || keyData.delete)) {
      setHelpOpen(false);
    }

    if (modeCycleHotkey.check(key, keyData)) {
      const nextMode = getNextPromptMode(toolPermissionContext, appState.teamContext);
      trackEvent("tengu_mode_cycle", { to: nextMode });
      if (toolPermissionContext.mode === "plan" && nextMode !== "plan") setPlanModeExit(true);
      if (shouldPromptModeSwitch(toolPermissionContext.mode, nextMode) && toolPermissionContext.mode === "delegate" && nextMode !== "delegate") {
        setAutoAcceptMode(true);
        setEnableEditsToast(true);
      }
      if (nextMode === "plan") {
        updateSessionHints((current: any) => ({
          ...current,
          lastPlanModeUse: Date.now()
        }));
      }
      if (nextMode === "acceptEdits") trackShortcut("auto-accept-mode");

      const updated = applyModeChange(toolPermissionContext, {
        type: "setMode",
        mode: nextMode,
        destination: "session"
      });

      setToolPermissionContext(updated);
      if (helpOpen) setHelpOpen(false);
      return;
    }

    if (modelPickerHotkey.check(key, keyData)) {
      setShowModelPicker((prev) => !prev);
      if (helpOpen) setHelpOpen(false);
      return;
    }

    if (thinkingPickerHotkey.check(key, keyData)) {
      setShowThinkingPicker((prev) => !prev);
      if (helpOpen) setHelpOpen(false);
      return;
    }

    if (keyData.escape) {
      if (tasksSelected || teamsSelected || diffSelected) {
        setSelectionMode("none");
        setTeamsSelected(false);
        return;
      }
      if (appState.queuedCommands.some((entry: any) => isQueuedCommandMode(entry.mode))) {
        popQueuedCommand();
        return;
      }
      if (messages.length > 0 && !input && !isLoading) showMessageSelector();
    }

    if (keyData.return && helpOpen) setHelpOpen(false);
  });

  const { columns, rows } = useTerminalSize();
  const inputColumns = columns - 3;
  const activeBanner = getActiveBanner();

  const inputPlaceholder = (() => {
    if (!shouldShowInlineSuggestion || !inlineSuggestion) return promptStatusText;
    const suffix = INLINE_SUGGESTION_HINT;
    const suggestionLength = normalizePastedText(inlineSuggestion).length;
    const suffixLength = normalizePastedText(suffix).length;
    const padding = 3;
    const available = inputColumns;
    if (suggestionLength + suffixLength + padding > available) return inlineSuggestion;
    const spaceCount = available - suggestionLength - suffixLength;
    return inlineSuggestion + " ".repeat(spaceCount) + dimText(suffix);
  })();

  const isInputWrapped = useMemo(() => {
    const lines = input.split("\n");
    for (const line of lines) {
      if (line.length > inputColumns) return true;
    }
    return lines.length > 1;
  }, [input, inputColumns]);

  const handleModelSelection = useCallback(
    (modelId: string) => {
      setAppState((current: any) => ({
        ...current,
        mainLoopModel: modelId,
        mainLoopModelForSession: null
      }));
      setShowModelPicker(false);
      trackEvent("tengu_model_picker_hotkey", { model: modelId });
    },
    [setAppState]
  );

  const cancelModelSelection = useCallback(() => {
    setShowModelPicker(false);
  }, []);

  const modelPicker = useMemo(() => {
    if (!showModelPicker) return null;
    return (
      <Box flexDirection="column" marginTop={1}>
        <ModelPicker
          initial={appState.mainLoopModel}
          sessionModel={appState.mainLoopModelForSession}
          onSelect={handleModelSelection}
          onCancel={cancelModelSelection}
          isStandaloneCommand
        />
      </Box>
    );
  }, [showModelPicker, appState.mainLoopModel, appState.mainLoopModelForSession, handleModelSelection, cancelModelSelection]);

  const handleThinkingSelection = useCallback(
    (enabled: boolean) => {
      setAppState((current: any) => ({
        ...current,
        thinkingEnabled: enabled
      }));
      setShowThinkingPicker(false);
      trackEvent("tengu_thinking_toggled_hotkey", { enabled });
      addNotification({
        key: "thinking-toggled-hotkey",
        jsx: (
          <Text color={enabled ? "suggestion" : undefined} dimColor={!enabled}>
            Thinking {enabled ? "on" : "off"}
          </Text>
        ),
        priority: "immediate",
        timeoutMs: 3000
      });
    },
    [setAppState, addNotification]
  );

  const cancelThinkingSelection = useCallback(() => {
    setShowThinkingPicker(false);
  }, []);

  const thinkingPicker = useMemo(() => {
    if (!showThinkingPicker) return null;
    return (
      <Box flexDirection="column" marginTop={1}>
        <ThinkingModePicker
          currentValue={appState.thinkingEnabled}
          onSelect={handleThinkingSelection}
          onCancel={cancelThinkingSelection}
          isMidConversation={messages.some((entry: any) => entry.type === "assistant")}
        />
      </Box>
    );
  }, [showThinkingPicker, appState.thinkingEnabled, handleThinkingSelection, cancelThinkingSelection, messages.length]);

  if (showBashesDialog) {
    return (
      <BashDialog
        onDone={() => {
          setShowBashesDialog(false);
        }}
        toolUseContext={getToolUseContext(messages, [], new AbortController(), [], undefined, shellSnapshot)}
      />
    );
  }

  if (modelPicker) return modelPicker;
  if (thinkingPicker) return thinkingPicker;

  const inputProps = {
    multiline: true,
    onSubmit: submitInput,
    onChange: handleInputChange,
    value: historyMatch ? normalizeHistorySearchEntry(typeof historyMatch === "string" ? historyMatch : historyMatch.display) : input,
    onHistoryUp: handleHistoryUp,
    onHistoryDown: handleHistoryDown,
    onHistoryReset: resetHistory,
    placeholder: inputPlaceholder,
    onExit,
    onExitMessage: (message: string) => setExitMessage({ show: true, key: message }),
    onImagePaste: handleImagePaste,
    columns: inputColumns,
    disableCursorMovementForUpDownKeys: suggestions.length > 0,
    cursorOffset,
    onChangeCursorOffset: setCursorOffset,
    onPaste: handleTextPaste,
    onIsPastingChange: setIsPasting,
    focus: !isSearchingHistory && !pastedContentSelected,
    showCursor: !tasksSelected && !teamsSelected && !diffSelected && !isSearchingHistory && !pastedContentSelected,
    argumentHint: commandArgumentHint,
    onUndo: canUndo
      ? () => {
        const undoState = undoInput();
        if (undoState) {
          onInputChange(undoState.text);
          setCursorOffset(undoState.cursorOffset);
          setPastedContents(undoState.pastedContents);
        }
      }
      : undefined,
    highlights: inputHighlights
  };

  const getBorderColor = () => {
    const modeColorMap: Record<string, string> = {
      bash: "bashBorder",
      background: "background"
    };
    if (modeColorMap[mode]) return modeColorMap[mode];
    const customColor = getBorderColorOverride();
    if (customColor && KNOWN_BORDER_COLORS.includes(customColor)) return borderColorLookup[customColor];
    return "promptBorder";
  };

  if (isExternalEditorOpen) {
    return (
      <Box
        flexDirection="row"
        alignItems="center"
        justifyContent="center"
        borderColor={getBorderColor()}
        borderDimColor
        borderStyle="round"
        borderLeft={false}
        borderRight={false}
        borderBottom
        width="100%"
      >
        <Text dimColor italic>
          Save and close editor to continue...
        </Text>
      </Box>
    );
  }

  const inputElement = isVimInputEnabled() ? (
    <VimTerminalInput {...inputProps} initialMode={vimMode} onModeChange={setVimMode} isLoading={isLoading} />
  ) : (
    <TerminalInput {...inputProps} />
  );

  return (
    <Box flexDirection="column" marginTop={1} width="100%">
      <PreInputOverlays isLoading={isLoading} />
      <TopPromptAddon />
      <PromptStashIndicator hasStash={stashedPrompt !== undefined} />
      <PastedContentIndicator pastedContents={pastedContents} isSelected={pastedContentSelected} selectedIndex={selectedPastedIndex} />
      <Separator />
      <Box
        flexDirection="row"
        alignItems="flex-start"
        justifyContent="flex-start"
        paddingX={1}
        width="100%"
      >
        <PromptModeIndicator mode={mode} isLoading={isLoading} />
        <Box flexGrow={1} flexShrink={1}>
          {inputElement}
        </Box>
      </Box>
      <Separator />
      <TerminalStatusLine
        apiKeyStatus={apiKeyStatus}
        debug={debug}
        exitMessage={exitMessage}
        vimMode={vimMode}
        mode={mode}
        autoUpdaterResult={autoUpdaterResult}
        isAutoUpdating={isAutoUpdating}
        verbose={verbose}
        onAutoUpdaterResult={onAutoUpdaterResult}
        onChangeIsUpdating={setIsAutoUpdating}
        suggestions={suggestions}
        selectedSuggestion={selectedSuggestion}
        toolPermissionContext={toolPermissionContext}
        helpOpen={helpOpen}
        suppressHint={input.length > 0}
        tasksSelected={tasksSelected}
        teamsSelected={teamsSelected}
        diffSelected={diffSelected}
        ideSelection={ideSelection}
        mcpClients={mcpClients}
        isPasting={isPasting}
        isInputWrapped={isInputWrapped}
        messages={messages}
        isSearching={isSearchingHistory}
        historyQuery={historyQuery}
        setHistoryQuery={setHistoryQuery}
        historyFailedMatch={historyFailedMatch}
      />
    </Box >
  );
}
