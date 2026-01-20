import React, { useCallback, useMemo, useState } from "react";
import { Box, Text, useInput } from "ink";
import { Select } from "@inkjs/ui";
import InkTextInput from "ink-text-input";
import path from "node:path";
import chalk from "chalk";
import { resolvePath } from "../../services/sandbox/pathResolver.js";
import { getFileSystem } from "../../utils/file-system/fileUtils.js";
import { useCtrlExit } from "../../hooks/useCtrlExit.js";

interface AddDirectoryProps {
    onAddDirectory: (path: string, remember: boolean) => void;
    onCancel: () => void;
    permissionContext: any;
    directoryPath?: string;
}

type AddDirectoryChoice = "yes-session" | "yes-remember" | "no";

const ADD_DIRECTORY_OPTIONS: { value: AddDirectoryChoice; label: string }[] = [
    { value: "yes-session", label: "Yes, for this session" },
    { value: "yes-remember", label: "Yes, and remember this directory" },
    { value: "no", label: "No" }
];

const TextInput = InkTextInput as unknown as React.FC<{
    value: string;
    onChange: (value: string) => void;
    onSubmit?: (value: string) => void;
    placeholder?: string;
    showCursor?: boolean;
    columns?: number;
    cursorOffset?: number;
    onChangeCursorOffset?: (offset: number) => void;
}>;

const SelectInput = Select as unknown as React.FC<{
    options: { value: AddDirectoryChoice; label: string }[];
    onChange: (value: AddDirectoryChoice) => void;
    onCancel?: () => void;
}>;

function AddDirectoryPermissionNote(): React.ReactElement {
    return (
        <Text dimColor>
            Claude Code will be able to read files in this directory and make edits when auto-accept edits is on.
        </Text>
    );
}

function AddDirectoryPathSummary({ path }: { path: string }): React.ReactElement {
    return (
        <Box flexDirection="column" paddingX={2} gap={1}>
            <Text color="permission">{path}</Text>
            <AddDirectoryPermissionNote />
        </Box>
    );
}

function AddDirectoryPathInput({
    value,
    onChange,
    onSubmit,
    error
}: {
    value: string;
    onChange: (value: string) => void;
    onSubmit: (value: string) => void;
    error: string | null;
}): React.ReactElement {
    return (
        <Box flexDirection="column">
            <Text>Enter the path to the directory:</Text>
            <Box borderDimColor borderStyle="round" marginY={1} paddingLeft={1}>
                <TextInput
                    showCursor={true}
                    placeholder="Directory path…"
                    value={value}
                    onChange={onChange}
                    onSubmit={onSubmit}
                    columns={80}
                    cursorOffset={value.length}
                    onChangeCursorOffset={() => {}}
                />
            </Box>
            {error && <Text color="error">{error}</Text>}
        </Box>
    );
}

function getWorkingDirectories(permissionContext: any): string[] {
    const workingDirectories = new Set<string>();
    const additional = permissionContext?.additionalWorkingDirectories;

    if (typeof permissionContext?.workingDirectory === "string") {
        workingDirectories.add(permissionContext.workingDirectory);
    }
    if (typeof permissionContext?.cwd === "string") {
        workingDirectories.add(permissionContext.cwd);
    }
    if (typeof permissionContext?.originalCwd === "string") {
        workingDirectories.add(permissionContext.originalCwd);
    }
    if (Array.isArray(permissionContext?.workingDirectories)) {
        for (const dir of permissionContext.workingDirectories) {
            if (dir) workingDirectories.add(dir);
        }
    }

    if (additional instanceof Map) {
        for (const dir of additional.keys()) {
            if (dir) workingDirectories.add(dir);
        }
    } else if (Array.isArray(additional)) {
        for (const dir of additional) {
            if (dir) workingDirectories.add(dir);
        }
    }

    if (Array.isArray(permissionContext?.additionalDirectories)) {
        for (const dir of permissionContext.additionalDirectories) {
            if (dir) workingDirectories.add(dir);
        }
    }

    if (workingDirectories.size === 0) {
        workingDirectories.add(process.cwd());
    }

    return Array.from(workingDirectories);
}

function isPathWithinDirectory(targetPath: string, baseDir: string): boolean {
    const resolvedTarget = path.resolve(targetPath);
    const resolvedBase = path.resolve(baseDir);
    const relativePath = path.relative(resolvedBase, resolvedTarget);

    return relativePath === "" || (!relativePath.startsWith("..") && !path.isAbsolute(relativePath));
}

function validateAddDirectoryPath(
    rawPath: string,
    permissionContext: any
):
    | { resultType: "emptyPath" }
    | { resultType: "pathNotFound"; directoryPath: string; absolutePath: string }
    | { resultType: "notADirectory"; directoryPath: string; absolutePath: string }
    | { resultType: "alreadyInWorkingDirectory"; directoryPath: string; workingDir: string }
    | { resultType: "success"; absolutePath: string } {
    if (!rawPath) {
        return { resultType: "emptyPath" };
    }

    const absolutePath = resolvePath(rawPath);
    const fileSystem = getFileSystem();

    if (!fileSystem.existsSync(absolutePath)) {
        return {
            resultType: "pathNotFound",
            directoryPath: rawPath,
            absolutePath
        };
    }

    if (!fileSystem.statSync(absolutePath).isDirectory()) {
        return {
            resultType: "notADirectory",
            directoryPath: rawPath,
            absolutePath
        };
    }

    for (const workingDir of getWorkingDirectories(permissionContext)) {
        if (isPathWithinDirectory(absolutePath, workingDir)) {
            return {
                resultType: "alreadyInWorkingDirectory",
                directoryPath: rawPath,
                workingDir
            };
        }
    }

    return { resultType: "success", absolutePath };
}

function formatAddDirectoryValidationMessage(
    result: ReturnType<typeof validateAddDirectoryPath>
): string {
    switch (result.resultType) {
        case "emptyPath":
            return "Please provide a directory path.";
        case "pathNotFound":
            return `Path ${chalk.bold(result.absolutePath)} was not found.`;
        case "notADirectory": {
            const parentDir = path.dirname(result.absolutePath);
            return `${chalk.bold(result.directoryPath)} is not a directory. Did you mean to add the parent directory ${chalk.bold(parentDir)}?`;
        }
        case "alreadyInWorkingDirectory":
            return `${chalk.bold(result.directoryPath)} is already accessible within the existing working directory ${chalk.bold(result.workingDir)}.`;
        case "success":
            return `Added ${chalk.bold(result.absolutePath)} as a working directory.`;
    }
}

export const AddDirectory: React.FC<AddDirectoryProps> = ({
    onAddDirectory,
    onCancel,
    permissionContext,
    directoryPath
}) => {
    const [inputValue, setInputValue] = useState("");
    const [error, setError] = useState<string | null>(null);
    const ctrlExit = useCtrlExit();
    const options = useMemo(() => ADD_DIRECTORY_OPTIONS, []);

    useInput(
        useCallback(
            (input, key) => {
                if (key.escape || (key.ctrl && input === "c")) {
                    onCancel();
                }
            },
            [onCancel]
        )
    );

    const handleSubmit = useCallback(
        (pathInput: string) => {
            const result = validateAddDirectoryPath(pathInput, permissionContext);
            if (result.resultType === "success") {
                onAddDirectory(result.absolutePath, false);
                return;
            }
            setError(formatAddDirectoryValidationMessage(result));
        },
        [permissionContext, onAddDirectory]
    );

    const handleChoice = useCallback(
        (value: AddDirectoryChoice) => {
            if (!directoryPath) return;
            switch (value) {
                case "yes-session":
                    onAddDirectory(directoryPath, false);
                    break;
                case "yes-remember":
                    onAddDirectory(directoryPath, true);
                    break;
                case "no":
                    onCancel();
                    break;
            }
        },
        [directoryPath, onAddDirectory, onCancel]
    );

    return (
        <Box flexDirection="column" gap={1}>
            <Box
                flexDirection="column"
                borderStyle="round"
                paddingLeft={1}
                paddingRight={1}
                gap={1}
                borderColor="permission"
            >
                <Text bold color="permission">Add directory to workspace</Text>

                {directoryPath ? (
                    <Box flexDirection="column" gap={1}>
                        <AddDirectoryPathSummary path={directoryPath} />
                        <SelectInput
                            options={options}
                            onChange={handleChoice}
                            onCancel={() => handleChoice("no")}
                        />
                    </Box>
                ) : (
                    <Box flexDirection="column" gap={1} marginX={2}>
                        <AddDirectoryPermissionNote />
                        <AddDirectoryPathInput
                            value={inputValue}
                            onChange={setInputValue}
                            onSubmit={handleSubmit}
                            error={error}
                        />
                    </Box>
                )}
            </Box>

            {!directoryPath && (
                <Box marginLeft={3}>
                    {ctrlExit.pending ? (
                        <Text dimColor>Press {ctrlExit.keyName} again to exit</Text>
                    ) : (
                        <Text dimColor>Enter to add · Esc to cancel</Text>
                    )}
                </Box>
            )}
        </Box>
    );
};
