/**
 * File: src/components/permissions/usePermissionDialog.ts
 * Role: Hook for managing permission dialog state and logic.
 * Derived from chunk1513 (dPK) and chunk1524 logic.
 */

import { useState, useCallback, useMemo } from 'react';
import { useInput } from 'ink';

export interface PermissionOptionValue {
    type: 'accept-once' | 'accept-session' | 'accept-always' | 'reject';
    scope?: string;
    label: string;
    description?: string;
    feedbackConfig?: {
        type: 'accept' | 'reject';
        placeholder?: string;
    };
}

export interface PermissionToolConfirm {
    tool: {
        name: string;
        isMcp?: boolean;
    };
    input: any;
    assistantMessage: {
        message: {
            id: string;
        };
    };
    onAllow: (result: any, feedback?: string, options?: any) => void;
}

export interface UsePermissionDialogProps {
    filePath?: string;
    completionType?: string;
    languageName?: string;
    toolUseConfirm: PermissionToolConfirm;
    onDone: (result?: any) => void; // Using generic any for now
    onReject: () => void;
    parseInput: (input: any) => any;
    operationType?: string;
}

// Logic from FPK (inferred)
const getPermissionOptions = (
    { yesInputMode, noInputMode, toolPermissionContext }:
        { yesInputMode: boolean; noInputMode: boolean; toolPermissionContext: any }
): { option: PermissionOptionValue, label: string }[] => {
    // This would typically be dynamic based on context
    return [
        {
            option: { type: 'accept-once', label: 'Allow this time', feedbackConfig: { type: 'accept' } },
            label: 'Allow this time'
        },
        {
            option: { type: 'accept-session', scope: 'session', label: 'Allow for this session', feedbackConfig: { type: 'accept' } },
            label: 'Allow for this session'
        },
        {
            option: { type: 'accept-always', label: 'Always allow', feedbackConfig: { type: 'accept' } },
            label: 'Always allow'
        },
        {
            option: { type: 'reject', label: 'Reject', feedbackConfig: { type: 'reject' } },
            label: 'Reject'
        }
    ];
};

export const usePermissionDialog = (props: UsePermissionDialogProps) => {
    const {
        toolUseConfirm,
        onDone,
        onReject,
    } = props;

    const [selectedIndex, setSelectedIndex] = useState(0);

    const options = useMemo(() => [
        {
            option: { type: 'accept-once', label: 'Allow once', scope: 'once' },
            label: 'Allow once'
        },
        {
            option: { type: 'accept-session', label: 'Allow for this session', scope: 'session' },
            label: 'Allow for this session'
        },
        {
            option: { type: 'accept-always', label: 'Always allow', scope: 'always' },
            label: 'Always allow'
        },
        {
            option: { type: 'reject', label: 'Deny', scope: 'once' },
            label: 'Deny'
        }
    ], []);

    const onChange = useCallback((option: any) => {
        if (option.type === 'reject') {
            onReject();
        } else {
            toolUseConfirm.onAllow(toolUseConfirm.input, undefined, {
                scope: option.scope
            });
            onDone();
        }
    }, [onReject, toolUseConfirm, onDone]);

    return {
        options,
        onChange,
        selectedIndex,
        setSelectedIndex
    };
};
