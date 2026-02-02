import React from 'react';
import { PromptManager } from '../services/conversation/PromptManager.js';

interface CompactCommandProps {
    messages: any[];
    setMessages: (updater: (prev: any[]) => any[]) => void;
    setIsTyping: (isTyping: boolean) => void;
    onDone: (message: string, options?: { display: 'system' }) => void;
}

export const CompactCommand: React.FC<CompactCommandProps> = ({
    messages,
    setMessages,
    setIsTyping,
    onDone
}) => {
    React.useEffect(() => {
        setIsTyping(true);
        PromptManager.compactMessages(messages, { model: "claude-3-5-sonnet-20241022" })
            .then(compacted => {
                setMessages(() => compacted);
                onDone('Conversation history compacted.', { display: 'system' });
            })
            .catch(err => {
                onDone(`Failed to compact conversation: ${err.message}`, { display: 'system' });
            })
            .finally(() => setIsTyping(false));
    }, []);

    return null;
};
