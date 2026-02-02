export function formatTranscript(messages: any[]): string {
    return messages.map(m => {
        const role = m.role.toUpperCase();
        const content = typeof m.content === 'string' ? m.content : JSON.stringify(m.content);
        return `[${role}]\n${content}\n`;
    }).join('\n---\n\n');
}
