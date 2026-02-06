/**
 * File: src/services/teams/MailboxPollingService.ts
 * Role: Periodically checks the teammate mailbox and triggers notifications for new messages.
 */

import { readUnreadMessages, markMessagesAsRead } from './TeammateMailbox.js';
import { notificationQueue } from '../terminal/NotificationService.js';
import { EnvService } from '../config/EnvService.js';

export class MailboxPollingService {
    private static interval: NodeJS.Timeout | null = null;
    private static isPolling = false;

    /**
     * Starts the background polling interval.
     */
    static start(intervalMs: number = 3000) {
        if (this.interval) return;

        const agentName = EnvService.get("CLAUDE_CODE_AGENT_NAME");
        const teamName = EnvService.get("CLAUDE_CODE_TEAM_NAME");

        if (!agentName || !teamName) return;

        this.interval = setInterval(() => {
            this.poll(agentName, teamName);
        }, intervalMs);
    }

    /**
     * Stops the background polling interval.
     */
    static stop() {
        if (this.interval) {
            clearInterval(this.interval);
            this.interval = null;
        }
    }

    /**
     * Performs a single poll of the mailbox.
     */
    private static async poll(agentName: string, teamName: string) {
        if (this.isPolling) return;
        this.isPolling = true;

        try {
            const unread = readUnreadMessages(agentName, teamName);
            if (unread.length > 0) {
                for (const msg of unread) {
                    notificationQueue.add({
                        text: `New message from ${msg.from}: ${msg.text.slice(0, 50)}${msg.text.length > 50 ? '...' : ''}`,
                        type: 'info',
                        timeoutMs: 5000
                    });
                }
                // Mark as read so we don't notify multiple times
                markMessagesAsRead(agentName, teamName);
            }
        } catch {
            // Silently fail polling errors to avoid spamming the terminal
        } finally {
            this.isPolling = false;
        }
    }
}
