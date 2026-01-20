
// Logic from chunk_560.ts (Guest Passes & Legal Notices)

import React, { useEffect, useState } from "react";
import { Box, Text, useInput } from "ink";
import { useCtrlExit } from "../../hooks/useCtrlExit.js";
import ExternalLink from "./ExternalLink.js";
import { copyToClipboard, getClipboardErrorMessage } from "./SessionResumeDashboard.js";
import { checkPassEligibility } from "./AppDashboard.js";

type GuestPass = {
    passNumber: number;
    isAvailable: boolean;
};

type PassEligibility = {
    eligible: boolean;
    referral_code_details?: { referral_link?: string };
};

type PassRedemptions = { redemptions?: Array<unknown | null>; limit?: number };

function trackEvent(_name: string, _payload?: Record<string, any>) {}
function logError(_error: unknown) {}

async function fetchPassRedemptions(): Promise<PassRedemptions> {
    return { redemptions: [], limit: 3 };
}

// --- Guest Pass Component (yY9) ---
export function GuestPassesView({ onDone }: { onDone: (message?: any, options?: any) => void }) {
    const [loading, setLoading] = useState(true);
    const [passes, setPasses] = useState<GuestPass[]>([]);
    const [eligible, setEligible] = useState(false);
    const [referralLink, setReferralLink] = useState<string | null>(null);
    const ctrlExit = useCtrlExit(() =>
        Promise.resolve(onDone("Guest passes dialog dismissed", { display: "system" }))
    );

    useInput((_input, key) => {
        if (key.escape) {
            onDone("Guest passes dialog dismissed", { display: "system" });
        }
        if (key.return && referralLink) {
            (async () => {
                if (await copyToClipboard(referralLink)) {
                    trackEvent("tengu_guest_passes_link_copied", {});
                    onDone("Referral link copied to clipboard!");
                } else {
                    onDone(getClipboardErrorMessage(), { display: "system" });
                }
            })();
        }
    });

    useEffect(() => {
        async function loadPasses() {
            try {
                const eligibility = (await checkPassEligibility()) as PassEligibility | null;
                if (!eligibility || !eligibility.eligible) {
                    setEligible(false);
                    setLoading(false);
                    return;
                }

                setEligible(true);
                if (eligibility.referral_code_details?.referral_link) {
                    setReferralLink(eligibility.referral_code_details.referral_link);
                }

                let redemptions: PassRedemptions;
                try {
                    redemptions = await fetchPassRedemptions();
                } catch (error) {
                    logError(error);
                    setEligible(false);
                    setLoading(false);
                    return;
                }

                const redemptionList = redemptions.redemptions || [];
                const limit = redemptions.limit || 3;
                const nextPasses: GuestPass[] = [];
                for (let index = 0; index < limit; index += 1) {
                    const entry = redemptionList[index];
                    nextPasses.push({
                        passNumber: index + 1,
                        isAvailable: !entry
                    });
                }
                setPasses(nextPasses);
                setLoading(false);
            } catch (error) {
                logError(error);
                setEligible(false);
                setLoading(false);
            }
        }
        loadPasses();
    }, []);

    if (loading) {
        return (
            <Box flexDirection="column" marginTop={1} gap={1}>
                <Text dimColor>Loading guest pass information…</Text>
                <Text dimColor italic>
                    {ctrlExit.pending ? `Press ${ctrlExit.keyName} again to exit` : "Esc to cancel"}
                </Text>
            </Box>
        );
    }

    if (!eligible) {
        return (
            <Box flexDirection="column" marginTop={1} gap={1}>
                <Text>Guest passes are not currently available.</Text>
                <Text dimColor italic>
                    {ctrlExit.pending ? `Press ${ctrlExit.keyName} again to exit` : "Esc to cancel"}
                </Text>
            </Box>
        );
    }

    const availableCount = passes.filter((pass) => pass.isAvailable).length;
    const sortedPasses = [...passes].sort((left, right) => Number(right.isAvailable) - Number(left.isAvailable));

    const renderTicket = (pass: GuestPass) => {
        if (!pass.isAvailable) {
            return (
                <Box key={pass.passNumber} flexDirection="column" marginRight={1}>
                    <Text dimColor>┌─────────╱</Text>
                    <Text dimColor> ) CC ✻ ┊╱</Text>
                    <Text dimColor>└───────╱</Text>
                </Box>
            );
        }

        return (
            <Box key={pass.passNumber} flexDirection="column" marginRight={1}>
                <Text>┌──────────┐</Text>
                <Text>
                    {" ) CC "}
                    <Text color="claude">✻</Text>
                    {" ┊ ( "}
                </Text>
                <Text>└──────────┘</Text>
            </Box>
        );
    };

    return (
        <Box flexDirection="column" marginTop={1} gap={1}>
            <Text color="permission">Guest passes · {availableCount} left</Text>
            <Box flexDirection="row" marginLeft={2}>
                {sortedPasses.map(renderTicket)}
            </Box>
            {referralLink && <Box marginLeft={2}><Text>{referralLink}</Text></Box>}
            <Box flexDirection="column" marginLeft={2}>
                <Text dimColor>Share a free week of Claude Code with friends.</Text>
            </Box>
            <Box>
                <Text dimColor italic>
                    {ctrlExit.pending ? `Press ${ctrlExit.keyName} again to exit` : "Enter to copy link · Esc to cancel"}
                </Text>
            </Box>
        </Box>
    );
}

// --- Legal Notice Visibility (fY9) ---
export function shouldShowGroveNotice(
    accountInfo: { grove_enabled?: boolean | null; grove_notice_viewed_at?: string | null } | null,
    noticeInfo: { notice_is_grace_period?: boolean | null; notice_reminder_frequency?: number | null } | null,
    forceShow: boolean
) {
    if (accountInfo !== null && accountInfo.grove_enabled !== null) return false;
    if (forceShow) return true;
    if (noticeInfo !== null && !noticeInfo.notice_is_grace_period) return true;

    const reminderFrequency = noticeInfo?.notice_reminder_frequency;
    if (reminderFrequency !== null && reminderFrequency !== undefined && accountInfo?.grove_notice_viewed_at) {
        const lastViewed = new Date(accountInfo.grove_notice_viewed_at).getTime();
        return Math.floor((Date.now() - lastViewed) / 86400000) >= reminderFrequency;
    }

    const lastViewed = accountInfo?.grove_notice_viewed_at;
    return lastViewed === null || lastViewed === undefined;
}

// --- Term Updates Notice (VG7 / HG7) ---
export function UpcomingTermsNotice() {
    return (
        <>
            <Box flexDirection="column">
                <Text bold color="professionalBlue">Updates to Consumer Terms and Policies</Text>
                <Text>
                    An update to our Consumer Terms and Privacy Policy will take effect on{" "}
                    <Text bold>October 8, 2025</Text>. You can accept the updated terms today.
                </Text>
            </Box>
            <Box flexDirection="column">
                <Text>What's changing?</Text>
                <Box paddingLeft={1}>
                    <Text>
                        <Text>• </Text>
                        <Text bold>You can help improve Claude </Text>
                        <Text>
                            — Allow the use of your chats and coding sessions to train and improve Anthropic AI models.
                            Change anytime in your Privacy Settings (
                            <ExternalLink url="https://claude.ai/settings/data-privacy-controls" />
                            ).
                        </Text>
                    </Text>
                </Box>
                <Box paddingLeft={1}>
                    <Text>
                        <Text>• </Text>
                        <Text bold>Updates to data retention </Text>
                        <Text>
                            — To help us improve our AI models and safety protections, we're extending data retention to
                            5 years.
                        </Text>
                    </Text>
                </Box>
            </Box>
            <Text>
                Learn more (
                <ExternalLink url="https://www.anthropic.com/news/updates-to-our-consumer-terms" />
                ) or read the updated Consumer Terms (
                <ExternalLink url="https://anthropic.com/legal/terms" />
                ) and Privacy Policy (
                <ExternalLink url="https://anthropic.com/legal/privacy" />
                )
            </Text>
        </>
    );
}

export function EffectiveTermsNotice() {
    return (
        <>
            <Box flexDirection="column">
                <Text bold color="professionalBlue">Updates to Consumer Terms and Policies</Text>
                <Text>We've updated our Consumer Terms and Privacy Policy.</Text>
            </Box>
            <Box flexDirection="column" gap={1}>
                <Text>What's changing?</Text>
                <Box flexDirection="column">
                    <Text bold>Help improve Claude</Text>
                    <Text>
                        Allow the use of your chats and coding sessions to train and improve Anthropic AI models. You
                        can change this anytime in Privacy Settings
                    </Text>
                    <ExternalLink url="https://claude.ai/settings/data-privacy-controls" />
                </Box>
                <Box flexDirection="column">
                    <Text bold>How this affects data retention</Text>
                    <Text>
                        Turning ON the improve Claude setting extends data retention from 30 days to 5 years. Turning it
                        OFF keeps the default 30-day data retention. Delete data anytime.
                    </Text>
                </Box>
            </Box>
            <Text>
                Learn more (
                <ExternalLink url="https://www.anthropic.com/news/updates-to-our-consumer-terms" />
                ) or read the updated Consumer Terms (
                <ExternalLink url="https://anthropic.com/legal/terms" />
                ) and Privacy Policy (
                <ExternalLink url="https://anthropic.com/legal/privacy" />
                )
            </Text>
        </>
    );
}
