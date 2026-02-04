import React from 'react';

interface DoctorCommandProps {
    onDone: (message: string, options?: { display: 'system' }) => void;
}

/**
 * DoctorCommand component for running system diagnostics in the UI.
 * Aligned with 2.1.19 gold reference (chunk1242/chunk1138).
 */
export const DoctorCommand: React.FC<DoctorCommandProps> = ({
    onDone
}) => {
    React.useEffect(() => {
        (async () => {
            try {
                const { DoctorService } = await import('../services/terminal/DoctorService.js');
                const info = await DoctorService.getDiagnosticInfo();

                let report = `## Diagnostics\n`;
                report += `└ Currently running: ${info.installationType} (${info.version})\n`;
                report += `${info.packageManager ? `└ Package manager: ${info.packageManager}\n` : ""}`;
                report += `└ Path: ${info.installationPath}\n`;
                report += `└ Invoked: ${info.invokedBinary}\n`;
                report += `└ Config install method: ${info.configInstallMethod}\n`;
                report += `└ Search (ripgrep): ${info.ripgrepStatus.workingDirectory ? "OK" : "Not working"} (${info.ripgrepStatus.mode === 'system' ? info.ripgrepStatus.systemPath || 'system' : info.ripgrepStatus.mode})\n`;

                report += `\n### Integration Status\n`;
                report += `└ Git: ${info.gitStatus.isRepo ? `OK (${info.gitStatus.originUrl || 'Local only'})` : 'Not in a git repo'}\n`;
                report += `└ GitHub CLI (gh): ${info.ghStatus.installed ? (info.ghStatus.authenticated ? `Authenticated (Scopes: ${info.ghStatus.scopes.join(", ")})` : 'Installed, but not authenticated') : 'Not found'}\n`;

                if (info.warnings.length > 0) {
                    report += `\n### ⚠️ Warnings\n`;
                    for (const w of info.warnings) {
                        report += `- **${w.issue}**\n  *Fix:* ${w.fix}\n`;
                    }
                }

                report += `\n### Environment Variables\n`;
                for (const ev of info.envVars) {
                    const statusIcon = ev.status === 'ok' ? '✅' : '❌';
                    report += `└ ${statusIcon} **${ev.name}**: ${ev.message}\n`;
                }

                onDone(report, { display: 'system' });
            } catch (err: any) {
                onDone(`Error running diagnostics: ${err.message}`, { display: 'system' });
            }
        })();
    }, [onDone]);

    return null;
};
