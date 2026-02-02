import React from 'react';

interface DoctorCommandProps {
    onDone: (message: string, options?: { display: 'system' }) => void;
}

export const DoctorCommand: React.FC<DoctorCommandProps> = ({
    onDone
}) => {
    React.useEffect(() => {
        (async () => {
            try {
                const { DoctorService } = await import('../services/terminal/DoctorService.js');
                const results = await DoctorService.getDiagnostics();

                let report = "**System Health Diagnostics**\n\n";
                for (const res of results.healthChecks) {
                    const icon = res.status === 'ok' ? '✅' : (res.status === 'warn' ? '⚠️' : '❌');
                    report += `${icon} **${res.name}**: ${res.message}\n`;
                }

                onDone(report, { display: 'system' });
            } catch (err: any) {
                onDone(`Error running diagnostics: ${err.message}`, { display: 'system' });
            }
        })();
    }, []);

    return null;
};
