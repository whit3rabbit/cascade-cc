import React from 'react';

interface CostCommandProps {
    onDone: (message: string, options?: { display: 'system' }) => void;
}

export const CostCommand: React.FC<CostCommandProps> = ({
    onDone
}) => {
    React.useEffect(() => {
        (async () => {
            try {
                const { costService } = await import('../services/terminal/CostService.js');
                const usage = costService.getUsage();
                const totalCost = costService.calculateCost();

                onDone(`**Session Usage & Cost**
- **Tokens**: ${usage.inputTokens} in / ${usage.outputTokens} out
- **Cache**: ${usage.cacheReadTokens || 0} read / ${usage.cacheWriteTokens || 0} written
- **Estimated Cost**: **$${totalCost.toFixed(4)}**`, { display: 'system' });
            } catch (err: any) {
                onDone(`Error calculating cost: ${err.message}`, { display: 'system' });
            }
        })();
    }, []);

    return null;
};
