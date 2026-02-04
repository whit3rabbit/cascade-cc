import { useState, useEffect } from 'react';

export interface TermSize {
    columns: number;
    rows: number;
}

/**
 * Hook to track terminal dimensions and respond to resize events.
 * Aligned with 2.1.19 gold reference (implied by StatusLine and Transcript needs).
 */
export function useTermSize(): TermSize {
    const [size, setSize] = useState<TermSize>({
        columns: process.stdout.columns || 80,
        rows: process.stdout.rows || 24
    });

    useEffect(() => {
        const handleResize = () => {
            setSize({
                columns: process.stdout.columns,
                rows: process.stdout.rows
            });
        };

        process.stdout.on('resize', handleResize);
        return () => {
            process.stdout.off('resize', handleResize);
        };
    }, []);

    return size;
}
