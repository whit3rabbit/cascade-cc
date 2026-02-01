import React, { useState, useEffect } from 'react';
import { Box, Text, useInput } from 'ink';
import { Card } from '../common/Card.js';
import { detectProjectType, ProjectType } from '../../services/lsp/ProjectDetection.js';

interface LspRecommendationDialogProps {
    onDone: () => void;
}

export function LspRecommendationDialog({ onDone }: LspRecommendationDialogProps) {
    const [projectType, setProjectType] = useState<ProjectType | null>(null);
    const [status, setStatus] = useState<'scanning' | 'found' | 'skipped'>('scanning');

    useEffect(() => {
        const scan = async () => {
            const result = await detectProjectType(process.cwd());
            if (result) {
                setProjectType(result);
                setStatus('found');
            } else {
                onDone(); // Nothing found, skip
            }
        };
        scan();
    }, [onDone]);

    const handleInstall = () => {
        // In a real implementation, we would toggle the LSP enablement in settings
        console.log(`Enabling ${projectType?.recommendedLsp}...`);
        onDone();
    };

    useInput((input, key) => {
        if (status === 'found') {
            if (key.return) {
                handleInstall();
            }
            if (key.escape) {
                onDone();
            }
        }
    });

    if (status === 'scanning') {
        return <Text>Scanning project structure...</Text>;
    }

    if (status === 'found' && projectType) {
        return (
            <Card title="Language Server Recommendation" borderColor="magenta">
                <Box flexDirection="column" gap={1}>
                    <Text>
                        DeepMind Agent detected a <Text bold color="cyan">{projectType.description}</Text> ({projectType.configFile}).
                    </Text>
                    <Text>
                        Would you like to enable <Text bold color="green">{projectType.recommendedLsp}</Text> for better code intelligence?
                    </Text>
                    <Box gap={2} marginTop={1}>
                        <Text color="green" bold underline>
                            Enter to Enable
                        </Text>
                        <Text color="gray">
                            Esc to Skip
                        </Text>
                    </Box>
                </Box>
            </Card>
        );
    }

    return null;
}
