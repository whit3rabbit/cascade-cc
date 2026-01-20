
import crypto from 'crypto';

export function hashContent(content: string): string {
    return crypto.createHash('sha256').update(content).digest('hex');
}

export function generateToken(length: number = 32): string {
    return crypto.randomBytes(length).toString('hex');
}
